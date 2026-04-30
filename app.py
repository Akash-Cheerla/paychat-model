"""
FYOE Multi-Intent Detection API (v3 super-app)

Detects one or more actionable intents in a chat message and returns
structured payloads + ready-to-fire deep links per intent.

v3 intents (18, see training/v3_intents.py):
  money | alarm | contact | calendar | maps |
  food_order | ride | travel | shopping | music | video | tickets |
  reservation | task | note | bills | health | weather

The API supports legacy 2-label (money-only) and 5-label models too — any
intents the loaded model wasn't trained on simply default to 0.0 confidence
and never fire, so we can ship the API ahead of the model swap.

Back-compat: flat is_money / should_popup / trigger_type / etc. are populated
from the money intent (if any) so the existing Android build keeps working.

Endpoints:
  POST /detect                   single message inference (chat_id enables multi-turn context)
  WS   /ws/detect                real-time stream
  GET  /health                   health + model version
  GET  /metrics                  live stats
  POST /reload                   hot-reload model from disk
  POST /payment-complete/{chat_id}    signal a payment landed (money only)
  POST /popup-dismissed/{chat_id}     signal user dismissed the popup
  POST /reset-cooldown/{chat_id}      force-clear cooldowns for any intent
  GET  /chat-state/{chat_id}          inspect tracker (per-intent)
"""

import json
import os
import re
import sys
import time
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    # legacy aliases kept so any older code paths importing them still work
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

# Make sibling training/v3_intents.py importable regardless of CWD.
_TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training")
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)
try:
    from v3_intents import INTENTS_V3, build_action_url  # type: ignore
except Exception:  # pragma: no cover — fall back if module missing
    INTENTS_V3 = ["money", "alarm", "contact", "calendar", "maps"]
    def build_action_url(intent, payload, targeting=None):  # type: ignore
        return None

# Optional: dateparser is in requirements.txt but guard against import failure so
# the server still boots if it's missing in dev.
try:
    import dateparser
    from dateparser.search import search_dates as _dp_search_dates
    _HAS_DATEPARSER = True
except Exception:  # pragma: no cover
    dateparser = None
    _dp_search_dates = None
    _HAS_DATEPARSER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paychat")


# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────
MODEL_DIR            = Path(os.getenv("MODEL_DIR", "./saved_model"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# Per-intent threshold overrides. Loaded from a layered cascade:
#   1. Hard-coded conservative defaults (this file).
#   2. saved_model/thresholds.json if present (written by train.py after
#      tuning F1 per intent on the val set -- this is the *learned* value).
#   3. INTENT_THRESHOLDS_JSON env var (operator override; trumps both).
# Later layers fully override earlier ones for a given intent.
def _parse_thresholds_env() -> Dict[str, float]:
    raw = os.getenv("INTENT_THRESHOLDS_JSON")
    if not raw:
        return {}
    try:
        d = json.loads(raw)
        return {str(k): float(v) for k, v in d.items()}
    except Exception:
        logger.warning(f"INTENT_THRESHOLDS_JSON not parseable: {raw!r}")
        return {}


def _parse_thresholds_file() -> Dict[str, float]:
    path = MODEL_DIR / "thresholds.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            d = json.load(f)
        return {str(k): float(v) for k, v in d.items()}
    except Exception as e:
        logger.warning(f"{path} not parseable: {e}")
        return {}


PER_INTENT_THRESHOLDS: Dict[str, float] = {
    # Conservative -- false positives are user-visible interrupts.
    # Contact stays at 0.55 because false "save contact" popups are jarring
    # and the v2 model is well-calibrated on contact phrasing. Money sits at
    # 0.50 to keep recall on edge phrasings like "split the bill" / "owe you"
    # where the model floats around 0.50-0.55 with no $ amount or app token.
    "contact":     0.55,
    "money":       0.50,
    "calendar":    0.50,
    "alarm":       0.50,
    # Balanced
    "maps":        0.50,
    "food_order":  0.50,
    "ride":        0.50,
    "travel":      0.50,
    "shopping":    0.50,
    "reservation": 0.50,
    "tickets":     0.50,
    "bills":       0.50,
    "health":      0.50,
    # Looser -- these just open a search page; recall > precision
    "music":       0.45,
    "video":       0.45,
    "task":        0.45,
    "note":        0.45,
    "weather":     0.45,
}
# Layer 2: learned thresholds from training (per-intent F1-optimal on val set).
PER_INTENT_THRESHOLDS.update(_parse_thresholds_file())
# Layer 3: explicit operator override via env.
PER_INTENT_THRESHOLDS.update(_parse_thresholds_env())


def _threshold_for(intent: str) -> float:
    return PER_INTENT_THRESHOLDS.get(intent, CONFIDENCE_THRESHOLD)

# Popup anti-spam windows. Stateless model stays dumb; the API holds UX policy.
POPUP_COOLDOWN_SECONDS     = int(os.getenv("POPUP_COOLDOWN_SECONDS",     "300"))
DISMISSED_COOLDOWN_SECONDS = int(os.getenv("DISMISSED_COOLDOWN_SECONDS", "900"))
POST_PAYMENT_GRACE_SECONDS = int(os.getenv("POST_PAYMENT_GRACE_SECONDS", "60"))
TRACKER_EVICTION_SECONDS   = int(os.getenv("TRACKER_EVICTION_SECONDS",   "1800"))

MAX_LEN = 128
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Canonical v3 intent list. Order is the same the trainer uses.
INTENTS = list(INTENTS_V3)
LEGACY_5_INTENTS = ["money", "alarm", "contact", "calendar", "maps"]

# Multi-turn chat history cap (per chat_id). Older messages slide off.
CHAT_HISTORY_LEN = int(os.getenv("CHAT_HISTORY_LEN", "4"))
# Whether to prepend prior turns to the model input. Off by default — flipping
# this on makes terse follow-ups ("yeah do it", "sure") inherit the prior turn's
# intent. Cooldowns still gate the popup, so spam is prevented either way.
USE_CHAT_CONTEXT = os.getenv("USE_CHAT_CONTEXT", "0").lower() in ("1", "true", "yes")


# ─────────────────────────────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────────────────────────────
model_state = {
    "model": None,
    "tokenizer": None,
    "num_labels": None,         # 2 = legacy money-only, 5 = multi-intent
    "label_order": None,        # order of intents in model output logits
    "version": None,
    "loaded_at": None,
}

stats = {
    "requests": 0,
    "money_detected": 0,
    "intents_detected": {i: 0 for i in INTENTS},
    "popups_fired": 0,
    "popups_suppressed": 0,
    "started_at": datetime.utcnow().isoformat(),
    "avg_latency_ms": 0.0,
    "_latency_sum": 0.0,
}

# Per-(chat_id, intent) popup tracker. Keys are tuples like ("room_abc", "money").
# Each value looks like:
#   {
#     "state":           "cooldown" | "dismissed" | "post_payment" | "idle",
#     "last_popup_ts":   float,
#     "last_event_ts":   float,
#     "last_payload":    dict | None,   # last payload we fired on (for dedupe)
#     "popup_count":     int,
#     "suppression_count": int,
#     "reason_for_current_state": str,
#   }
popup_tracker: Dict[Tuple[str, str], dict] = {}

# Per-chat rolling history (multi-turn context). Keys are chat_ids; values are
# deques of (ts, sender, text) tuples. Used to give the classifier a tiny bit
# of context so "yeah let's do it" inherits the prior message's intent.
chat_history: Dict[str, "deque[Tuple[float, Optional[str], str]]"] = {}


# ─────────────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────────────
def load_model(model_dir: Path = MODEL_DIR):
    """Load or hot-swap the model from disk.

    Supports:
      - 2-label  legacy money-only classifier (softmax)
      - 5-label  v2 multi-intent (money/alarm/contact/calendar/maps)
      - 18-label v3 super-app (all v3 intents)
      - any other num_labels — derived from id2label if available
    """
    logger.info(f"Loading model from {model_dir}")

    # Auto* picks the right architecture from config.json (DistilBERT for v1/v2,
    # RoBERTa for v3, DeBERTa-v3 if we ever swap, etc.) — no need to hard-code.
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model = model.to(DEVICE)
    model.eval()
    logger.info(
        f"Loaded {type(model).__name__} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)"
    )

    num_labels = model.config.num_labels

    # Derive intent order: trust id2label if present, else fall back to defaults.
    id2label = getattr(model.config, "id2label", None) or {}
    # id2label may come back as str keys (HF serialization quirk), normalize.
    try:
        id2label_norm = {int(k): v for k, v in id2label.items()}
    except (TypeError, ValueError):
        id2label_norm = id2label

    def _has_real_labels(order: List[str]) -> bool:
        return order and not any(str(lbl).startswith("LABEL_") for lbl in order)

    if num_labels == 2:
        label_order = ["not_money", "money"]
    elif num_labels == 5:
        label_order = [id2label_norm.get(i, LEGACY_5_INTENTS[i]) for i in range(5)]
        if not _has_real_labels(label_order):
            label_order = list(LEGACY_5_INTENTS)
    elif num_labels == len(INTENTS_V3):
        label_order = [id2label_norm.get(i, INTENTS_V3[i]) for i in range(num_labels)]
        if not _has_real_labels(label_order):
            label_order = list(INTENTS_V3)
    else:
        # Unknown size — try id2label; if that fails, take the first N from the v3 list.
        label_order = [id2label_norm.get(i) for i in range(num_labels)]
        if not _has_real_labels(label_order):
            label_order = (INTENTS_V3 + [f"label_{i}" for i in range(num_labels)])[:num_labels]
        logger.warning(
            f"Unusual num_labels={num_labels}. Derived label_order={label_order}"
        )

    version = None
    report_path = model_dir / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        version = {
            "trained_at":       report.get("trained_at"),
            "test_accuracy":    report.get("test_accuracy") or report.get("test_exact_match"),
            "test_f1":          report.get("test_f1"),
            "test_exact_match": report.get("test_exact_match"),
            "test_hamming":     report.get("test_hamming"),
            "per_intent":       report.get("per_intent"),
            "intents":          report.get("intents"),
        }

    model_state.update({
        "model": model,
        "tokenizer": tokenizer,
        "num_labels": num_labels,
        "label_order": label_order,
        "version": version,
        "loaded_at": datetime.utcnow().isoformat(),
    })

    logger.info(
        f"Model loaded | num_labels={num_labels} | intents={label_order} | "
        f"device={DEVICE}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ─────────────────────────────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FYOE Multi-Intent Detection API",
    description=(
        "Detects multiple actionable intents per chat message (money, alarm, "
        "contact, calendar, maps, food_order, ride, travel, shopping, music, "
        "video, tickets, reservation, task, note, bills, health, weather). "
        "Each fired intent ships with a structured payload + a one-tap "
        "action_url deep link. Multi-turn chat history is used for context."
    ),
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═════════════════════════════════════════════════════════════════════
#  PAYLOAD EXTRACTORS
# ═════════════════════════════════════════════════════════════════════

# ─── Money: amount ───
_AMOUNT_RE = re.compile(
    r'\$[\d,]+(?:\.\d{1,2})?'
    r'|\b\d+\s*\$'
    r'|\b\d+\s*(?:dollars?|bucks?)\b',
    re.IGNORECASE,
)


def _extract_amount(text: str) -> Optional[str]:
    m = _AMOUNT_RE.search(text)
    if not m:
        return None
    amount = m.group(0)
    if re.match(r'^\d+\s*\$', amount):
        amount = '$' + amount.replace('$', '').strip()
    return amount


# ─── Contact: phone ───
# US formats: +1 415-555-1234, (415) 555-1234, 415.555.1234, 4155551234, 1-415-555-1234
# India: +91 98765 43210, +91-98765-43210, 9876543210 (starts 6-9)
_PHONE_US_RE = re.compile(
    r'(?:\+?1[\s\-\.]?)?'           # optional +1
    r'\(?\d{3}\)?[\s\-\.]?'         # area code
    r'\d{3}[\s\-\.]?\d{4}'          # 7-digit local
)
_PHONE_IN_RE = re.compile(
    r'(?:\+?91[\s\-]?)?'            # optional +91
    r'[6-9]\d{4}[\s\-]?\d{5}'       # 10 digits starting 6-9
)

# Things that look phone-y but aren't: credit-card 4-digits, order IDs, etc.
_PHONE_CONTEXT_STOPWORDS = {
    "card", "credit", "debit", "order", "tracking", "invoice",
    "pin", "passport", "zip", "room", "flight", "ticket",
    "build", "version", "score", "year",
}


def _extract_phone(text: str) -> Optional[str]:
    """Extract first phone number that doesn't look like an ID/serial."""
    t_lower = text.lower()

    # Skip if the line is clearly talking about a non-phone number
    for word in _PHONE_CONTEXT_STOPWORDS:
        if word in t_lower:
            # Only skip if the stopword is *near* the digits
            m = re.search(r'\d', t_lower)
            if m:
                idx = m.start()
                window = t_lower[max(0, idx - 30): idx + 30]
                if word in window:
                    return None

    # Try India first (more specific)
    m = _PHONE_IN_RE.search(text)
    if m:
        raw = m.group(0)
        digits = re.sub(r'\D', '', raw)
        if len(digits) == 10 and digits[0] in "6789":
            return f"+91 {digits[:5]} {digits[5:]}"
        if len(digits) == 12 and digits.startswith("91"):
            rest = digits[2:]
            return f"+91 {rest[:5]} {rest[5:]}"

    # Then US
    m = _PHONE_US_RE.search(text)
    if m:
        raw = m.group(0)
        digits = re.sub(r'\D', '', raw)
        if len(digits) == 10:
            return f"+1 {digits[:3]} {digits[3:6]} {digits[6:]}"
        if len(digits) == 11 and digits.startswith("1"):
            rest = digits[1:]
            return f"+1 {rest[:3]} {rest[3:6]} {rest[6:]}"

    return None


# ─── Alarm / Calendar: time + label ───
_RELATIVE_DATES = {
    "tomorrow", "tmrw", "today", "tonight",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "noon", "midnight", "morning", "afternoon", "evening",
}


# dateparser.search_dates is liberal — it'll match 2-letter words like "me" or
# "on" as dates. We post-filter: matched phrase must either contain a digit or
# a known date/time word. This kills 99% of false positives without hurting recall.
_TIME_WORDS = {
    "tomorrow", "tmrw", "today", "tonight", "yesterday",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "tues", "wed", "thu", "thurs", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "noon", "midnight", "morning", "afternoon", "evening", "night",
    "am", "pm", "a.m.", "p.m.",
    "hour", "hours", "hr", "hrs", "min", "mins", "minute", "minutes",
    "week", "weeks", "weekend", "month", "months",
    "next", "this", "last",
}


def _is_valid_time_match(phrase: str) -> bool:
    """Reject dateparser garbage like 'me', 'set', 'on', and phone-number-like phrases."""
    if not phrase or len(phrase) < 3:
        return False
    p = phrase.strip()
    low = p.lower()

    # ── Phone-number rejection ──────────────────────────────────────
    # Country-code style: "+91", "+1", "+44 1234..." — these were being parsed as years.
    if re.match(r"^\+\d", p):
        return False
    # Long digit runs (7+ consecutive digits) — phone numbers, not dates.
    if re.search(r"\d{7,}", p):
        return False

    # Must have a digit OR a known time word
    if any(c.isdigit() for c in low):
        return True
    for w in _TIME_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", low):
            return True
    return False


# Normalize "10 a.m." / "10 a.m" / "10 P.M." → "10am" / "10pm" before dateparser sees it.
# dateparser's tokenizer often chokes on the periods, returning no match.
_AMPM_NORMALIZE_RE = re.compile(r"\b([apAP])\.?\s*([mM])\.?(?=\W|$)")

# "5 in the morning" / "7 in the evening" / "9 in the night" → "5am" / "7pm" / "9pm"
# dateparser doesn't grok the "N in the X" idiom and silently drops the time.
_TIME_OF_DAY_RE = re.compile(
    r"\b(\d{1,2})(?:\s*:\s*(\d{2}))?\s+in\s+the\s+(morning|afternoon|evening|night)\b",
    re.IGNORECASE,
)


def _time_of_day_repl(m: "re.Match") -> str:
    h = m.group(1)
    mn = m.group(2)
    period = m.group(3).lower()
    suffix = "am" if period == "morning" else "pm"
    return f"{h}:{mn}{suffix}" if mn else f"{h}{suffix}"


def _normalize_for_datetime(text: str) -> str:
    """Tighten common ASR/typing artifacts before feeding dateparser."""
    out = _AMPM_NORMALIZE_RE.sub(lambda m: m.group(1).lower() + m.group(2).lower(), text)
    # "5 in the morning" → "5am"
    out = _TIME_OF_DAY_RE.sub(_time_of_day_repl, out)
    # Collapse "10 : 45" or "10 :45" → "10:45"
    out = re.sub(r"(\d)\s*:\s*(\d)", r"\1:\2", out)
    return out


# dateparser sometimes drops or mis-parses the time portion of a phrase
# like "the 27th at 10 a.m" or "wake me up at 6am tomorrow" — we get the
# date right but the time wrong. If the matched phrase clearly contains an
# explicit clock time (e.g. "at 10am", "10:45 pm", "6am"), patch hour/minute
# to match what the user actually said.
_EXPLICIT_TIME_RE = re.compile(
    r"(?:\bat\s+)?\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
    re.IGNORECASE,
)

# Words/patterns that anchor a date to a specific day. If the phrase
# contains any of these, dateparser's date is trustworthy. Otherwise
# (bare "5am" with no day context), we force date = today (or tomorrow
# if the hour has already passed).
_DATE_CONTEXT_RE = re.compile(
    r"\b(?:tomorrow|today|tonight|tmrw|yesterday|"
    r"mon|tue|wed|thu|fri|sat|sun|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december|"
    r"next|this|last|"
    r"\d{1,2}(?:st|nd|rd|th)|"  # ordinals like "27th"
    r"\d{1,2}/\d{1,2}|"          # explicit dates like 4/28
    r"\d{4})\b",
    re.IGNORECASE,
)


def _patch_iso_hour(iso: str, phrase: str) -> str:
    m = _EXPLICIT_TIME_RE.search(phrase)
    if not m:
        return iso
    hour = int(m.group(1)) % 12
    if m.group(3).lower() == "pm":
        hour += 12
    minute = int(m.group(2) or 0)

    # If the phrase has NO real date anchor (no day-name, month, ordinal,
    # etc.), dateparser sometimes misreads the hour digit as month-of-year
    # (e.g. "5am" → 2026-05-DD). Force the date to today/tomorrow.
    if not _DATE_CONTEXT_RE.search(phrase):
        now = datetime.now()
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate = candidate + timedelta(days=1)
        return candidate.strftime("%Y-%m-%dT") + f"{hour:02d}:{minute:02d}"

    return iso[:11] + f"{hour:02d}:{minute:02d}"


def _parse_datetime(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Pull the first valid date/time phrase out of text.

    Returns: (iso_string, raw_phrase). Both None if nothing parseable found.
    """
    if not _HAS_DATEPARSER:
        return None, None
    try:
        normalized = _normalize_for_datetime(text)
        results = _dp_search_dates(
            normalized,
            languages=["en"],
            settings={
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": False,
                "PARSERS": ["relative-time", "absolute-time", "timestamp"],
            },
        )
        if not results:
            return None, None
        # Take the first valid (non-garbage) match
        for raw_phrase, dt in results:
            if _is_valid_time_match(raw_phrase):
                iso = dt.isoformat(timespec="minutes")
                iso = _patch_iso_hour(iso, raw_phrase)
                return iso, raw_phrase
        return None, None
    except Exception as e:
        logger.debug(f"dateparser failed: {e}")
        return None, None


def _extract_duration_seconds(text: str) -> Optional[int]:
    """Catch 'in 20 min', 'in 2 hours' for alarms — anchored to *now*."""
    m = re.search(r'\bin\s+(\d+)\s*(min|mins|minutes|hour|hours|hr|hrs)\b', text, re.IGNORECASE)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("h"):
        return n * 3600
    return n * 60


def _strip_date_phrase(text: str, phrase: Optional[str]) -> str:
    if not phrase:
        return text
    # The phrase may have come from a normalized version of `text` (e.g.
    # "10 am" parsed from text containing "10 a.m"). Try the original first,
    # then normalize and try again so the strip succeeds either way.
    if phrase in text:
        out = text.replace(phrase, " ")
    else:
        normalized = _normalize_for_datetime(text)
        if phrase in normalized:
            out = normalized.replace(phrase, " ")
        else:
            out = text
    # Collapse runs of whitespace
    out = re.sub(r"\s+", " ", out)
    return out.strip(" ,.-")


def _extract_alarm_payload(text: str) -> Dict[str, Any]:
    """
    Alarm payload: { label, time_iso, time_phrase, seconds_from_now? }

    Tries two paths:
      1. Relative duration: 'in 20 min' -> seconds_from_now
      2. Absolute time: 'at 6am tomorrow' via dateparser -> time_iso
    """
    iso, phrase = _parse_datetime(text)
    seconds = _extract_duration_seconds(text)

    # Label = text stripped of the reminder verb + the time phrase
    label = text
    for verb in [
        "remind me to ", "remind me ", "set a reminder to ",
        "set a reminder for ", "set an alarm for ", "set an alarm to ",
        "set alarm for ", "wake me up ", "wake me ",
        "ping me at ", "ping me in ", "ping me when ",
        "notify me at ", "alert me at ", "buzz me at ",
        "don't let me forget to ", "note to self ",
        "reminder: ",
    ]:
        if label.lower().startswith(verb):
            label = label[len(verb):]
            break
    # Remove "at <time>" / "in <duration>" suffixes even if they're mid-sentence
    label = _strip_date_phrase(label, phrase)
    label = re.sub(r'\bat\s*$', '', label).strip(" ,.-")
    label = re.sub(r'\bin\s+\d+\s*(min\w*|hour\w*|hr\w*)\s*$', '', label).strip(" ,.-")

    payload = {"label": label or None}
    if iso:
        payload["time_iso"] = iso
        payload["time_phrase"] = phrase
    if seconds is not None:
        payload["seconds_from_now"] = seconds

    return payload


# ─── Calendar title helpers ─────────────────────────────────────────────
_CALENDAR_PREFIXES = [
    # mark / put / add to calendar
    "mark the event on the calendar ", "mark the event on my calendar ",
    "mark the event ", "mark an event ", "mark event ",
    "put the event on the calendar ", "put it on the calendar ",
    "put on the calendar ", "put on my calendar ",
    "add the event to the calendar ", "add the event to my calendar ",
    "add to the calendar ", "add to my calendar ", "add to calendar ",
    "add an event ", "add event ",
    # create / schedule
    "create an event ", "create event ", "create a calendar event ",
    "make an event ", "make event ",
    "schedule a meeting ", "schedule a call ", "schedule a sync ",
    "schedule the ", "schedule an ", "schedule a ", "schedule ",
    "let's schedule ", "lets schedule ",
    # set up / setup
    "i would like to set up ", "i'd like to set up ", "i want to set up ",
    "let me set up ", "let's set up ", "lets set up ",
    "set up a meeting ", "set up a call ", "set up an event ", "set up ",
    # book / block
    "book a ", "book the ", "book ", "block off ", "block my calendar for ",
    "block my calendar ",
    # save the date
    "pencil me in ", "save the date for ", "save the date ",
    "remind me of ", "remind me about ",
    # short
    "put ", "add ",
]

# Strip these *anywhere* (mid-string is fine) — they are calendar housekeeping noise.
_CALENDAR_NOISE_PATTERNS = [
    r"\bon\s+(?:the\s+)?calendar\b",
    r"\bin\s+(?:the\s+|my\s+)?calendar\b",
    r"\bto\s+(?:the\s+|my\s+)?calendar\b",
    r"\bfor\s+(?:the\s+|my\s+)?calendar\b",
    r"\bas\s+said\s+\w+",                  # "as said meeting"
    r"\bas\s+(?:a|an|the)\s+",              # "as a meeting" -> drop the "as a "
]

# If after cleanup the title is empty/junk, fall back to the event noun mentioned.
_EVENT_NOUNS = [
    "interview", "1:1", "1on1", "standup", "stand-up",
    "happy hour", "catch up", "catchup", "catch-up", "coffee chat",
    "meeting", "call", "sync", "conference", "review", "demo",
    "lunch", "dinner", "brunch", "breakfast", "drinks", "coffee",
    "appointment", "party", "session", "workshop", "presentation",
]


def _fallback_event_noun(text: str) -> Optional[str]:
    low = text.lower()
    for noun in sorted(_EVENT_NOUNS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(noun)}\b", low):
            return noun.capitalize() if " " not in noun else noun.title()
    return None


def _extract_calendar_payload(text: str) -> Dict[str, Any]:
    """
    Calendar payload: { title, start_iso, start_phrase, duration_minutes? }
    """
    iso, phrase = _parse_datetime(text)

    title = text
    title_low = title.lower()
    for prefix in sorted(_CALENDAR_PREFIXES, key=len, reverse=True):
        if title_low.startswith(prefix):
            title = title[len(prefix):]
            break

    # Drop the date phrase from the title
    title = _strip_date_phrase(title, phrase)

    # Drop calendar housekeeping noise ("on the calendar", "as a meeting", etc.)
    for pat in _CALENDAR_NOISE_PATTERNS:
        title = re.sub(pat, " ", title, flags=re.IGNORECASE)

    # Collapse + clean trailing/leading filler words
    title = re.sub(r"\s+", " ", title).strip(" ,.-")
    title = re.sub(r"\b(at|on|for|to|from|the|a|an|of|with|about)\s*$", "", title, flags=re.IGNORECASE).strip(" ,.-")
    title = re.sub(r"^\s*(hey|yo|btw|fyi|ok so|so|also|wait|please)\s+", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"^(the|a|an|for|to|of|on|at|with|about)\s+", "", title, flags=re.IGNORECASE).strip()

    # If we ended up with junk or nothing, fall back to a known event noun.
    junk_titles = {"the", "a", "an", "for", "to", "of", "on", "at", "with", "about"}
    if not title or len(title) <= 2 or title.lower() in junk_titles:
        title = _fallback_event_noun(text) or title
    # Single-word event nouns get title-cased ("meeting" -> "Meeting")
    if title and " " not in title and title.lower() in {n.lower() for n in _EVENT_NOUNS}:
        title = title.capitalize()

    payload = {"title": title or None}
    if iso:
        payload["start_iso"] = iso
        payload["start_phrase"] = phrase
        # Rough default: 30 min for meetings/calls, 60 min for dinner/lunch/events
        if re.search(r"\b(meeting|call|sync|standup|1:1|interview|review|demo|presentation)\b", text, re.IGNORECASE):
            payload["duration_minutes"] = 30
        else:
            payload["duration_minutes"] = 60

    return payload


# ─── Maps: place ───
_MAPS_PREFIX_VERBS = [
    "meet me at ", "lets meet at ", "let's meet at ", "meet at ", "meet you at ",
    "see u at ", "see you at ", "catch you at ", "meet up at ",
    "come to ", "come meet me at ",
    "i'm at ", "im at ", "i am at ", "currently at ",
    "im parked at ", "waiting at ", "hanging at ", "chilling at ",
    "i'm inside ", "im inside ", "we're at ", "we are at ",
    "heading to ", "on my way to ", "omw to ", "driving to ",
    "pulling up to ", "pulling into ", "headed over to ", "going to ",
    "making my way to ", "en route to ",
    "directions to ", "how do i get to ", "navigate to ", "map me to ",
    "drive to ", "take me to ", "ride to ",
    "open ", "find ", "pull up ",
    "the address is ", "spot is ", "address for ", "venue is ",
    "here's the address ", "event location: ", "event location is ",
]


# Trailing junk that should NEVER be part of a place query.
# These all appear AFTER the actual place in chat: "at 5pm", "for dinner", "tomorrow", etc.
_PLACE_TRAILING_JUNK = [
    # event suffixes — "for the meeting", "for dinner", "for said meeting"
    r"\s+for\s+(?:the\s+|a\s+|an\s+|said\s+|our\s+|this\s+|that\s+)?\w+(?:\s+\w+){0,2}\s*$",
    # day-of-week / temporal anchors at end
    r"\s+(?:tomorrow|today|tonight|tmrw|monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun|next\s+\w+|this\s+\w+)\s*$",
]


def _strip_place_trailers(place: str) -> str:
    """Remove trailing date/event/temporal junk from a candidate place string."""
    out = place
    # NOTE: we deliberately do NOT call _parse_datetime here — it eats real
    # address pieces like "1600" (year 1600) and "5th" (5th of month) that
    # are part of the street, not a date. The trailing-junk regex below
    # already handles concrete day-of-week / "tomorrow" suffixes.
    # Strip event/day suffixes (run twice — sometimes layered: "for dinner tonight")
    for _ in range(2):
        for pat in _PLACE_TRAILING_JUNK:
            out = re.sub(pat, "", out, flags=re.IGNORECASE)
    # Trailing connectors and whitespace cleanup
    out = re.sub(r"\s+", " ", out).strip(" ,.-!?")
    out = re.sub(r"\b(at|on|for|to|from)\s*$", "", out, flags=re.IGNORECASE).strip(" ,.-!?")
    return out


def _extract_place(text: str) -> Optional[str]:
    """
    Strip the maps verb prefix and return whatever remains as the 'place query'.
    We lean on Google Maps' own search — it handles addresses, named places,
    and partial references well. Goal here is just to remove chat noise.
    """
    t = text.strip()
    low = t.lower()
    for verb in sorted(_MAPS_PREFIX_VERBS, key=len, reverse=True):
        if low.startswith(verb):
            tail = t[len(verb):]
            return _strip_place_trailers(tail) or None

    # Didn't match a prefix — try to grab whatever follows "at/to" after a known verb mid-sentence.
    m = re.search(r'\b(?:at|to|from)\s+(.+?)(?:\s+(?:at|tomorrow|today|tonight|tmrw|mon|tue|wed|thu|fri|sat|sun)\b|$)',
                  t, re.IGNORECASE)
    if m:
        return _strip_place_trailers(m.group(1)) or None

    # Last resort: use the full text as the search query (Maps will handle it).
    return _strip_place_trailers(t) or None


# ═════════════════════════════════════════════════════════════════════
#  TARGETING SIGNAL EXTRACTORS
# ═════════════════════════════════════════════════════════════════════

# Small name list — picks up common first names that appear in the training data.
# App side will combine this with actual chat participant list to compute
# the final target user. Model doesn't need to know the chat members.
_COMMON_NAMES_LOWER = {
    "akash", "rohit", "priya", "samyak", "nikhil", "aditi", "aunty", "uncle",
    "mom", "dad", "sarah", "mike", "john", "emma", "alex", "chris",
    "jessica", "brian", "kevin", "amanda", "rachel", "dave", "meera",
    "kiran", "anjali", "sid", "rohan", "neha",
}

_ADDRESSEE_RE = re.compile(
    r'(?:^|\s)(?:hey|yo|ok|@)\s*@?([A-Za-z]{2,})\b',
    re.IGNORECASE,
)
# Word-bounded tokens (no substring bleed into "call", "mall", etc.)
_MUTUAL_WORD_RE = re.compile(
    r"\b(?:everyone|everybody|y'?all|guys|team|squad|group|all\s+of\s+us|"
    r"we|we're|us\s+all|the\s+whole\s+team|entire\s+team|full\s+team)\b",
    re.IGNORECASE,
)
_SELF_PHRASES = [
    "remind me", "ping me", "wake me", "buzz me", "notify me", "alert me",
    "i need ", "i'll ", "ima ", "imma ", "myself", "i have to ", "i gotta ",
    "note to self", "my number", "my cell", "my alarm", "my calendar",
]


def _extract_addressee(text: str) -> Optional[str]:
    """Get the name the message is addressed to, if any. Returns lowercased name."""
    t = text.lower()
    m = _ADDRESSEE_RE.search(" " + t)
    if m:
        cand = m.group(1).lower()
        if cand in _COMMON_NAMES_LOWER:
            return cand
    # Also handle leading "NAME," or "NAME:" patterns
    m2 = re.match(r'^\s*([A-Za-z]{2,})[,:]', text)
    if m2:
        cand = m2.group(1).lower()
        if cand in _COMMON_NAMES_LOWER:
            return cand
    return None


# ─── Contact-specific: pull the name to save the number AS ──────────────
_CONTACT_NAME_PATTERNS = [
    # "under (the) name X"  /  "with (the) name X"  /  "using name X"
    re.compile(r"(?:under|with|using)\s+(?:the\s+)?name\s+(?:of\s+)?(?:'|\")?([A-Za-z][A-Za-z0-9 .'\-]{0,40})", re.IGNORECASE),
    # "named X"
    re.compile(r"\bnamed\s+(?:'|\")?([A-Za-z][A-Za-z0-9 .'\-]{0,40})", re.IGNORECASE),
    # "save X's number" / "save X number" / "save the X's number"  -> X
    re.compile(r"\bsave\s+(?:the\s+)?(?:my\s+)?([A-Za-z][A-Za-z0-9'\-]{1,40}?)(?:'?s)?\s+(?:phone\s+)?(?:number|contact|cell|details|info)\b", re.IGNORECASE),
    # "add X's number" / "add X to (my) contacts"
    re.compile(r"\badd\s+(?:the\s+)?([A-Za-z][A-Za-z0-9'\-]{1,40}?)(?:'?s)?\s+(?:phone\s+|contact\s+)?(?:number|contact|cell|details|info)\b", re.IGNORECASE),
    re.compile(r"\badd\s+([A-Za-z][A-Za-z0-9 .'\-]{1,40}?)\s+to\s+(?:my\s+|the\s+)?contacts?\b", re.IGNORECASE),
    # "save X as Y" / "save NUMBER as Y"
    re.compile(r"\bsave\s+(?:\+?\d[\d\s\-().]{6,}|[A-Za-z][A-Za-z0-9'\-]{0,40})\s+as\s+(?:'|\")?([A-Za-z][A-Za-z0-9 .'\-]{0,40})", re.IGNORECASE),
    # "call them X in (my) contacts"
    re.compile(r"\bcall\s+(?:them\s+|it\s+)?([A-Za-z][A-Za-z0-9 .'\-]{0,40}?)\s+in\s+(?:my\s+|the\s+)?contacts?\b", re.IGNORECASE),
    # "X's number is ..."  (X must be a real word, not a stopword)
    re.compile(r"^\s*([A-Za-z][A-Za-z'\-]{1,40})'?s\s+(?:phone\s+)?(?:number|cell|contact)\s+is\b", re.IGNORECASE),
]

# Words the regexes might catch that aren't really names.
_CONTACT_NAME_STOPWORDS = {
    "the", "this", "that", "these", "those", "my", "his", "her", "our", "their", "your",
    "a", "an", "some", "any", "no", "new", "saved", "phone", "contact",
    "number", "cell", "details", "info", "person", "people",
    # short fragments left over when an optional `'?s` swallows the trailing letter
    "th", "thi", "tha", "thes", "thos",
}


def _extract_contact_name(text: str) -> Optional[str]:
    """
    Best-effort name to save a contact under. Tries explicit "save X's number" /
    "under the name X" patterns first, then falls back to the chat-style
    addressee extractor ("hey NAME save this number").
    """
    for pat in _CONTACT_NAME_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        name = m.group(1).strip(" ,.-'?!\"")
        if not name:
            continue
        # Reject digit-bearing or stopword captures
        if any(c.isdigit() for c in name):
            continue
        if len(name) > 50:
            continue
        low = name.lower()
        # Reject if the captured chunk OR the chunk + 's' is a stopword
        # (the optional `'?s` group in our patterns sometimes eats the trailing
        # 's' of words like "this"/"that", leaving "thi"/"tha")
        if low in _CONTACT_NAME_STOPWORDS or (low + "s") in _CONTACT_NAME_STOPWORDS:
            continue
        # Drop trailing 's if accidentally captured ("akash's" -> "akash")
        name = re.sub(r"['\u2019]s$", "", name).strip()
        if name:
            return name

    # Fall back: catches "hey NAME save this number..."
    return _extract_addressee(text)


def _extract_third_party(text: str, addressee: Optional[str], sender: Optional[str]) -> Optional[str]:
    """Find a name mentioned that's not the addressee and not the sender."""
    t_lower = text.lower()
    sender_lower = (sender or "").lower()
    for name in _COMMON_NAMES_LOWER:
        if name == addressee or name == sender_lower:
            continue
        # word-boundary match
        if re.search(rf'\b{re.escape(name)}\b', t_lower):
            return name
    return None


def _is_self(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _SELF_PHRASES)


def _is_mutual(text: str) -> bool:
    return bool(_MUTUAL_WORD_RE.search(text))


def _build_targeting(text: str, sender: Optional[str]) -> Dict[str, Any]:
    addressee = _extract_addressee(text)
    third_party = _extract_third_party(text, addressee, sender)
    return {
        "addressee":   addressee,
        "third_party": third_party,
        "is_self":     _is_self(text),
        "is_mutual":   _is_mutual(text),
    }


# ═════════════════════════════════════════════════════════════════════
#  MONEY-SPECIFIC CLASSIFIERS (kept from v1 for back-compat)
# ═════════════════════════════════════════════════════════════════════

def _classify_trigger(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["venmo", "cashapp", "cash app", "zelle", "apple pay", "paypal"]):
        return "payment_app"
    if any(w in t for w in ["split", "halves", "half", "divide", "chip in", "go dutch"]):
        return "bill_splitting"
    if any(w in t for w in ["owe", "owed", "pay me back", "pay back", "pay you back",
                            "where's my money", "my money back", "pay me"]):
        return "owing_debt"
    if "$" in t or any(w in t for w in ["dollars", "bucks"]):
        return "direct_amount"
    if any(w in t for w in ["my treat", "on me", "i got you", "i'll cover", "let me cover",
                            "i'll take care", "let me take care", "cover", "spot", "front"]):
        return "general_money"
    return "general_money"


def _classify_direction(text: str) -> str:
    t = text.lower()
    offer_patterns = [
        "i owe", "i'll pay", "i'll send", "let me pay", "let me send",
        "i can pay", "i can send", "i'll venmo", "i'll cashapp", "i'll zelle",
        "shall i send", "should i send", "want me to send", "want me to pay",
        "do i owe", "how much do i owe", "i need to pay", "paying you",
        "send you", "pay you back", "i'll cover", "let me cover",
        "i got you", "my treat", "i'll get this", "on me",
        "sending you", "lemme pay", "lemme send", "ima send", "ima pay",
        "lemme venmo", "lemme cashapp", "lemme zelle",
        "venmo you", "cashapp you", "zelle you",
        "i'll get you", "let me get you", "i'll take care of",
    ]
    for p in offer_patterns:
        if p in t:
            return "offer"
    split_patterns = ["split", "halves", "half", "divide", "each",
                      "chip in", "go dutch", "share the"]
    for p in split_patterns:
        if p in t:
            return "split"
    request_patterns = ["you owe", "owe me", "pay me", "send me", "pay up",
                        "venmo me", "cashapp me", "zelle me", "where's my",
                        "give me", "front me", "spot me", "cover me",
                        "you still owe", "need my money", "hit me with", "throw me"]
    for p in request_patterns:
        if p in t:
            return "request"
    return "request"


# ═════════════════════════════════════════════════════════════════════
#  POPUP COOLDOWN POLICY (per-intent)
# ═════════════════════════════════════════════════════════════════════

def _evict_stale_trackers():
    """Drop tracker entries that have been idle for TRACKER_EVICTION_SECONDS."""
    now = time.time()
    stale = [
        key for key, s in popup_tracker.items()
        if now - s.get("last_event_ts", 0) > TRACKER_EVICTION_SECONDS
    ]
    for key in stale:
        popup_tracker.pop(key, None)


def _payload_changed(old_payload: Optional[dict], new_payload: Optional[dict], intent: str) -> bool:
    """Decide whether the new payload represents a 'new transaction' for this intent.

    Override key per intent:
      money:    amount
      alarm:    time_iso / seconds_from_now
      contact:  phone
      calendar: start_iso
      maps:     place
    """
    if not old_payload or not new_payload:
        return False
    keys = {
        "money":       ["amount"],
        "alarm":       ["time_iso", "seconds_from_now"],
        "contact":     ["phone"],
        "calendar":    ["start_iso"],
        "maps":        ["place"],
        "food_order":  ["item", "cuisine"],
        "ride":        ["dropoff"],
        "travel":      ["destination"],
        "shopping":    ["item"],
        "music":       ["track", "artist", "playlist"],
        "video":       ["title"],
        "tickets":     ["event"],
        "reservation": ["venue"],
        "task":        ["title"],
        "note":        ["content", "url"],
        "bills":       ["kind", "amount"],
        "health":      ["need", "kind"],
        "weather":     ["location"],
    }.get(intent, [])
    for k in keys:
        ov, nv = old_payload.get(k), new_payload.get(k)
        if ov and nv and ov != nv:
            return True
    return False


def _should_show_popup(
    chat_id: Optional[str],
    intent: str,
    payload: Optional[dict],
) -> Tuple[bool, Optional[str], int, str]:
    """
    Per-intent popup decision.

    Returns: (should_popup, suppressed_reason, cooldown_remaining_seconds, chat_state)
    """
    now = time.time()

    if not chat_id:
        return True, None, 0, "untracked"

    _evict_stale_trackers()

    key = (chat_id, intent)
    state = popup_tracker.get(key)

    if state is None:
        return True, None, 0, "idle"

    current = state["state"]

    # Post-payment grace only applies to money
    if current == "post_payment":
        grace_elapsed = now - state["last_event_ts"]
        if grace_elapsed < POST_PAYMENT_GRACE_SECONDS:
            return False, "post_payment_grace", int(POST_PAYMENT_GRACE_SECONDS - grace_elapsed), "post_payment"
        return True, None, 0, "idle"

    cooldown = DISMISSED_COOLDOWN_SECONDS if current == "dismissed" else POPUP_COOLDOWN_SECONDS
    elapsed = now - state["last_popup_ts"]

    if elapsed >= cooldown:
        return True, None, 0, "idle"

    # New distinct payload = new action, pop again
    if _payload_changed(state.get("last_payload"), payload, intent):
        return True, None, 0, current

    remaining = int(cooldown - elapsed)
    reason = "recently_dismissed" if current == "dismissed" else "cooldown_active"
    return False, reason, remaining, current


def _record_popup_fired(chat_id: str, intent: str, payload: Optional[dict]):
    now = time.time()
    key = (chat_id, intent)
    existing = popup_tracker.get(key, {})
    popup_tracker[key] = {
        "state": "cooldown",
        "last_popup_ts": now,
        "last_event_ts": now,
        "last_payload": payload or existing.get("last_payload"),
        "popup_count": existing.get("popup_count", 0) + 1,
        "suppression_count": existing.get("suppression_count", 0),
        "reason_for_current_state": "popup_just_fired",
    }
    stats["popups_fired"] += 1


def _record_popup_suppressed(chat_id: str, intent: str):
    key = (chat_id, intent)
    if key in popup_tracker:
        popup_tracker[key]["suppression_count"] = popup_tracker[key].get("suppression_count", 0) + 1
    stats["popups_suppressed"] += 1


# ═════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═════════════════════════════════════════════════════════════════════

def _sigmoid(x):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-x))


def run_inference(text: str) -> dict:
    """
    Run the model on a single message.

    Returns:
      {
        "intent_probs": { "money": float, "alarm": float, ... },
        "latency_ms":   float,
      }
    For a 2-class legacy model, only "money" will have a real probability;
    the rest default to 0.0 so they never fire.
    """
    import numpy as np
    t0 = time.time()

    tokenizer = model_state["tokenizer"]
    model = model_state["model"]
    num_labels = model_state["num_labels"]
    label_order = model_state["label_order"]

    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0].cpu().numpy()

    intent_probs = {i: 0.0 for i in INTENTS}
    if num_labels == 2:
        # Legacy 2-class softmax — only money
        exp = np.exp(logits - logits.max())
        softmax = exp / exp.sum()
        intent_probs["money"] = float(softmax[1])
    else:
        # Multi-label sigmoid heads. Map each head's output to its named intent.
        probs = _sigmoid(logits)
        for i, intent_name in enumerate(label_order):
            if intent_name in intent_probs:
                intent_probs[intent_name] = float(probs[i])

    latency_ms = (time.time() - t0) * 1000

    stats["requests"] += 1
    stats["_latency_sum"] += latency_ms
    stats["avg_latency_ms"] = stats["_latency_sum"] / stats["requests"]

    return {
        "intent_probs": intent_probs,
        "latency_ms": round(latency_ms, 2),
    }


# ═════════════════════════════════════════════════════════════════════
#  v3 EXTRACTORS — small, regex+heuristic best-effort
# ═════════════════════════════════════════════════════════════════════

# Token banks (stay in sync with training/v3_intents.py — but app.py only
# needs to *recognize* what the model already classifies, so we keep these
# lists tight and lower-cased).
_FOOD_TERMS = {
    "pizza","sushi","burger","burgers","ramen","tacos","burrito","burritos",
    "biryani","pad thai","chinese","wings","salad","sandwich","sandwiches",
    "noodles","fried chicken","chipotle","shawarma","bbq","pho","kebab","kebabs",
    "dim sum","dumplings","sushi roll","fish and chips","smoothie","milkshake",
    "donut","donuts","cookies","ice cream","subway","kfc","mcdonald","pasta",
    "tofu","poke","poke bowl","gyro","falafel","curry","dosa","paneer","thali",
}
_CUISINE_TERMS = {
    "italian","indian","thai","chinese","japanese","mexican","korean",
    "vietnamese","mediterranean","greek","american","french","ethiopian",
    "lebanese","spanish","turkish",
}
_FOOD_PROVIDERS = {"doordash","uber eats","ubereats","grubhub","swiggy","zomato","postmates"}
_RIDE_PROVIDERS = {"uber","lyft","ola","rapido","cab","taxi"}

_AIRPORTS = {"jfk","lax","sfo","sea","blr","bom","del","lhr","cdg","dxb","ord","atl","ewr","lga","yyz","hkg","sin"}

_BILL_TERMS = {
    "rent","electricity","electric","internet","phone","wifi","cable","gas",
    "water","credit card","insurance","gym","netflix","spotify","amex","comcast",
    "apple","amazon prime","subscription","mortgage","emi","loan",
}
_HEALTH_KIND_PHARMACY = {"pharmacy","cvs","walgreens","1mg","apollo","medplus","drugstore","chemist"}
_HEALTH_KIND_DOCTOR   = {"doctor","doc","appointment","dentist","ophthalmologist","physio","telehealth","urgent care","checkup","clinic"}
_HEALTH_KIND_MEDS     = {"meds","medicine","prescription","ibuprofen","tylenol","advil","antibiotic","refill"}

_VIDEO_KIND_MOVIE = {"movie","film","trailer"}
_VIDEO_KIND_SHOW  = {"show","series","episode","season","ep","binge"}

_URL_RE = re.compile(r"https?://\S+")
_QTY_RE = re.compile(r"\b(\d+)\s+(?:people|ppl|guys|of us|persons?)\b", re.IGNORECASE)
_PARTY_SIZE_RE = re.compile(r"\b(?:table\s+for|reservation\s+for|book(?:ing)?\s+for|party\s+of)\s+(\d+|two|three|four|five|six|seven|eight)\b", re.IGNORECASE)
_NUMBER_WORDS = {"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
_MONEY_RE_NEW = re.compile(r"\$\s?\d[\d,]*(?:\.\d{1,2})?|\b\d+\s*(?:dollars?|bucks?|rs|inr|rupees)\b", re.IGNORECASE)


def _first_match(text: str, terms) -> Optional[str]:
    """Return first term from `terms` (iterable of lowercased phrases) found in text."""
    low = text.lower()
    for t in sorted(terms, key=len, reverse=True):
        if re.search(rf"\b{re.escape(t)}\b", low):
            return t
    return None


def _extract_food_order_payload(text: str) -> Dict[str, Any]:
    # Quantity heuristic: "2 pizzas", "a couple of burritos", "three tacos".
    qty = None
    m = re.search(r"\b(\d+)\s+(?=\w+)", text)
    if m:
        qty = int(m.group(1))
    else:
        m = re.search(r"\b(one|two|three|four|five|six|couple|dozen)\s+(?=\w)", text, re.IGNORECASE)
        if m:
            word = m.group(1).lower()
            if word == "one":
                qty = 1
            elif word == "couple":
                qty = 2
            elif word == "dozen":
                qty = 12
            else:
                qty = _NUMBER_WORDS.get(word, None)
    return {
        "item":          _first_match(text, _FOOD_TERMS),
        "cuisine":       _first_match(text, _CUISINE_TERMS),
        "provider_hint": _first_match(text, _FOOD_PROVIDERS),
        "qty":           qty,
    }


def _extract_ride_payload(text: str) -> Dict[str, Any]:
    provider = _first_match(text, _RIDE_PROVIDERS)
    pickup = None
    dropoff = None
    # "from X to Y"
    m = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE)
    if m:
        pickup = m.group(1).strip(" ,.!?")
        dropoff = m.group(2).strip(" ,.!?")
    else:
        # "uber/lyft/ride to X"
        m = re.search(r"\b(?:uber|lyft|ola|rapido|ride|cab|taxi)\s+(?:pool\s+)?to\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE)
        if m:
            dropoff = m.group(1).strip(" ,.!?")
        else:
            # "ubering to X"
            m = re.search(r"\b(?:ubering|lyfting)\s+to\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE)
            if m:
                dropoff = m.group(1).strip(" ,.!?")
    # If dropoff looks like an airport code in caps, normalize
    if dropoff and dropoff.lower().split()[0] in _AIRPORTS:
        dropoff = dropoff.upper().split()[0]
    return {
        "pickup":        pickup,
        "dropoff":       dropoff,
        "provider_hint": provider,
    }


def _extract_travel_payload(text: str) -> Dict[str, Any]:
    iso, phrase = _parse_datetime(text)
    # "trip/flights/flight/vacation to X" or "X trip"
    dest = None
    origin = None
    m = re.search(r"\b(?:trip|flights?|flight|vacation|holiday|tickets?|hotel|airbnb|stay)\s+(?:from\s+(.+?)\s+)?to\s+(.+?)(?:\s+(?:for|on|in|next|this|tomorrow|today)\b|[,.!?]|$)", text, re.IGNORECASE)
    if m:
        origin = (m.group(1) or "").strip(" ,.!?") or None
        dest = m.group(2).strip(" ,.!?") or None
    if not dest:
        m = re.search(r"\b(?:going|head(?:ed|ing)?|fly(?:ing)?)\s+to\s+(.+?)(?:\s+(?:for|on|in|next|this|tomorrow|today)\b|[,.!?]|$)", text, re.IGNORECASE)
        if m:
            dest = m.group(1).strip(" ,.!?") or None
    trip_type = "round_trip" if "round trip" in text.lower() else (
        "one_way" if "one way" in text.lower() else None
    )
    return {
        "destination": dest,
        "origin":      origin,
        "trip_type":   trip_type,
        "when_phrase": phrase,
        "when_iso":    iso,
    }


def _extract_shopping_payload(text: str) -> Dict[str, Any]:
    # "order/buy/get [QTY] [ITEM]"
    item = None
    qty = None
    m = re.search(r"\b(?:order|buy|get|grab|need|pick up)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+|some\s+|more\s+)?(?:(\d+)\s+)?([a-zA-Z][\w \-]{1,40}?)(?:\s+(?:from|on|at|please)\b|[,.!?]|$)", text, re.IGNORECASE)
    if m:
        if m.group(1):
            qty = m.group(1)
        item = m.group(2).strip(" ,.!?")
    # Strip "amazon" if it ended up in the item phrase
    if item:
        item = re.sub(r"\b(?:from|on)?\s*amazon\b", "", item, flags=re.IGNORECASE).strip(" ,.!?")
    return {"item": item or None, "qty": qty}


def _extract_music_payload(text: str) -> Dict[str, Any]:
    # "play X" / "X by Y" / "song X"
    track = None
    artist = None
    m = re.search(r"\b(?:play|queue|add|listen to|stream|put on)\s+(?:the\s+)?(.+?)(?:\s+by\s+(.+?))?(?:[,.!?]|\s+on\s+(?:spotify|apple music)|$)", text, re.IGNORECASE)
    if m:
        track = m.group(1).strip(" ,.!?'\"")
        if m.group(2):
            artist = m.group(2).strip(" ,.!?'\"")
    if not artist:
        m = re.search(r"\b(?:by|from)\s+([A-Z][\w' \-]{1,40})\b", text)
        if m:
            artist = m.group(1).strip(" ,.!?'\"")
    return {"track": track, "artist": artist, "playlist": None}


def _extract_video_payload(text: str) -> Dict[str, Any]:
    title = None
    kind = None
    low = text.lower()
    if any(w in low for w in _VIDEO_KIND_SHOW):
        kind = "show"
    elif any(w in low for w in _VIDEO_KIND_MOVIE):
        kind = "movie"
    m = re.search(r"\b(?:watch|streaming|watching|put on|stream|binge|rewatch(?:ing)?|see)\s+(?:the\s+)?(.+?)(?:[,.!?]|\s+(?:on|tonight|tomorrow|this\s+weekend)\b|$)", text, re.IGNORECASE)
    if m:
        title = m.group(1).strip(" ,.!?'\"")
        title = re.sub(r"^(movie|show|series|film)\s+", "", title, flags=re.IGNORECASE).strip()
    return {"title": title, "kind": kind}


def _extract_tickets_payload(text: str) -> Dict[str, Any]:
    iso, phrase = _parse_datetime(text)
    # "tickets to/for X"  /  "X concert/game/show"
    event = None
    m = re.search(r"\btickets?\s+(?:to|for)\s+(?:the\s+)?(.+?)(?:[,.!?]|\s+(?:on|for|this|next|tomorrow)\b|$)", text, re.IGNORECASE)
    if m:
        event = m.group(1).strip(" ,.!?")
    if not event:
        m = re.search(r"\b(.+?)\s+(?:concert|game|match|show|live)\b", text, re.IGNORECASE)
        if m:
            event = m.group(1).strip(" ,.!?")
    venue = None
    m = re.search(r"\b(?:at|venue\s+is)\s+([A-Z][\w' \-]{2,40})\b", text)
    if m:
        venue = m.group(1).strip(" ,.!?")
    return {"event": event, "venue": venue, "when_phrase": phrase, "when_iso": iso}


def _extract_reservation_payload(text: str) -> Dict[str, Any]:
    iso, phrase = _parse_datetime(text)
    venue = None
    party = None
    # "reservation/table/book at X" / "book X for ..."
    m = re.search(r"\b(?:at|reserve|book(?:ing)?)\s+(?:a\s+)?(?:table\s+at\s+)?([A-Z][\w' \-]{1,50}?)(?:\s+(?:for|tonight|tomorrow|this|next|on|at\b)|[,.!?]|$)", text)
    if m:
        venue = m.group(1).strip(" ,.!?")
    m = _PARTY_SIZE_RE.search(text)
    if m:
        raw = m.group(1).lower()
        party = _NUMBER_WORDS.get(raw, None) or (int(raw) if raw.isdigit() else None)
    if not party:
        m = _QTY_RE.search(text)
        if m:
            party = int(m.group(1))
    return {"venue": venue, "party_size": party, "when_phrase": phrase, "when_iso": iso}


def _extract_task_payload(text: str) -> Dict[str, Any]:
    iso, phrase = _parse_datetime(text)
    title = text
    for prefix in [
        "i need to ", "i have to ", "i gotta ", "i should ", "ima ", "imma ",
        "i'm supposed to ", "supposed to ", "still need to ", "still gotta ",
        "todo: ", "to do: ", "task: ", "remember to ", "make sure to ",
        "add it to the list -- ", "add it to the list - ", "add it to the list — ",
        "follow up on ", "checklist: ", "homework: ",
        "lemme add ", "i still need to ",
        # Reminder phrasings -- super common in chat
        "remind me to ", "please remind me to ", "can you remind me to ",
        "don't forget to ", "dont forget to ", "don't let me forget to ",
        "ping me to ", "nudge me to ",
    ]:
        if title.lower().startswith(prefix):
            title = title[len(prefix):]
            break
    title = _strip_date_phrase(title, phrase)
    title = re.sub(r"\s+", " ", title).strip(" ,.-!?")
    return {"title": title or None, "due_phrase": phrase, "due_iso": iso}


def _extract_note_payload(text: str) -> Dict[str, Any]:
    url = None
    m = _URL_RE.search(text)
    if m:
        url = m.group(0).rstrip(" ,.!?")
    content = text
    for prefix in [
        "save this link: ", "save this: ", "note to self: ", "important: ",
        "remember this — ", "interesting article: ", "for the records: ",
        "snippet to save: ", "good ", "save the ",
    ]:
        if content.lower().startswith(prefix):
            content = content[len(prefix):]
            break
    content = content.strip(" ,.-!?\"'")
    return {"content": content or None, "url": url}


def _extract_bills_payload(text: str) -> Dict[str, Any]:
    # Strip $-amounts before parsing the date — otherwise "$2400" gets read as
    # the year 2400 by dateparser.
    text_for_date = _MONEY_RE_NEW.sub(" ", text)
    iso, phrase = _parse_datetime(text_for_date)
    kind = _first_match(text, _BILL_TERMS)
    amount = None
    m = _MONEY_RE_NEW.search(text)
    if m:
        amount = m.group(0)
    return {"kind": kind, "amount": amount, "due_phrase": phrase, "due_iso": iso}


def _extract_health_payload(text: str) -> Dict[str, Any]:
    low = text.lower()
    kind = None
    if any(w in low for w in _HEALTH_KIND_PHARMACY):
        kind = "pharmacy"
    elif any(w in low for w in _HEALTH_KIND_DOCTOR):
        kind = "doctor"
    elif any(w in low for w in _HEALTH_KIND_MEDS):
        kind = "meds"
    need = None
    m = re.search(r"\b(?:need|refill|order)\s+(?:my\s+|a\s+|some\s+)?(.+?)(?:[,.!?]|\s+(?:from|at|near)\b|$)", text, re.IGNORECASE)
    if m:
        need = m.group(1).strip(" ,.!?")
    return {"kind": kind, "need": need}


_WEATHER_LOC_RE = re.compile(
    r"\b(?:weather|forecast|temperature|temp|raining|rain|sunny|snowing|snow|humid|cold|cloudy|hot|warm|chilly|windy)"
    r"(?:\s+(?:like|going\s+to\s+be|gonna\s+be))?"
    r"\s+(?:in|at|for|over\s+(?:in|at))\s+"
    r"(.+?)(?:[,.!?]|\s+(?:tomorrow|today|tonight|this|next|on|at)\b|$)",
    re.IGNORECASE,
)
# Fallback: "<verb> rain/snow/sun ... in <loc>" -- catches "is it gonna rain in tokyo"
_WEATHER_LOC_RE_ALT = re.compile(
    r"\b(?:rain|snow|sun|hail|thunderstorm|storm)\w*\s+(?:in|at|over)\s+(.+?)"
    r"(?:[,.!?]|\s+(?:tomorrow|today|tonight|this|next|on|at)\b|$)",
    re.IGNORECASE,
)


def _extract_weather_payload(text: str) -> Dict[str, Any]:
    iso, phrase = _parse_datetime(text)
    loc = None
    m = _WEATHER_LOC_RE.search(text) or _WEATHER_LOC_RE_ALT.search(text)
    if m:
        loc = m.group(1).strip(" ,.!?")
    return {"location": loc, "when_phrase": phrase, "when_iso": iso}


# Dispatch table for the v3 extractors.
_V3_EXTRACTORS = {
    "food_order":  _extract_food_order_payload,
    "ride":        _extract_ride_payload,
    "travel":      _extract_travel_payload,
    "shopping":    _extract_shopping_payload,
    "music":       _extract_music_payload,
    "video":       _extract_video_payload,
    "tickets":     _extract_tickets_payload,
    "reservation": _extract_reservation_payload,
    "task":        _extract_task_payload,
    "note":        _extract_note_payload,
    "bills":       _extract_bills_payload,
    "health":      _extract_health_payload,
    "weather":     _extract_weather_payload,
}


def build_intent_payload(intent: str, text: str) -> Dict[str, Any]:
    """Build the action payload for a fired intent. Used by Android, iOS, and web clients."""
    if intent == "money":
        amount = _extract_amount(text)
        return {
            "amount":       amount,
            "trigger_type": _classify_trigger(text),
            "direction":    _classify_direction(text),
        }
    if intent == "alarm":
        return _extract_alarm_payload(text)
    if intent == "contact":
        phone = _extract_phone(text)
        name_hint = _extract_contact_name(text)
        return {
            "phone": phone,
            "name_hint": name_hint,  # best-effort; client picks up from chat context
        }
    if intent == "calendar":
        return _extract_calendar_payload(text)
    if intent == "maps":
        place = _extract_place(text)
        return {"place": place}

    # v3 intents
    extractor = _V3_EXTRACTORS.get(intent)
    if extractor is not None:
        return extractor(text)
    return {}


# ═════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ═════════════════════════════════════════════════════════════════════

class DetectRequest(BaseModel):
    text: str
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    sender: Optional[str] = None


class IntentResult(BaseModel):
    type: str
    confidence: float
    should_popup: bool
    suppressed_reason: Optional[str] = None
    cooldown_remaining_seconds: int = 0
    chat_state: str = "idle"
    payload: Dict[str, Any] = {}
    targeting: Dict[str, Any] = {}


class DetectResponse(BaseModel):
    # Multi-intent array (the new contract)
    intents: List[IntentResult] = []

    # Back-compat flat fields — populated from the money intent if present.
    # Keeps the existing Android build working while teams migrate to `intents`.
    is_money: bool = False
    confidence: float = 0.0
    trigger_type: Optional[str] = None
    direction: Optional[str] = None
    detected_amount: Optional[str] = None
    should_popup: bool = False
    suppressed_reason: Optional[str] = None
    cooldown_remaining_seconds: int = 0
    chat_state: str = "idle"

    latency_ms: float
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    sender: Optional[str] = None


class PaymentCompleteRequest(BaseModel):
    amount: Optional[str] = None
    payer: Optional[str] = None
    payee: Optional[str] = None
    method: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════
#  DETECTION ORCHESTRATOR (shared by HTTP + WS)
# ═════════════════════════════════════════════════════════════════════

def _push_history(chat_id: Optional[str], sender: Optional[str], text: str) -> None:
    """Append a chat turn to the rolling history. No-op when chat_id is missing."""
    if not chat_id:
        return
    buf = chat_history.get(chat_id)
    if buf is None:
        buf = deque(maxlen=CHAT_HISTORY_LEN)
        chat_history[chat_id] = buf
    buf.append((time.time(), sender, text))


def _build_context_text(chat_id: Optional[str], current_text: str) -> str:
    """
    Build the input string we feed to the model. When `USE_CHAT_CONTEXT=1` and
    we have prior turns for this chat, prepend the last few so terse follow-ups
    ("yeah do it", "sure") inherit context. The current message is always the
    last segment. With `USE_CHAT_CONTEXT=0` (default), this is a passthrough.
    """
    if not chat_id or not USE_CHAT_CONTEXT:
        return current_text
    buf = chat_history.get(chat_id)
    if not buf:
        return current_text
    pieces = [t for (_, _, t) in list(buf)[-(CHAT_HISTORY_LEN - 1):]]
    pieces.append(current_text)
    # Use a soft separator so the tokenizer treats them as one stream.
    return " | ".join(pieces).strip()


def _process_message(text: str, chat_id: Optional[str], sender: Optional[str]) -> Dict[str, Any]:
    """
    Run inference, build per-intent payloads + targeting, attach action_urls,
    apply per-intent cooldown policy, and return a dict ready to become a
    DetectResponse.

    If `chat_id` is provided, the previous N messages (CHAT_HISTORY_LEN) for
    that chat are prepended to the model input as context, and the new message
    is appended to the history afterwards.
    """
    context_text = _build_context_text(chat_id, text)
    infer = run_inference(context_text)
    probs = infer["intent_probs"]

    _evict_stale_trackers()

    targeting_shared = _build_targeting(text, sender)

    intents_list: List[Dict[str, Any]] = []

    for intent in INTENTS:
        conf = probs.get(intent, 0.0)
        if conf < _threshold_for(intent):
            continue

        # Extractors run on the *current* message only — prior turns are for
        # classification context, not field extraction.
        payload = build_intent_payload(intent, text)

        # Tack on a one-tap deep link the client can fire directly.
        try:
            action_url = build_action_url(intent, payload, targeting_shared)
        except Exception as e:  # never let URL-builder bugs break detection
            logger.debug(f"build_action_url failed for {intent}: {e}")
            action_url = None
        if action_url:
            payload["action_url"] = action_url

        # Per-intent cooldown decision
        should_popup, reason, cooldown_rem, chat_state = _should_show_popup(chat_id, intent, payload)

        if chat_id:
            if should_popup:
                _record_popup_fired(chat_id, intent, payload)
            else:
                _record_popup_suppressed(chat_id, intent)
            # Refresh chat_state to post-call truth
            chat_state = popup_tracker.get((chat_id, intent), {}).get("state", chat_state)

        stats["intents_detected"][intent] = stats["intents_detected"].get(intent, 0) + 1
        if intent == "money":
            stats["money_detected"] += 1

        intents_list.append({
            "type":                       intent,
            "confidence":                 round(conf, 4),
            "should_popup":               should_popup,
            "suppressed_reason":          reason,
            "cooldown_remaining_seconds": cooldown_rem,
            "chat_state":                 chat_state,
            "payload":                    payload,
            "targeting":                  targeting_shared,
        })

    # Update rolling history *after* inference (so the current turn is
    # available to disambiguate the *next* turn).
    _push_history(chat_id, sender, text)

    # ── Back-compat flat fields from the money intent ──
    money_intent = next((i for i in intents_list if i["type"] == "money"), None)
    if money_intent:
        flat = {
            "is_money":                    True,
            "confidence":                  money_intent["confidence"],
            "trigger_type":                money_intent["payload"].get("trigger_type"),
            "direction":                   money_intent["payload"].get("direction"),
            "detected_amount":             money_intent["payload"].get("amount"),
            "should_popup":                money_intent["should_popup"],
            "suppressed_reason":           money_intent["suppressed_reason"],
            "cooldown_remaining_seconds":  money_intent["cooldown_remaining_seconds"],
            "chat_state":                  money_intent["chat_state"],
        }
    else:
        # No money fired — still include raw money confidence for callers that care
        flat = {
            "is_money": False,
            "confidence": round(probs.get("money", 0.0), 4),
            "trigger_type": None,
            "direction": None,
            "detected_amount": None,
            "should_popup": False,
            "suppressed_reason": "not_money",
            "cooldown_remaining_seconds": 0,
            "chat_state": (
                popup_tracker.get((chat_id, "money"), {}).get("state", "idle")
                if chat_id else "untracked"
            ),
        }

    return {
        "intents":    intents_list,
        "latency_ms": infer["latency_ms"],
        **flat,
    }


# ═════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════

@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server is still starting.")

    result = _process_message(req.text, req.chat_id, req.sender)
    return DetectResponse(
        **result,
        chat_id=req.chat_id,
        message_id=req.message_id,
        sender=req.sender,
    )


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """Real-time stream — identical logic to POST /detect."""
    await websocket.accept()
    logger.info(f"WS client connected: {websocket.client}")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid JSON"}))
                continue

            text = msg.get("text", "")
            if not text.strip():
                await websocket.send_text(json.dumps({"error": "text required"}))
                continue

            chat_id = msg.get("chat_id")
            sender = msg.get("sender")
            result = _process_message(text, chat_id, sender)

            response = {
                **msg,
                "venmo_detection": {
                    # Back-compat money fields — keep key name for existing client
                    "is_money": result["is_money"],
                    "confidence": result["confidence"],
                    "trigger_type": result["trigger_type"],
                    "direction": result["direction"],
                    "detected_amount": result["detected_amount"],
                    "should_popup": result["should_popup"],
                    "suppressed_reason": result["suppressed_reason"],
                    "cooldown_remaining_seconds": result["cooldown_remaining_seconds"],
                    "chat_state": result["chat_state"],
                    "latency_ms": result["latency_ms"],
                },
                "intents": result["intents"],
            }
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        logger.info(f"WS client disconnected: {websocket.client}")


@app.get("/health")
async def health():
    return {
        "status":              "healthy" if model_state["model"] is not None else "loading",
        "device":              str(DEVICE),
        "model_dir":           str(MODEL_DIR),
        "num_labels":          model_state["num_labels"],
        "intents":             model_state["label_order"],
        "all_intents":         INTENTS,
        "loaded_at":           model_state["loaded_at"],
        "version":             model_state["version"],
        "threshold":           CONFIDENCE_THRESHOLD,
        "per_intent_thresholds": PER_INTENT_THRESHOLDS,
        "use_chat_context":    USE_CHAT_CONTEXT,
        "chat_history_len":    CHAT_HISTORY_LEN,
        "total_requests":      stats["requests"],
        "dateparser":          _HAS_DATEPARSER,
    }


@app.get("/metrics")
async def metrics():
    fired = stats["popups_fired"]
    suppressed = stats["popups_suppressed"]
    return {
        "requests":              stats["requests"],
        "money_detected":        stats["money_detected"],
        "intents_detected":      stats["intents_detected"],
        "detection_rate":        round(stats["money_detected"] / max(stats["requests"], 1), 4),
        "popups_fired":          fired,
        "popups_suppressed":     suppressed,
        "suppression_rate":      round(suppressed / max(fired + suppressed, 1), 4),
        "active_chat_trackers":  len(popup_tracker),
        "avg_latency_ms":        round(stats["avg_latency_ms"], 2),
        "started_at":            stats["started_at"],
    }


@app.post("/payment-complete/{chat_id}")
async def payment_complete(chat_id: str, body: Optional[PaymentCompleteRequest] = None):
    """Only affects the money intent. Clears money cooldown + 60s grace."""
    now = time.time()
    key = (chat_id, "money")
    existing = popup_tracker.get(key, {})
    previous_state = existing.get("state", "idle")

    popup_tracker[key] = {
        "state": "post_payment",
        "last_popup_ts": existing.get("last_popup_ts", now),
        "last_event_ts": now,
        "last_payload": {"amount": (body.amount if body else None)} if body and body.amount
                        else existing.get("last_payload"),
        "popup_count": existing.get("popup_count", 0),
        "suppression_count": existing.get("suppression_count", 0),
        "reason_for_current_state": "payment_completed",
    }

    logger.info(
        f"payment_complete chat={chat_id} prev={previous_state} "
        f"amount={body.amount if body else None} method={body.method if body else None}"
    )
    return {
        "status": "ok",
        "chat_id": chat_id,
        "intent": "money",
        "chat_state": "post_payment",
        "previous_state": previous_state,
        "grace_window_seconds": POST_PAYMENT_GRACE_SECONDS,
        "popup_count_total": popup_tracker[key]["popup_count"],
    }


@app.post("/popup-dismissed/{chat_id}")
async def popup_dismissed(
    chat_id: str,
    intent: str = Query("money", description="Which intent was dismissed. Defaults to money for back-compat."),
):
    """Extend cooldown for a specific intent in this chat."""
    if intent not in INTENTS:
        raise HTTPException(status_code=400, detail=f"unknown intent '{intent}'. Valid: {INTENTS}")

    now = time.time()
    key = (chat_id, intent)
    existing = popup_tracker.get(key, {})
    previous_state = existing.get("state", "idle")

    popup_tracker[key] = {
        "state": "dismissed",
        "last_popup_ts": now,
        "last_event_ts": now,
        "last_payload": existing.get("last_payload"),
        "popup_count": existing.get("popup_count", 0),
        "suppression_count": existing.get("suppression_count", 0),
        "reason_for_current_state": "user_dismissed_popup",
    }
    return {
        "status": "ok",
        "chat_id": chat_id,
        "intent": intent,
        "chat_state": "dismissed",
        "previous_state": previous_state,
        "cooldown_seconds": DISMISSED_COOLDOWN_SECONDS,
    }


@app.post("/reset-cooldown/{chat_id}")
async def reset_cooldown(
    chat_id: str,
    intent: Optional[str] = Query(None, description="Specific intent to clear. Omit to clear all intents for this chat."),
):
    """Force-clear cooldown for one or all intents in a chat."""
    if intent is not None and intent not in INTENTS:
        raise HTTPException(status_code=400, detail=f"unknown intent '{intent}'. Valid: {INTENTS}")

    if intent is None:
        # Clear all intents for this chat
        removed = [k for k in list(popup_tracker.keys()) if k[0] == chat_id]
        for k in removed:
            popup_tracker.pop(k, None)
        return {
            "status": "ok",
            "chat_id": chat_id,
            "cleared_intents": [k[1] for k in removed],
            "chat_state": "idle",
        }
    else:
        key = (chat_id, intent)
        existed = key in popup_tracker
        popup_tracker.pop(key, None)
        return {
            "status": "ok",
            "chat_id": chat_id,
            "intent": intent,
            "existed": existed,
            "chat_state": "idle",
        }


@app.get("/chat-state/{chat_id}")
async def chat_state(chat_id: str):
    """Return per-intent tracker state for a chat."""
    now = time.time()
    entries = {}
    for (cid, intent), state in popup_tracker.items():
        if cid != chat_id:
            continue
        current = state["state"]
        if current == "post_payment":
            remaining = max(0, POST_PAYMENT_GRACE_SECONDS - (now - state["last_event_ts"]))
        else:
            cooldown = DISMISSED_COOLDOWN_SECONDS if current == "dismissed" else POPUP_COOLDOWN_SECONDS
            remaining = max(0, cooldown - (now - state["last_popup_ts"]))
        entries[intent] = {
            "state": current,
            "last_popup_at": datetime.utcfromtimestamp(state["last_popup_ts"]).isoformat() if state.get("last_popup_ts") else None,
            "last_event_at": datetime.utcfromtimestamp(state["last_event_ts"]).isoformat() if state.get("last_event_ts") else None,
            "last_payload": state.get("last_payload"),
            "popup_count": state.get("popup_count", 0),
            "suppression_count": state.get("suppression_count", 0),
            "cooldown_remaining_seconds": int(remaining),
            "reason_for_current_state": state.get("reason_for_current_state"),
        }

    if not entries:
        return {
            "chat_id": chat_id,
            "state": "idle",
            "intents": {},
            "message": "no tracker entries — next message will popup for any detected intent",
        }
    return {"chat_id": chat_id, "intents": entries}


@app.post("/reload")
async def reload_model_endpoint():
    try:
        load_model()
        return {
            "status": "ok",
            "loaded_at": model_state["loaded_at"],
            "num_labels": model_state["num_labels"],
            "intents": model_state["label_order"],
            "version": model_state["version"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Voice-driven demo UI — open in Chrome on desktop or mobile."""
    demo_dir = os.environ.get("DEMO_DIR", os.path.join(os.path.dirname(__file__), "demo"))
    path = os.path.join(demo_dir, "voice_demo.html")
    if not os.path.exists(path):
        return HTMLResponse(
            "<h1>Demo not packaged with this build</h1>"
            f"<p>Expected file at <code>{path}</code></p>",
            status_code=404,
        )
    return FileResponse(path, media_type="text/html")


@app.get("/")
async def root():
    return {
        "service": "FYOE Multi-Intent Detection API",
        "version": "3.0.0",
        "intents": INTENTS,
        "model_intents": model_state["label_order"],
        "chat_history_len": CHAT_HISTORY_LEN,
        "docs": "/docs",
        "health": "/health",
        "demo": "/demo",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
