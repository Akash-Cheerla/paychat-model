"""
PayChat Multi-Intent Detection API

Detects one or more actionable intents in a chat message and returns
structured payloads for each:
  - money    — payment/debt/split   -> Venmo / CashApp popup
  - alarm    — reminder / wake-up   -> Android AlarmClock intent
  - contact  — phone number share   -> ContactsContract.Insert intent
  - calendar — meeting / event      -> CalendarContract.Events insert
  - maps     — place / address      -> geo: intent (Google Maps)

Back-compat: flat is_money / should_popup / trigger_type / etc. are populated
from the money intent (if any) so the existing Android build keeps working.

Endpoints:
  POST /detect                   single message inference
  WS   /ws/detect                real-time stream
  GET  /health                   health + model version
  GET  /metrics                  live stats
  POST /reload                   hot-reload model from disk
  POST /payment-complete/{chat_id}    signal a payment landed (money only)
  POST /popup-dismissed/{chat_id}     signal user dismissed the popup
  POST /reset-cooldown/{chat_id}      force-clear all intent cooldowns
  GET  /chat-state/{chat_id}          inspect tracker (per-intent)
"""

import json
import os
import re
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

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

# Popup anti-spam windows. Stateless model stays dumb; the API holds UX policy.
POPUP_COOLDOWN_SECONDS     = int(os.getenv("POPUP_COOLDOWN_SECONDS",     "300"))
DISMISSED_COOLDOWN_SECONDS = int(os.getenv("DISMISSED_COOLDOWN_SECONDS", "900"))
POST_PAYMENT_GRACE_SECONDS = int(os.getenv("POST_PAYMENT_GRACE_SECONDS", "60"))
TRACKER_EVICTION_SECONDS   = int(os.getenv("TRACKER_EVICTION_SECONDS",   "1800"))

MAX_LEN = 128
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTENTS = ["money", "alarm", "contact", "calendar", "maps"]


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


# ─────────────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────────────
def load_model(model_dir: Path = MODEL_DIR):
    """Load or hot-swap the model from disk.

    Supports both:
      - 2-label legacy models (money-only). Only the money intent fires.
      - 5-label multi-intent models. All 5 intents are active.
    """
    logger.info(f"Loading model from {model_dir}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    model = model.to(DEVICE)
    model.eval()

    num_labels = model.config.num_labels

    # Derive intent order: trust id2label if present, else fall back to defaults.
    id2label = getattr(model.config, "id2label", None) or {}
    # id2label may come back as str keys (HF serialization quirk), normalize.
    try:
        id2label_norm = {int(k): v for k, v in id2label.items()}
    except (TypeError, ValueError):
        id2label_norm = id2label

    if num_labels == 5:
        # Try to use saved label order; fall back to canonical INTENTS order
        label_order = [id2label_norm.get(i, INTENTS[i]) for i in range(5)]
        # Sanity: if labels are just "LABEL_0"..."LABEL_4", override
        if any(lbl.startswith("LABEL_") for lbl in label_order):
            label_order = list(INTENTS)
    elif num_labels == 2:
        # Legacy money classifier — index 1 = money positive
        label_order = ["not_money", "money"]
    else:
        raise RuntimeError(f"Unsupported num_labels={num_labels}. Expected 2 or 5.")

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
    title="PayChat Multi-Intent Detection API",
    description="Detects money / alarm / contact / calendar / maps intents in chat. "
                "Returns per-intent payloads and targeting signals for Android intent wiring.",
    version="2.0.0",
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
    """Reject dateparser garbage like 'me', 'set', 'on'."""
    if not phrase or len(phrase) < 3:
        return False
    low = phrase.lower()
    # Must have a digit OR a known time word
    if any(c.isdigit() for c in low):
        return True
    for w in _TIME_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", low):
            return True
    return False


def _parse_datetime(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Pull the first valid date/time phrase out of text.

    Returns: (iso_string, raw_phrase). Both None if nothing parseable found.
    """
    if not _HAS_DATEPARSER:
        return None, None
    try:
        results = _dp_search_dates(
            text,
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
                return dt.isoformat(timespec="minutes"), raw_phrase
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
    # Replace with single space so mid-string removals don't leave "foo  bar"
    out = text.replace(phrase, " ")
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


def _extract_calendar_payload(text: str) -> Dict[str, Any]:
    """
    Calendar payload: { title, start_iso, start_phrase, duration_minutes? }
    """
    iso, phrase = _parse_datetime(text)

    title = text
    for prefix in [
        "schedule a meeting ", "schedule ", "let's schedule ",
        "put ", "add ", "book ", "block my calendar ", "block off ",
        "pencil me in ", "save the date ", "save the date for ",
    ]:
        if title.lower().startswith(prefix):
            title = title[len(prefix):]
            break
    title = _strip_date_phrase(title, phrase)
    # Clean up common filler
    title = re.sub(r'\b(at|on|for|to|from)\s*$', '', title).strip(" ,.-")
    title = re.sub(r'^\s*(hey|yo|btw|fyi|ok so|so|also|wait)\s+', '', title, flags=re.IGNORECASE).strip()

    payload = {"title": title or None}
    if iso:
        payload["start_iso"] = iso
        payload["start_phrase"] = phrase
        # Rough default: 30 min for meetings/calls, 60 min for dinner/lunch/events
        if re.search(r'\b(meeting|call|sync|standup|1:1|interview)\b', text, re.IGNORECASE):
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
    "open ", "find ", "pull up ",
    "the address is ", "spot is ", "address for ", "venue is ",
    "here's the address ", "event location: ", "event location is ",
]


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
            return t[len(verb):].strip(" ,.-!?")

    # Didn't match a prefix — try to grab whatever follows "at/to" after a known verb mid-sentence.
    m = re.search(r'\b(?:at|to|from)\s+(.+?)(?:\s+(?:at|tomorrow|today|tonight|tmrw|mon|tue|wed|thu|fri|sat|sun)\b|$)',
                  t, re.IGNORECASE)
    if m:
        return m.group(1).strip(" ,.-!?")

    # Last resort: use the full text as the search query (Maps will handle it).
    return t.strip(" ,.-!?") or None


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
        "money":    ["amount"],
        "alarm":    ["time_iso", "seconds_from_now"],
        "contact":  ["phone"],
        "calendar": ["start_iso"],
        "maps":     ["place"],
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
    if num_labels == 5:
        probs = _sigmoid(logits)  # independent sigmoids
        for i, intent_name in enumerate(label_order):
            if intent_name in intent_probs:
                intent_probs[intent_name] = float(probs[i])
    else:
        # Legacy 2-class softmax — only money
        exp = np.exp(logits - logits.max())
        softmax = exp / exp.sum()
        intent_probs["money"] = float(softmax[1])

    latency_ms = (time.time() - t0) * 1000

    stats["requests"] += 1
    stats["_latency_sum"] += latency_ms
    stats["avg_latency_ms"] = stats["_latency_sum"] / stats["requests"]

    return {
        "intent_probs": intent_probs,
        "latency_ms": round(latency_ms, 2),
    }


def build_intent_payload(intent: str, text: str) -> Dict[str, Any]:
    """Build the Android-consumable payload for a fired intent."""
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
        addressee = _extract_addressee(text)
        return {
            "phone": phone,
            "name_hint": addressee,  # best-effort; android picks up from chat context
        }
    if intent == "calendar":
        return _extract_calendar_payload(text)
    if intent == "maps":
        place = _extract_place(text)
        return {"place": place}
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

def _process_message(text: str, chat_id: Optional[str], sender: Optional[str]) -> Dict[str, Any]:
    """
    Run inference, build per-intent payloads + targeting, apply per-intent
    cooldown policy, and return a dict ready to become a DetectResponse.
    """
    infer = run_inference(text)
    probs = infer["intent_probs"]

    _evict_stale_trackers()

    targeting_shared = _build_targeting(text, sender)

    intents_list: List[Dict[str, Any]] = []

    for intent in INTENTS:
        conf = probs[intent]
        if conf < CONFIDENCE_THRESHOLD:
            continue

        payload = build_intent_payload(intent, text)

        # Per-intent cooldown decision
        should_popup, reason, cooldown_rem, chat_state = _should_show_popup(chat_id, intent, payload)

        if chat_id:
            if should_popup:
                _record_popup_fired(chat_id, intent, payload)
            else:
                _record_popup_suppressed(chat_id, intent)
            # Refresh chat_state to post-call truth
            chat_state = popup_tracker.get((chat_id, intent), {}).get("state", chat_state)

        stats["intents_detected"][intent] += 1
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
        "status":          "healthy" if model_state["model"] is not None else "loading",
        "device":          str(DEVICE),
        "model_dir":       str(MODEL_DIR),
        "num_labels":      model_state["num_labels"],
        "intents":         model_state["label_order"],
        "loaded_at":       model_state["loaded_at"],
        "version":         model_state["version"],
        "threshold":       CONFIDENCE_THRESHOLD,
        "total_requests":  stats["requests"],
        "dateparser":      _HAS_DATEPARSER,
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
        "service": "PayChat Multi-Intent Detection API",
        "version": "2.0.0",
        "intents": INTENTS,
        "docs": "/docs",
        "health": "/health",
        "demo": "/demo",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
