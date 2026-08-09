"""
PayChat Intent Detection API — v11 (9-intent) with full pipeline
FastAPI server with:
  - POST /detect     → single message detection (all 9 intents)
  - WS   /ws/detect  → real-time WebSocket detection for chat apps
  - GET  /health     → health check (model version, accuracy, uptime)
  - GET  /metrics    → inference stats

Pipeline phases:
  Phase 1: 9-intent RoBERTa model + fast keyword detection
  Phase 2: Conversation context — track recent messages per room, boost ambiguous intents
  Phase 3: Slot extraction — recipient, amount, time, destination from messages
  Phase 4: Cancel/defer/re-trigger — track active intents, detect lifecycle phrases

Intents: money, ride, food_order, contact, alarm, reminder, calendar, bills, travel
"""

import asyncio
import json
import os
import re
import time
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
try:
    from api.conversation import conversation_sm, Action, MANAGED_INTENTS
except ModuleNotFoundError:
    from conversation import conversation_sm, Action, MANAGED_INTENTS
import numpy as np
import spacy
try:
    import onnxruntime as ort
except ImportError:
    ort = None
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, RobertaModel, RobertaConfig
from safetensors.torch import load_file
try:
    from api.guardrails import run_guardrails
except ModuleNotFoundError:
    from guardrails import run_guardrails

_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──
MODEL_DIR      = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent.parent / "model" / "saved_model")))
MAX_LEN        = 256
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTEXT_TIME_CAP = 300  # 5 minutes — ignore context messages older than this
# The conversation classifier needs its own, much longer cap. CONTEXT_TIME_CAP exists
# for the intent model's context prefix, where five minutes is right. For the classifier
# the window IS the memory: if the request has aged out, the model judges "yeah one sec"
# with nothing in front of it and scores 0.03 instead of 0.997.
#
# Real conversations do not run at test speed. A reply seven minutes after the request
# is ordinary, and at 300s it silently produced no window at all — every eval measured
# messages replayed milliseconds apart and so never saw this.
#
# Matches PENDING_TTL in conversation.py (4h), which was raised from 5 minutes for the
# same reason on the rules path.
CONV_CONTEXT_TIME_CAP = int(os.environ.get("PAYCHAT_CONV_CONTEXT_TTL", 4 * 3600))

INTENTS = ["money", "ride", "food_order", "contact", "alarm", "reminder", "calendar", "bills", "travel"]

# Which of the nine the server actually SURFACES. The model still scores all nine and
# the scores are still logged; this only controls what lands in `intents`.
#
# Money and ride are the only two trained and evaluated to shipping standard. The other
# seven were split out of the training data months ago and never retrained, so they
# misfire on ordinary chat — "lunch at 1?" scores calendar, "please?" scores food_order.
# Product decision 2026-08-05: ship money and ride only until each of the rest gets its
# own training round.
#
# Set PAYCHAT_ACTIVE_INTENTS="all" to surface all nine, or a comma list to widen it
# selectively once an intent has been retrained.
_active_env = os.environ.get("PAYCHAT_ACTIVE_INTENTS", "money,ride").strip()
if _active_env.lower() in ("all", "*"):
    ACTIVE_INTENTS = None          # no filtering
else:
    ACTIVE_INTENTS = {i.strip() for i in _active_env.split(",") if i.strip()}
    unknown = ACTIVE_INTENTS - set(INTENTS)
    if unknown:
        raise ValueError(f"PAYCHAT_ACTIVE_INTENTS has unknown intents: {sorted(unknown)}")


# ── DualHeadRoberta (v20 architecture) ──
RESPONSE_CLASSES = ['ack', 'reject', 'future_promise', 'question', 'already_done', 'neutral']

class DualHeadRoberta(nn.Module):
    def __init__(self, model_name_or_config, num_intents=9, proj_dim=128, dropout=0.1):
        super().__init__()
        if isinstance(model_name_or_config, str):
            self.roberta = RobertaModel.from_pretrained(model_name_or_config)
        else:
            self.roberta = RobertaModel(model_name_or_config)
        hidden = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.topic_head = nn.Linear(hidden, num_intents)
        self.action_head = nn.Linear(hidden, 1)
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )
        self.response_head = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(RESPONSE_CLASSES)),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        topic_logits = self.topic_head(cls)
        action_logits = self.action_head(cls).squeeze(-1)
        projections = self.projection(cls)
        return topic_logits, action_logits, projections, cls

    def classify_response(self, pending_cls: torch.Tensor, current_cls: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([pending_cls, current_cls], dim=-1)
        return self.response_head(combined)


# ── Global state ──
model_state = {
    "model":      None,
    "tokenizer":  None,
    "thresholds": {},
    "version":    None,
    "loaded_at":  None,
}

stats = {
    "requests": 0,
    "detections": 0,
    "started_at": datetime.utcnow().isoformat(),
    "avg_latency_ms": 0,
    "_latency_sum": 0,
}


# ── Model loading ──
def load_model(model_dir: Path = MODEL_DIR):
    """Load DualHeadRoberta v20 model with dual-head gating."""
    logger.info(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    # Load model info for architecture params
    info_path = model_dir / "model_info.json"
    num_intents = len(INTENTS)
    proj_dim = 128
    dropout = 0.1
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        num_intents = info.get("num_intents", num_intents)
        proj_dim = info.get("projection_dim", proj_dim)
        dropout = info.get("dropout", dropout)

    # Build model from config + load safetensors weights
    config = RobertaConfig.from_pretrained(str(model_dir))
    model = DualHeadRoberta(config, num_intents=num_intents, proj_dim=proj_dim, dropout=dropout)
    weights = load_file(str(model_dir / "model.safetensors"))
    weights = {k.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias"): v for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    has_response_head = any(k.startswith("response_head") for k in weights)
    model = model.to(DEVICE)
    model.eval()

    thresholds = {i: 0.5 for i in INTENTS}
    thresh_path = model_dir / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresholds.update(json.load(f))

    version = None
    report_path = model_dir / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        version = {
            "model":            report.get("model", "v20"),
            "architecture":     report.get("architecture", "DualHeadRoberta"),
            "trained_at":       report.get("trained_at"),
            "test_exact_match": report.get("test_exact_match"),
            "test_hamming":     report.get("test_hamming"),
            "intents":          report.get("intents", INTENTS),
        }

    model_state["model"]      = model
    model_state["tokenizer"]  = tokenizer
    model_state["has_response_head"] = has_response_head
    model_state["thresholds"] = thresholds
    model_state["version"]    = version
    model_state["loaded_at"]  = datetime.utcnow().isoformat()

    # Load ONNX session if available (faster CPU inference) — prefer INT8 quantized
    onnx_int8 = model_dir / "model_int8.onnx"
    onnx_fp32 = model_dir / "model.onnx"
    onnx_path = onnx_int8 if onnx_int8.exists() else onnx_fp32
    if ort and onnx_path.exists():
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        model_state["onnx_session"] = ort.InferenceSession(
            str(onnx_path), so, providers=["CPUExecutionProvider"]
        )
        variant = "INT8" if onnx_path == onnx_int8 else "FP32"
        logger.info(f"ONNX {variant} session loaded ({onnx_path.stat().st_size / 1024 / 1024:.0f} MB)")
    else:
        model_state["onnx_session"] = None

    acc = version.get("test_exact_match") if version else None
    logger.info(f"Model loaded (DualHeadRoberta). {len(INTENTS)} intents. Exact-match: {acc:.2%}" if acc else "Model loaded.")
    logger.info(f"Thresholds: {thresholds}")
    logger.info(f"Action threshold: {thresholds.get('_action', 0.5)}")
    logger.info(f"Inference backend: {'ONNX' if model_state.get('onnx_session') else 'PyTorch'}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ── App ──
app = FastAPI(
    title="PayChat Intent Detection",
    description="Real-time 9-intent detection for chat apps: money, ride, food_order, contact, alarm, reminder, calendar, bills, travel.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Inference ──

def run_inference(text: str, prev_messages: list = None) -> dict:
    """Run dual-head model on current message ONLY (no context prefix).
    Context is handled by the conversation state machine, not the model.
    Intent fires only when BOTH topic head and action head agree."""
    t0 = time.time()

    tokenizer = model_state["tokenizer"]
    model = model_state["model"]
    thresholds = model_state["thresholds"]
    action_thresh = thresholds.get("_action", 0.5)

    # Model runs on current message standalone — context contamination fix.
    # Previous: prepended prev1 </s> prev2 </s> current, which dropped scores from 0.94 to 0.03.
    # Context-aware decisions are now handled by ConversationStateMachine in api/conversation.py.
    model_input = text

    onnx_session = model_state.get("onnx_session")

    if onnx_session:
        enc = tokenizer(model_input, max_length=MAX_LEN, truncation=True,
                        padding=True, return_tensors="np")
        onnx_out = onnx_session.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
        topic_probs = (1 / (1 + np.exp(-onnx_out[0][0]))).tolist()  # sigmoid
        action_prob = float(1 / (1 + np.exp(-onnx_out[1][0])))
        cls_embedding = torch.from_numpy(onnx_out[3][0])
    else:
        enc = tokenizer(model_input, max_length=MAX_LEN, padding="max_length",
                        truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            topic_logits, action_logits, _, cls_emb = model(input_ids=input_ids, attention_mask=attention_mask)
        topic_probs = torch.sigmoid(topic_logits[0]).cpu().tolist()
        cls_embedding = cls_emb[0].detach().cpu()
        action_prob = torch.sigmoid(action_logits).cpu().item()

    scores = {INTENTS[j]: round(topic_probs[j], 4) for j in range(len(INTENTS))}

    # Dual-head gating: intent fires only if topic >= threshold AND action >= action_thresh
    fired = []
    for i in INTENTS:
        if scores[i] >= thresholds.get(i, 0.5) and action_prob >= action_thresh:
            fired.append(i)

    latency_ms = (time.time() - t0) * 1000
    stats["requests"] += 1
    stats["_latency_sum"] += latency_ms
    stats["avg_latency_ms"] = stats["_latency_sum"] / stats["requests"]
    if fired:
        stats["detections"] += 1

    money_info = None
    if "money" in fired:
        money_info = _enrich_money(text)

    return {
        "intents":      fired,
        "scores":       scores,
        "action_score": round(action_prob, 4),
        "money":        money_info,
        "latency_ms":   round(latency_ms, 2),
        "_cls_embedding": cls_embedding,
    }


def fast_keyword_detect(text: str) -> dict:
    """
    Ultra-fast keyword-based detection for all 9 intents (<1ms).
    Used for instant chat responses before model inference completes.
    """
    t0 = time.time()
    t = text.lower()
    fired = []
    scores = {}

    # ── Money ──
    money_words = [
        'venmo', 'cashapp', 'cash app', 'zelle', 'apple pay',
        'pay me', 'owe me', 'owe you', 'i owe', 'you owe',
        'pay back', 'pay you back', 'pay me back',
        'split', 'halves', 'chip in', 'go dutch',
        'send me', 'send you', 'front me', 'spot me', 'cover me',
        "i'll cover", 'let me cover', 'my treat', 'on me',
        "let me pay", "let me send", "i'll pay", "i'll send",
        'ima send', 'ima pay', 'lemme pay', 'pay up',
        'i got you', "i'll get this", 'need my money',
    ]
    has_money_kw = any(w in t for w in money_words)
    has_amount = bool(re.search(r'[\$₹][\d,.]+|\d+\s*[\$₹]|\d+\s*(?:dollars?|bucks?|rupees?|rs\.?|inr|ringgit|rm|myr)\b', t))
    # Suppress false positives
    money_suppress = ['pay attention', 'pay respect', 'pay the price', 'i owe my success',
                      'owe it to', "don't owe", 'doesnt owe', "doesn't owe"]
    is_suppressed = any(s in t for s in money_suppress)
    # Bare amounts ($85, 200 bucks) without a money keyword are just price mentions — not requests.
    # Require a money keyword for the guardrail to fire; amounts alone only boost model scores.
    if has_money_kw and not is_suppressed:
        fired.append("money")
        scores["money"] = 0.90 if has_amount else 0.80

    # ── Ride ──
    ride_words = [
        'uber', 'lyft', 'cab', 'taxi', 'ola',
        'book a ride', 'book me a ride', 'get me a ride',
        'need a ride', 'give me a ride', 'pick me up',
        'drop me', 'get me home', 'get me there',
        'arrange transport', 'arrange pickup', 'call a cab',
        "i'm stranded", 'im stranded', 'need a way home',
        'need a way to get', 'get me moving', 'figure out transport',
    ]
    ride_suppress = ['wild ride', 'what a ride', 'roller coaster ride',
                     'pick me up off', 'pick me up some', 'pick me up a ',
                     'uber eats', 'uber surge', 'uber pricing',
                     'live in an uber', 'uber everywhere']
    has_ride = any(w in t for w in ride_words)
    ride_suppressed = any(s in t for s in ride_suppress)
    if has_ride and not ride_suppressed:
        fired.append("ride")
        scores["ride"] = 0.85

    # ── Food order ──
    food_words = [
        'order food', 'order pizza', 'order from', 'doordash',
        'uber eats', 'ubereats', 'grubhub', 'postmates',
        'order on', 'get delivery', 'food delivery',
        'order some', 'order me', 'let\'s order', "let's order",
        'mangwa', 'khana order',
    ]
    food_suppress = ['order me some motivation', 'order me some sleep',
                     'order me some happiness', 'order of operations',
                     'in order to', 'out of order', 'law and order',
                     'usual order', 'the usual']
    has_food = any(w in t for w in food_words)
    food_suppressed = any(s in t for s in food_suppress)
    if has_food and not food_suppressed:
        fired.append("food_order")
        scores["food_order"] = 0.85

    # ── Contact ──
    contact_words = [
        'call ', 'text ', 'message ', 'reach out', 'get in touch',
        'phone ', 'ring ', 'facetime', 'whatsapp ',
    ]
    contact_suppress = ['i call that', 'what i call', 'call it',
                        'call a cab', 'call an uber', 'close call',
                        'remind me to call', "don't forget to call"]
    has_contact = any(w in t for w in contact_words)
    contact_suppressed = any(s in t for s in contact_suppress)
    if has_contact and not contact_suppressed:
        fired.append("contact")
        scores["contact"] = 0.80

    # ── Alarm ──
    alarm_words = [
        'set alarm', 'set an alarm', 'alarm at', 'alarm for',
        'wake me', 'timer for', 'set timer', 'buzz me at',
    ]
    if any(w in t for w in alarm_words):
        fired.append("alarm")
        scores["alarm"] = 0.85

    # ── Reminder ──
    reminder_words = [
        'remind me to', 'remind me about', "don't forget to",
        'dont forget to', "don't let me forget", 'dont let me forget',
        "don't forget the", 'dont forget the',
        'ping me', 'nudge me',
        'remember to ', 'gotta remember',
        'give me a reminder', 'give me a heads up',
    ]
    reminder_suppress = ['reminds me of', 'remind me why', 'remind me of', 'ping pong']
    has_reminder = any(w in t for w in reminder_words)
    reminder_suppressed = any(s in t for s in reminder_suppress)
    if has_reminder and not reminder_suppressed:
        fired.append("reminder")
        scores["reminder"] = 0.85

    # ── Calendar ──
    calendar_words = [
        'schedule', 'calendar', 'block off', 'block time',
        'recurring event', 'book a slot', 'pencil in',
        'save the date', 'mark my calendar',
    ]
    calendar_suppress = ['my schedule is', 'checked the calendar',
                         'calendar year', 'on the calendar']
    has_calendar = any(w in t for w in calendar_words)
    calendar_suppressed = any(s in t for s in calendar_suppress)
    if has_calendar and not calendar_suppressed:
        fired.append("calendar")
        scores["calendar"] = 0.80

    # ── Bills ──
    bills_words = [
        'pay the bill', 'pay rent', 'pay utilities', 'electricity bill',
        'electric bill', 'phone bill', 'wifi bill', 'water bill', 'gas bill',
        'internet bill', 'netflix', 'spotify', 'subscription',
        'due date', 'bill is due', 'pay insurance', 'rent is due',
        'pay the electric', 'pay the wifi', 'pay the rent',
        'credit card bill', 'pay my bill', 'pay my rent',
        'mortgage', 'loan payment', 'cable bill', 'utility bill',
    ]
    if any(w in t for w in bills_words):
        fired.append("bills")
        scores["bills"] = 0.80

    # ── Travel ──
    travel_words = [
        'book flight', 'book a flight', 'plane ticket',
        'hotel reservation', 'book hotel', 'airbnb',
        'travel plan', 'trip to', 'vacation to',
        'travel to', 'fly to', 'flight to',
        'travel next', 'plan a trip', 'planning a trip',
    ]
    travel_suppress = ['what a trip', 'trip me out', 'road trip was']
    has_travel = any(w in t for w in travel_words)
    travel_suppressed = any(s in t for s in travel_suppress)
    if has_travel and not travel_suppressed:
        fired.append("travel")
        scores["travel"] = 0.80

    latency_ms = (time.time() - t0) * 1000

    # Money enrichment
    money_info = None
    if "money" in fired:
        money_info = _enrich_money(text)

    return {
        "intents":    fired,
        "scores":     scores,
        "money":      money_info,
        "latency_ms": round(latency_ms, 2),
    }


def _enrich_money(text: str) -> dict:
    """Extract money-specific details: amount, trigger type, direction."""
    t = text.lower()

    # Amount
    amount_match = re.search(
        r'[\$₹][\d,]+(?:\.\d{1,2})?|\b\d+\s*[\$₹]'
        r'|\b\d+\s*(?:dollars?|bucks?|rupees?|rs\.?|inr|ringgit|rm|myr)\b',
        text, re.IGNORECASE)
    amount = amount_match.group(0) if amount_match else None
    if amount and re.match(r'^\d+\s*\$', amount):
        amount = '$' + amount.replace('$', '').strip()

    # Trigger type
    if any(w in t for w in ["venmo", "cashapp", "cash app", "zelle", "apple pay"]):
        trigger = "payment_app"
    elif any(w in t for w in ["split", "halves", "half", "divide", "chip in"]):
        trigger = "bill_splitting"
    elif any(w in t for w in ["owe", "owed", "pay me back", "pay back"]):
        trigger = "owing_debt"
    elif "$" in t or "₹" in t or any(w in t for w in ["dollars", "bucks", "rupees", "ringgit", " rm ", " myr"]):
        trigger = "direct_amount"
    else:
        trigger = "general_money"

    # Direction
    offer_patterns = [
        "i owe", "i'll pay", "i'll send", "let me pay", "let me send",
        "i can pay", "i'll venmo", "i'll cashapp", "i'll zelle",
        "let me venmo", "let me cashapp", "let me zelle",
        "lemme venmo", "lemme cashapp", "lemme zelle",
        "shall i send", "should i send", "want me to send",
        "send you", "pay you back", "i'll cover", "let me cover",
        "venmo you", "cashapp you", "zelle you",
        "i got you", "my treat", "i'll get this", "on me",
        "sending you", "lemme pay", "ima send", "ima pay",
        "i already paid", "i already sent", "i already venmo",
        "i paid", "i sent", "i covered", "i transferred",
        "will send", "sending now", "will pay", "will venmo",
        "will cashapp", "will zelle", "will transfer",
        "paying now", "paying you", "transferring now",
        "just sent", "just paid", "just venmo", "just zelle",
        "ok sending", "okay sending", "ok paying", "okay paying",
    ]
    request_patterns = [
        "you owe", "owe me", "pay me", "send me", "pay up",
        "venmo me", "cashapp me", "zelle me", "where's my",
        "give me", "front me", "spot me", "cover me",
        "you still owe", "pay me back", "need my money",
    ]
    split_patterns = ["split", "halves", "half", "divide", "chip in", "go dutch"]

    direction = "request"
    for p in offer_patterns:
        if p in t:
            direction = "offer"
            break
    else:
        for p in request_patterns:
            if p in t:
                direction = "request"
                break
        else:
            for p in split_patterns:
                if p in t:
                    direction = "split"
                    break

    return {
        "detected_amount": amount,
        "trigger_type":    trigger,
        "direction":       direction,
    }


# ═══════════════════════════════════════════════════════════════════════
# Intent Targeting — who should see the popup
# Backend can't read messages (encrypted), so we must return this info.
# ═══════════════════════════════════════════════════════════════════════

_TARGET_SELF = [
    re.compile(r'\b(?:i\s+need|i\s+want|i\s+gotta|i\s+have\s+to|i\s+need\s+to|i\'?m\s+(?:gonna|going\s+to)|lemme|let\s+me|i\'?ll)\b', re.IGNORECASE),
    re.compile(r'\bfor\s+me\b', re.IGNORECASE),
    re.compile(r'\bmy\s+(?:uber|cab|ride|flight|alarm|reminder|bill|food|order|appointment|meeting)\b', re.IGNORECASE),
]

_TARGET_OTHER = [
    re.compile(r'\b(?:(?:for|to)\s+)?(?:him|her|them|he|she|they)\b', re.IGNORECASE),
    re.compile(r'\b(?:book|get|order|set|remind|call|save|send|pay|schedule)\s+(?:(?:my\s+)?(?:friend|bro|dude|man|buddy|mom|dad|sister|brother|roommate|boss))\b', re.IGNORECASE),
    re.compile(r'\b(?:book|get|order|set|remind|call|save|send|pay|schedule)\s+(?:\w+\s+)?(?:for\s+)?(?:him|her|them)\b', re.IGNORECASE),
]

_TARGET_GROUP = [
    re.compile(r'\b(?:let\'?s|we\s+should|everyone|all\s+of\s+us|for\s+(?:the\s+)?(?:group|team|everyone|us))\b', re.IGNORECASE),
    re.compile(r'\b(?:split|divide|chip\s+in|go\s+dutch|share)\b', re.IGNORECASE),
]

_TARGET_REQUEST_ME = [
    re.compile(r'\b(?:book|get|order|grab|call|find|set|schedule)\s+me\b', re.IGNORECASE),
    re.compile(r'\b(?:me\s+(?:an?\s+)?(?:uber|cab|lyft|ride|taxi|ola))\b', re.IGNORECASE),
    re.compile(r'\b(?:send|venmo|cashapp|zelle|pay|transfer)\s+me\b', re.IGNORECASE),
    re.compile(r'\b(?:remind|wake)\s+me\b', re.IGNORECASE),
]

# ── Who pays, when the firing message is a bare acceptance ──
#
# "Sure" carries no information about who parts with the money. The answer is in the
# message being accepted, and the two shapes point in OPPOSITE directions:
#
#   REQUEST  "can you send me 100"  -> "Sure"   the ACCEPTER pays
#   OFFER    "shall I send you 100" -> "Sure"   the OFFERER pays
#
# Phase 5b used to assume the accepter always pays. That is right for requests and
# backwards for offers, so an offer accepted with "Sure" put the payment sheet in front
# of the person RECEIVING the money. Reported from a real chat: "Shall I paypal you 100
# rupees?" / "Sure" showed the prompt to the payee.
#
# The apps are listed explicitly rather than as a generic verb because "paypal me 100"
# and "shall I paypal you" only differ by the me/you pronoun.
_PAY_APP = (r"(?:send|pay|transfer|venmo|cashapp|cash\s?app|zelle|paypal|gpay|"
            r"google\s?pay|paytm|phonepe|upi|spot|lend|give|shoot|drop)")
_RIDE_ACT = r"(?:book|get|order|grab|call|arrange)"

# The SPEAKER of this message is the one who will act (offer).
_OFFER_BY_SPEAKER = [
    re.compile(rf"\b(?:shall|should|can|may)\s+i\s+(?:just\s+)?{_PAY_APP}\s+(?:you|u)\b", re.IGNORECASE),
    re.compile(rf"\b(?:let\s+me|lemme)\s+{_PAY_APP}\s+(?:you|u)\b", re.IGNORECASE),
    re.compile(rf"\bwant\s+me\s+to\s+{_PAY_APP}\b", re.IGNORECASE),
    re.compile(rf"\bi(?:'?ll|ll|'?m|m)?\s*(?:will|gonna|going\s+to)?\s*{_PAY_APP}\s+(?:you|u)\b", re.IGNORECASE),
    re.compile(rf"\b(?:shall|should|can)\s+i\s+{_RIDE_ACT}\s+(?:you|u)\b", re.IGNORECASE),
    re.compile(rf"\b(?:let\s+me|lemme)\s+{_RIDE_ACT}\s+(?:you|u)\b", re.IGNORECASE),
]
# The OTHER person is the one who will act (request).
_REQUEST_OF_OTHER = [
    re.compile(rf"\b{_PAY_APP}\s+me\b", re.IGNORECASE),
    re.compile(rf"\bcan\s+(?:you|u)\s+(?:please\s+)?{_PAY_APP}\b", re.IGNORECASE),
    re.compile(rf"\b{_RIDE_ACT}\s+me\b", re.IGNORECASE),
    re.compile(r"\byou\s+(?:still\s+)?owe\s+me\b", re.IGNORECASE),
]


# A request every member owes separately, rather than one debt with one payer.
_DIVISIBLE = re.compile(
    r"\b(?:each|per\s+(?:head|person)|your\s+shares?|their\s+shares?|"
    r"split\s+(?:it\s+)?(?:\d+\s+ways|between|among)|\d+\s+ways|"
    r"everyone\s+(?:owes|sends?|pays?)|you\s+(?:two|three|four|all)|"
    r"all\s+of\s+(?:you|us))\b", re.IGNORECASE)

# "split 5 ways", "between 4 of us", "4 way split" — a headcount stated in the message.
_WAYS = re.compile(r"\b(?:split\s+)?(\d{1,2})\s*[- ]?(?:way|ways)\b|"
                   r"\b(?:between|among)\s+(?:the\s+)?(\d{1,2})\b", re.IGNORECASE)

_AMT_NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)")

# The figure is already a per-person share — never divide it again.
_PER_PERSON_AMT = re.compile(
    r"\d[\d,]*\s*(?:rs|rupees|dollars|bucks|\$)?\s*(?:each|per\s+(?:head|person)|a\s?piece)"
    r"|(?:each|per\s+(?:head|person))\s*[:=]?\s*\d", re.IGNORECASE)
# The figure is the whole bill, so a share can be derived from it.
_TOTAL_AMT = re.compile(
    r"\b(?:total|in\s+all|altogether|came\s+to|cost|costs|was|is|bill\s+was)\b", re.IGNORECASE)


def _per_person_share(text: str, total: str, participants: int = None):
    """Turn a stated TOTAL into a per-person figure, or None if we cannot be sure.

    "trip was 5000, send me your shares" is 1000 each in a room of five and 2500 in a
    room of two. The message alone never says which, so the headcount has to come from
    the message ("split 5 ways") or from the caller (`participants`).

    Returns None rather than guessing. An empty amount field is a minor annoyance; a
    confidently wrong one on a payment screen invites sending the wrong sum.
    """
    if not total:
        return None
    # Only a TOTAL can be divided. "5000, thats 1000 each" already states the share, and
    # dividing it again gave 200 — worse than the blank field this function exists to
    # avoid. Require the figure to be marked as a total, and bail if the message marks
    # it per-person.
    if _PER_PERSON_AMT.search(text or ""):
        return None
    if not _TOTAL_AMT.search(text or ""):
        return None
    m = _AMT_NUM.search(str(total))
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return None

    n = None
    w = _WAYS.search(text or "")
    if w:
        n = int(w.group(1) or w.group(2))
    elif participants and participants > 1:
        n = participants
    if not n or n < 2:
        return None

    share = value / n
    if share != int(share):
        return None                      # uneven split — let a person decide, not us
    keep = str(total).replace(m.group(1), "")     # preserve the currency mark
    return f"{keep.strip()}{int(share)}".strip() or str(int(share))


def _norm_slot(v) -> str:
    """Compare slot values by their digits/words, not their formatting.

    "100 rupees", "₹100", "$100" and "100" are the same payment written four ways. A raw
    string compare would call each one a new action and prompt again for it.
    """
    if v in (None, "", []):
        return ""
    s = str(v).lower()
    digits = re.sub(r"[^0-9.]", "", s)
    return digits or re.sub(r"[^a-z]", "", s)


def _resolve_payer(room_id: str, sender: str) -> str:
    """Who parts with the money — the sender of this message, or the other party.

    Walks back to the last message from a different sender and asks which shape it was.
    Only that message matters: it is the thing the current message is accepting. Falls
    back to `sender` when nothing matches, which keeps the previous behaviour (accepter
    pays) for every shape this does not recognise.

    Returning the payer's ID rather than a bool lets the duplicate-fire check key on the
    payer. An offer and its acceptance come from DIFFERENT senders but name the SAME
    payer, so a sender-keyed check sees two unrelated fires and lets both through —
    that is why "shall I send you 100" / "Sure" produced two prompts for one payment.
    """
    if not room_id or not sender:
        return sender
    # Scan back for the most recent message that actually PROPOSES something. Stopping
    # at the immediately preceding turn was too strict: filler between the proposal and
    # the acceptance hid it. "can i pay you 100 rupees" / "??" / "Sure" resolved off the
    # "??" and pointed the sheet at the payee.
    #
    # Own messages count too — if I offered and I am now saying "Sending now", I am
    # still the payer, and both fires must resolve to the same person or the duplicate
    # check compares two different keys and lets both through.
    for entry in reversed(conversation_ctx.get_window(room_id, size=10)):
        who = str(entry.get("sender"))
        t = entry.get("text") or ""
        if any(p.search(t) for p in _OFFER_BY_SPEAKER):
            return who                    # whoever offered is the one who pays
        if any(p.search(t) for p in _REQUEST_OF_OTHER):
            # Someone asked to be paid; the payer is whoever answers, not the asker.
            return sender if who != str(sender) else sender
    return sender


def detect_target(text: str, intents: list, money_info: dict = None) -> dict:
    """
    Detect who the intent is directed at.

    Returns:
        {
            "show_to": "sender" | "others" | "group",
            "reason": "brief explanation"
        }

    Logic:
        - Money with direction "request" (venmo ME) → show to others (they pay)
        - Money with direction "offer" (I'll venmo YOU) → show to sender (they pay)
        - Money with direction "split" → show to group
        - "book/get/order ME" → show to sender (they want the action done for them)
        - "for him/her/them" → show to sender (they're requesting on behalf of someone)
        - "let's / everyone / split" → show to group
        - Default → show to sender
    """
    tl = text.lower()

    # Money has its own direction system
    if "money" in intents and money_info:
        direction = money_info.get("direction", "request")
        if direction == "offer":
            return {"show_to": "sender", "reason": "sender_offering_to_pay"}
        elif direction == "split":
            return {"show_to": "group", "reason": "bill_splitting"}
        else:
            return {"show_to": "others", "reason": "sender_requesting_payment"}

    # Group actions
    if any(p.search(tl) for p in _TARGET_GROUP):
        return {"show_to": "group", "reason": "group_action"}

    # "book ME an uber", "get ME food", "remind ME" → sender wants it for themselves
    # But the popup should still show to sender — they expressed the need
    if any(p.search(tl) for p in _TARGET_REQUEST_ME):
        return {"show_to": "sender", "reason": "sender_requesting_for_self"}

    # "for him/her/them", "book my friend" → sender requesting for someone else
    # Show to sender — they're the one initiating the action
    if any(p.search(tl) for p in _TARGET_OTHER):
        return {"show_to": "sender", "reason": "sender_requesting_for_other"}

    # "I need", "I gotta", "I'm gonna" → sender expressing own need
    if any(p.search(tl) for p in _TARGET_SELF):
        return {"show_to": "sender", "reason": "sender_expressing_need"}

    # Default: show to sender (they said it, they probably need it)
    return {"show_to": "sender", "reason": "default"}


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — Conversation Context
# ═══════════════════════════════════════════════════════════════════════

class ConversationContext:
    """Track recent messages per room. Provides last 3 messages (within 5 min)
    for model-level context concatenation via </s> separator."""

    # Must be at least (conversation model window - 1), since the model needs that many
    # history messages plus the current one. v2 trains at window 10, so 8 would have
    # silently starved it of context with no error anywhere. 16 leaves headroom.
    CONTEXT_WINDOW = 16
    DECAY_SECONDS = 120

    def __init__(self):
        self.rooms: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.CONTEXT_WINDOW))

    def add(self, room_id: str, text: str, intents: list, scores: dict,
            sender: str = None):
        self.rooms[room_id].append({
            "text": text,
            "intents": intents,
            "scores": scores,
            "sender": sender,
            "ts": time.time(),
        })

    def get_window(self, room_id: str, size: int = 5) -> list:
        """Recent (sender, text) pairs for the conversation classifier.

        Unlike get_prev_messages this keeps the sender, which the classifier needs to
        work out who is answering whom. Entries recorded before senders were tracked
        have sender None and are skipped, since an unattributable message would corrupt
        the speaker mapping.

        Uses CONV_CONTEXT_TIME_CAP (4h), NOT the 5-minute CONTEXT_TIME_CAP used for the
        intent model's prefix. For this model the window is the entire memory — at 5
        minutes a request made seven minutes ago disappears and the reply is judged with
        no context, scoring 0.03 where it should score 0.997.
        """
        now = time.time()
        out = []
        for entry in self.rooms.get(room_id, []):
            if now - entry["ts"] > CONV_CONTEXT_TIME_CAP:
                continue
            if entry.get("sender") is None:
                continue
            # ts rides along so slot recovery can scope itself to messages that came
            # AFTER the request being answered. build_window ignores extra keys.
            out.append({"sender": entry["sender"], "text": entry["text"],
                        "ts": entry["ts"]})
        return out[-size:] if out else []

    def get_prev_messages(self, room_id: str, max_msgs: int = 3) -> list:
        """Return up to 3 recent message texts within the time cap for model context."""
        now = time.time()
        msgs = []
        for entry in self.rooms.get(room_id, []):
            age = now - entry["ts"]
            if age <= CONTEXT_TIME_CAP:
                msgs.append(entry["text"])
        return msgs[-max_msgs:] if msgs else []

    def recent_intents(self, room_id: str) -> dict:
        """Weighted bag of intents from recent context. More recent = higher weight."""
        now = time.time()
        weights: dict[str, float] = {}
        for entry in self.rooms.get(room_id, []):
            age = now - entry["ts"]
            if age > self.DECAY_SECONDS:
                continue
            recency = 1.0 - (age / self.DECAY_SECONDS)
            for intent in entry["intents"]:
                score = entry["scores"].get(intent, 0.8)
                weights[intent] = max(weights.get(intent, 0), score * recency)
        return weights


conversation_ctx = ConversationContext()


class RequestMeta:
    """Carries `triggered_by` for the classifier path. Decides nothing.

    The classifier reads a window and answers "should this fire" — it has no notion of
    which earlier message the fire relates to. But the mobile apps need exactly that:
    API_MOBILE.md tells iOS and Android to read the amount from
    `conversation_state.triggered_by.slots`, because the acceptance ("ok sending")
    never contains the amount — the request does. Without it the payment prompt appears
    with no amount and no recipient.

    So this records requests as they go past and hands the most recent one back when a
    fire happens. It is deliberately NOT a decision-making pending store: nothing here
    can cause or suppress a fire, so it cannot reintroduce the failure modes the rule
    layer had. If it is wrong, the prompt is missing metadata; the detection is
    unaffected.
    """

    MAX_PER_ROOM = 8

    def __init__(self):
        self.rooms: dict[str, list] = defaultdict(list)
        # room -> {(intent, sender): ts} of the last fire, for duplicate suppression.
        # Cleared when a new request arrives, so the next commitment counts as a real
        # fire.
        #
        # Keyed by SENDER as well as intent. Keyed by room alone it suppressed the
        # second person in "paid the wifi bill, 800 each you two" / "sending now" /
        # "same, sending mine too" — the flatmate's commitment was read as an echo of
        # mine and he silently got no payment sheet. Two people paying the same request
        # is ordinary; one person restating their own commitment is the thing to
        # suppress, and that is a per-speaker question.
        self.last_fire: dict[str, dict] = defaultdict(dict)
        # Same timestamps, but NEVER cleared. This is the "everything before here is
        # settled" line used to scope slot recovery. It has to survive clear_fired():
        # a new request is exactly when the boundary matters most, and wiping it let a
        # paid-off counter-offer leak into the next payment.
        self.settled_at: dict[str, dict] = defaultdict(dict)
        # The slots the last fire resolved to, and whether a NEW request has arrived
        # since. One negotiation can fire twice — a counter-offer flips ownership, the
        # asker accepts ("yeah 3000 helps") and then the payer confirms ("sending").
        # Both are the same payment and must show the same figure. Scoping the second
        # one to "after the last fire" hid the counter-offer and put the ORIGINAL 5000
        # on the sheet instead of the agreed 3000. So: if no new request has arrived,
        # this fire is a continuation — reuse what the last one resolved.
        self.last_resolved: dict[str, dict] = defaultdict(dict)
        self.new_request_since: dict[str, dict] = defaultdict(dict)

    def continues_last(self, room_id, intent):
        """Is this fire part of the same negotiation as the previous one?"""
        return (bool(self.last_resolved.get(room_id, {}).get(intent))
                and not self.new_request_since.get(room_id, {}).get(intent))

    def remember_resolved(self, room_id, intent, slots):
        if room_id and slots:
            self.last_resolved[room_id][intent] = dict(slots)

    def mark_fired(self, room_id, intent, sender):
        """Record a fire for duplicate suppression. Does NOT move the settled line."""
        if room_id:
            self.new_request_since[room_id][intent] = False
            self.last_fire[room_id][(intent, sender)] = time.time()

    def mark_settled(self, room_id, intent):
        """Close off a negotiation — only call when a real request was answered.

        Kept separate from mark_fired because the model sometimes fires on a message
        that answers nothing ("Sure but send it now as i need it urgently" scores 1.00
        but takes no request, since the speaker is the one who asked). Advancing the
        settled line there hid the "I can only do 60$" counter-offer from the fire that
        followed, and the payment sheet showed the original 100 instead of the agreed
        60. A fire that resolves no request settles nothing.
        """
        if room_id:
            # per-room, not per-speaker: it marks where a negotiation closed, which is
            # a property of the conversation.
            self.settled_at[room_id][intent] = time.time()

    def recently_fired(self, room_id, intent, sender):
        ts = self.last_fire.get(room_id, {}).get((intent, sender))
        return ts is not None and (time.time() - ts) <= CONV_CONTEXT_TIME_CAP

    def clear_fired(self, room_id, intent):
        """A new request reopens the intent for everyone in the room."""
        room = self.last_fire.get(room_id)
        if room:
            for key in [k for k in room if k[0] == intent]:
                room.pop(key, None)

    def record(self, room_id, intent, sender, text, message_id, slots, divisible=False):
        if not room_id or not sender:
            return
        # A new request starts a new negotiation — the next fire must resolve its own
        # figures rather than inherit the previous payment's.
        self.new_request_since[room_id][intent] = True
        q = self.rooms[room_id]
        q.append({"intent": intent, "sender": sender, "text": text,
                  "message_id": message_id, "slots": slots or None,
                  "ts": time.time(),
                  # A divisible request is owed by every member separately ("1000 each"),
                  # so it must survive being answered. take() consumes an ordinary
                  # request, which is right for "lend me 500" — one debt, one payer —
                  # and wrong for a split: the second, third and fourth payers found
                  # nothing left to read and their prompts opened with no amount.
                  "divisible": bool(divisible),
                  "taken_by": set()})
        del q[:-self.MAX_PER_ROOM]

    def has_open(self, room_id, intent, sender):
        """Is there an unanswered request for this intent from someone else?"""
        now = time.time()
        return any(e["intent"] == intent and e["sender"] != sender
                   and now - e["ts"] <= CONV_CONTEXT_TIME_CAP
                   for e in self.rooms.get(room_id) or [])

    def take(self, room_id, intent, sender):
        """Most recent matching request from someone else. None if absent.

        Ordinary requests are consumed. A divisible one ("1000 each") is owed by every
        member separately, so it stays in the queue and only records who has answered —
        otherwise the first payer eats it and everyone after them gets a prompt with no
        amount. Each sender may still take it only once, which keeps a single person
        restating their own commitment from resolving it twice.
        """
        q = self.rooms.get(room_id)
        if not q:
            return None
        now = time.time()
        for i in range(len(q) - 1, -1, -1):
            e = q[i]
            if e["intent"] != intent or e["sender"] == sender:
                continue
            if now - e["ts"] > CONV_CONTEXT_TIME_CAP:
                continue
            if e.get("divisible"):
                if sender in e.get("taken_by", ()):
                    continue
                e.setdefault("taken_by", set()).add(sender)
                return e
            return q.pop(i)
        return None


request_meta = RequestMeta()

# ── Conversation classifier (experimental, off by default) ──
# Set PAYCHAT_CONV_CLASSIFIER=1 to let the conversation-level model decide money/ride
# instead of the rule-based state machine. Off by default so the shipped path is
# untouched; the flag exists so both can be measured through the same server on the
# same eval, which is the only way to answer "do the rules still earn their place".
#
# NOTE: the pending-request splice (build_window's pending_texts) is deliberately NOT
# used here. Zero training windows contain the "[earlier]" marker it renders, so
# feeding it would be out-of-distribution. See task #40.
CONV_CLASSIFIER_ON = os.environ.get("PAYCHAT_CONV_CLASSIFIER", "").strip() in ("1", "true", "yes")
conv_classifier = None
if CONV_CLASSIFIER_ON:
    try:
        from conv_classifier import ConversationClassifier, build_window
        # Resolved against this file, not the working directory, so the server does not
        # silently disable itself depending on where it was started from.
        conv_classifier = ConversationClassifier(
            os.environ.get("PAYCHAT_CONV_MODEL",
                           str(Path(__file__).resolve().parent / "conv_model")))
        if not conv_classifier.ok:
            logger.error("PAYCHAT_CONV_CLASSIFIER=1 but the model failed to load — "
                         "refusing to fall back silently")
            raise SystemExit("conversation classifier requested but unavailable")
        logger.info("conversation classifier ENABLED — money/ride decided by the model")
    except ImportError as e:
        raise SystemExit(f"conversation classifier requested but import failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — Slot Extraction
# ═══════════════════════════════════════════════════════════════════════

# Common first names for recipient extraction
_COMMON_NAMES = {
    "rahul", "sarah", "jake", "mom", "dad", "bro", "sis", "john", "jane",
    "alex", "sam", "mike", "chris", "david", "emma", "priya", "akash",
    "amrit", "samyak", "nick", "anna", "lisa", "tom", "ben", "max",
    "raj", "ravi", "amit", "neha", "pooja", "ankit", "vinay", "kamal",
    "maria", "james", "robert", "mary", "mark", "paul", "kate",
}

_TIME_PATTERNS = [
    # "for" belongs here as much as "at" — "book a cab to hsr for 7:30" is ordinary
    # phrasing and was returning no time at all. The trailing lookahead keeps it from
    # swallowing "for 20 mins", which is a duration and matched below.
    # The \s* sits INSIDE the optional am/pm group on purpose: outside it, "at 8 please"
    # captured "8 " with a trailing space and shipped that to the client.
    (r"\b(?:at|by|around|before|after|for)\s+(\d{1,2}(?::\d{2})?(?:\s*(?:am|pm|AM|PM))?)\b"
     r"(?!\s*(?:minutes?|mins?|hours?|hrs?|days?|weeks?))", "specific_time"),
    (r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))\b", "specific_time"),
    # "tonight 9" / "tomorrow 8" — a bare clock time straight after a day word, with no
    # preposition. Combines with the relative_day match below into "tonight at 9".
    (r"\b(?:tomorrow|tonight|today|this\s+(?:evening|morning|afternoon))\s+(?:at\s+)?"
     r"(\d{1,2}(?::\d{2})?(?:\s*(?:am|pm|AM|PM))?)\b", "specific_time"),
    (r"\b(tomorrow|tonight|today|this\s+(?:evening|morning|afternoon))\b", "relative_day"),
    (r"\b(?:in|for)\s+(\d+)\s*(minutes?|mins?|hours?|hrs?)\b", "relative_offset"),
    (r"\b(\d+)\s*(?:minutes?|mins?|hours?|hrs?)\s+(?:from\s+now)\b", "relative_offset"),
    (r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "day_of_week"),
    (r"\b(next\s+week|this\s+week|this\s+weekend)\b", "relative_week"),
    (r"\b(\d{1,2}(?::\d{2})?)\s*(?:o'?clock)\b", "specific_time"),
]

_DEST_PATTERNS = [
    (r"\bto\s+(the\s+)?(airport|station|mall|office|hospital|gym|school|college|university|downtown|club|park|store|shop|market|theater|cinema|church|temple|mosque|library|museum|restaurant|cafe|hotel)\b", "place"),
    (r"\b(?:get\s+me\s+|take\s+me\s+|drop\s+me\s+(?:at|to)\s+|go\s+to\s+|going\s+to\s+|head\s+to\s+)([\w\s]{2,25})\b", "freeform"),
    (r"\b(?:get\s+me\s+|need\s+(?:a\s+)?(?:ride|way)\s+)home\b", "home"),
    (r"\b(?:to|from)\s+([A-Z][\w]+(?:\s+[A-Z][\w]+)*)\b", "proper_noun"),
]

_FOOD_PATTERNS = [
    (r"\border\s+(?:me\s+)?(?:a\s+|some\s+)?([\w\s]+?)(?:\s+from|\s+on|\s*$)", "food_item"),
    (r"\b(?:from|on)\s+(doordash|uber\s*eats|grubhub|postmates|dominos?|mcdonalds?|chipotle|subway|pizza\s+hut|taco\s+bell|wendys?|chick-fil-a)\b", "restaurant"),
    (r"\b(pizza|burger|sushi|tacos?|noodles?|ramen|biryani|pasta|sandwich|wings?|fries|salad|chinese|indian|thai|mexican|italian)\b", "cuisine"),
]


_INTENT_SLOT_KEYS = {
    "money":      ["recipient", "amount", "note"],
    "ride":       ["destination", "pickup", "time"],
    "travel":     ["destination", "pickup", "time"],
    "food_order": ["food"],
    "contact":    ["recipient", "phone"],
    "alarm":      ["time"],
    "reminder":   ["task", "time"],
    "calendar":   ["event", "time"],
    "bills":      ["bill_name", "amount"],
}


def extract_slots(text: str, intents: list, room_id: str = None, prev_messages: list = None) -> dict:
    """Extract structured slots based on detected intents."""
    slots = {}
    for intent in intents:
        for key in _INTENT_SLOT_KEYS.get(intent, []):
            if key not in slots:
                slots[key] = None
    t = text.strip()
    tl = t.lower()

    # ── Recipient (for money, contact, ride) ──
    if any(i in intents for i in ["money", "contact"]):
        recipient = _extract_recipient(t, tl)
        if not recipient and room_id:
            recipient = _resolve_pronoun_recipient(tl, room_id)
        if recipient:
            slots["recipient"] = recipient

    # ── Amount (for money, bills) ──
    if any(i in intents for i in ["money", "bills"]):
        amount = _extract_amount(t)
        if amount:
            slots["amount"] = amount

    # ── Time (for alarm, reminder, calendar, ride, travel) ──
    if any(i in intents for i in ["alarm", "reminder", "calendar", "ride", "travel"]):
        time_slot = _extract_time(tl, intents)
        if time_slot:
            slots["time"] = time_slot

    # ── Pickup & Destination (for ride, travel) ──
    if any(i in intents for i in ["ride", "travel"]):
        ride_slots = _extract_ride_slots(t, prev_messages=prev_messages)
        if ride_slots.get("destination"):
            slots["destination"] = ride_slots["destination"]
        if ride_slots.get("pickup"):
            slots["pickup"] = ride_slots["pickup"]

    # ── Food items (for food_order) ──
    if "food_order" in intents:
        food = _extract_food(tl)
        if food:
            slots["food"] = food

    # ── Task (for reminder) ──
    if "reminder" in intents:
        task = _extract_reminder_task(tl)
        if task:
            slots["task"] = task

    # ── Note (for money — what it's for) ──
    if "money" in intents:
        note = _extract_money_note(tl)
        if note:
            slots["note"] = note

    # ── Phone number (for contact) ──
    if "contact" in intents:
        phone = _extract_phone(t)
        if phone:
            slots["phone"] = phone
        if "recipient" not in slots:
            name = _extract_contact_name(t, tl)
            if name:
                slots["recipient"] = name

    # ── Bill name (for bills) ──
    if "bills" in intents:
        bill = _extract_bill_name(tl)
        if bill:
            slots["bill_name"] = bill

    # ── Event name (for calendar) ──
    if "calendar" in intents:
        event = _extract_calendar_event(t, tl)
        if event:
            slots["event"] = event

    return slots


def _extract_recipient(text: str, text_lower: str) -> Optional[str]:
    patterns = [
        r"(?:venmo|cashapp|zelle|pay|send(?:\s+money)?|text|call|message|reach|contact|ring|facetime|whatsapp)\s+(?:to\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:tell|ask|ping|remind)\s+([A-Z][a-z]+)",
        r"([A-Z][a-z]+)\s+(?:owes?\s+me|owes?\s+us|owe\s+him|owe\s+her)",
        r"(?:to|for|from)\s+([A-Z][a-z]+)(?:\s|$|,|\.|!|\?)",
        r"\b(mom|dad|bro|sis)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            name = m.group(1).strip()
            if name.lower() not in {"the", "a", "an", "my", "your", "our", "his", "her",
                                     "me", "you", "us", "it", "this", "that", "some",
                                     "uber", "lyft", "venmo", "doordash", "pizza"}:
                return name
    for name in _COMMON_NAMES:
        if re.search(r'\b' + name + r'\b', text_lower):
            return name.capitalize()
    return None


def _resolve_pronoun_recipient(text_lower: str, room_id: str) -> Optional[str]:
    """If text uses a pronoun (him/her/them), look up recent context for a name."""
    pronouns = re.search(r'\b(him|her|them|he|she|they)\b', text_lower)
    if not pronouns:
        return None
    for entry in reversed(list(conversation_ctx.rooms.get(room_id, []))):
        prev_text = entry.get("text", "")
        m = re.search(r'\b([A-Z][a-z]{2,})\b', prev_text)
        if m:
            name = m.group(1)
            if name.lower() not in {"the", "uber", "lyft", "venmo", "pizza", "doordash",
                                     "how", "can", "get", "set", "don", "send", "pay", "book"}:
                return name
    return None


# Amounts that appear while NEGOTIATING, which _extract_amount does not catch — it only
# recognises request phrasings ("lend me 2000", "send me 200"). A counter-offer says
# "i can only do 1000" or "1000 works", and without these the prompt keeps showing the
# figure originally asked for.
#
# Deliberately narrow: it must match haggling, not any number in the conversation.
# "dinner was 900 btw" after a request for 200 must NOT override the 200 — that is a
# fact being reported, not a new offer.
_COUNTER_AMOUNT = re.compile(
    r"\b(?:can\s+(?:only\s+)?do|i(?:'?ll| will)\s+do|make\s+it|how\s+about|"
    r"lets\s+say|say)\s+(?:\$|₹|rs\.?\s*)?(\d[\d,]*)"
    r"|(?:\$|₹|rs\.?\s*)?(\d[\d,]*)\s+(?:works|is\s+fine|sounds\s+(?:good|fine)|"
    r"then|it\s+is)\b",
    re.IGNORECASE)


def _negotiated_amount(text: str) -> Optional[str]:
    """An amount offered or accepted mid-negotiation, or None."""
    m = _COUNTER_AMOUNT.search(text)
    if not m:
        return None
    val = m.group(1) or m.group(2)
    if not val:
        return None
    # The capture class is [\d,]* to allow "1,500", which also swallows a trailing
    # comma: "how about 3000, thats all i have" produced "$3000," on the payment sheet.
    val = val.rstrip(",").replace(",", "")
    return f"${val}" if val else None


# A destination restated after the original request. The ride equivalent of a
# counter-offer: "book me a cab to koramangala" -> "actually make it indiranagar".
_COUNTER_DEST = re.compile(
    r"\b(?:make\s+it|change\s+it\s+to|change\s+to|switch\s+to|"
    r"actually\s+(?:make\s+it\s+|go\s+to\s+)?|lets\s+(?:do|say|make\s+it)|"
    r"instead\s+(?:go\s+to\s+)?)\s*"
    r"([A-Za-z][\w'&-]*(?:\s+[\w'&-]+){0,2})",
    re.IGNORECASE)

# "where to?" / "where do you want to go" — the reply is the destination.
_WHERE_Q = re.compile(
    # "where to", "where u going", "where r u headed", "where do you want to go",
    # "which station", "to where" — chat asks this a dozen ways.
    r"\bwhere\s+(?:to|at|u\b|you\b|r\s+u\b|are\s+you\b|do\s+you\b|should\s+i\b|"
    r"we\s+going\b|is\s+that\b)|"
    r"\bto\s+where\b|\bwhich\s+(?:one|place|station|airport|mall|address)\b|"
    r"\bwhat'?s?\s+the\s+(?:address|destination|location|place)\b|"
    r"\bpick\s+(?:you|u)\s+up\s+where\b", re.IGNORECASE)

# Short replies that answer something other than "where to", plus the filler that
# "make it ___" attracts. "yeah, pls make it rn if u can" matched _COUNTER_DEST and
# produced destination "Rn If U" — a booking sent to a place that does not exist.
_NOT_A_PLACE = {
    "yes", "no", "yeah", "yep", "nah", "sure", "ok", "okay", "k", "bet", "fine",
    "cool", "alright", "thanks", "thx", "ty", "please", "pls", "asap", "now",
    "soon", "idk", "dunno", "maybe", "wait", "hold", "one", "sec", "lol", "lmao",
    "rn", "it", "that", "this", "them", "us", "quick", "quicker", "faster", "fast",
    "happen", "work", "possible", "official", "count", "sooner", "later", "earlier",
    "two", "three", "double", "half", "anything", "something", "whatever",
}

# "make it 6pm" / "make it 2" is a time or a count, never a destination.
_TIMEY = re.compile(r"^\d|\b(?:am|pm|o'?clock|hrs?|hours?|mins?|minutes?)\b", re.IGNORECASE)


def _norm_place(p: Optional[str]) -> Optional[str]:
    """Same casing rule clean_phrase() applies, so both paths agree."""
    if not p:
        return None
    if p.lower() in ("home", "my place", "my house", "your place"):
        return p.lower()
    return p.title() if p.islower() and len(p) > 3 else p


def _plausible_place(cand: Optional[str]) -> Optional[str]:
    """Reject filler and times that slipped through a 'make it ___' match."""
    if not cand:
        return None
    words = cand.split()
    if not words or _TIMEY.search(cand):
        return None
    # Any filler word in the phrase means it is not a place name.
    if any(w.lower().strip(".,!?") in _NOT_A_PLACE for w in words):
        return None
    return cand


def _restated_destination(text: str) -> Optional[str]:
    """A destination named after the original request, or None."""
    m = _COUNTER_DEST.search(text)
    if m:
        cand = _plausible_place(_tidy_place(m.group(1).strip()))
        if cand:
            return _norm_place(cand)
    return _norm_place(_destination_fallback(text))


def _latest_destination(hist: list, rider: str = None) -> Optional[str]:
    """Most recently named destination in the window, newest first.

    Two things break the naive "read it off the request" approach:
      1. "book me a cab to koramangala" -> "actually make it indiranagar"
      2. "can you get me a cab" -> "where to?" -> "whitefield"
    In (1) the request holds a destination that is no longer wanted; in (2) it
    holds none at all and the real one arrived in a later message.

    `rider` restricts donations to the person who asked for the ride. Without it the
    BOOKER could silently redirect the trip. Reported from a real chat:

        A: book me one now to Windsor, Ontario, I'm at London Ontario
        G: Can book from your location to the Toronto airport      <- never agreed to
        G: Okay. Booking                                           -> booked to Toronto

    Toronto is ~200km from Windsor. FIRING_RULE §6e: a counter-offer is a new proposal,
    not an acceptance, so it cannot rewrite the trip on its own. Both cases above still
    work — the rider is the one who says "actually make it indiranagar" and the one who
    answers "where to?". If the rider does agree to the booker's suggestion, their
    agreement is a later message and wins on recency.
    """
    # Check the sender of the message that DONATES the destination, rather than
    # filtering the window first. Filtering removes the booker's "where to?", and the
    # answer-detection below needs it to sit immediately before the rider's reply.
    def owns(i):
        return rider is None or str(hist[i].get("sender")) == str(rider)

    texts = [msg.get("text", "") for msg in hist]
    for i in range(len(texts) - 1, -1, -1):
        # A message only donates a destination if it is plausibly about the ride.
        # Three ways it can qualify:
        #   - it restates the destination ("actually make it indiranagar")
        #   - it answers a "where to?" question
        #   - it mentions a cab at all
        # Without the third condition an unrelated "took the dog to the vet" sitting in
        # the window became the destination, and the cab would have been booked to
        # "vet". _latest_time has had this guard since it was written; the destination
        # side was missing it.
        restated = _COUNTER_DEST.search(texts[i])
        answers_where = i > 0 and _WHERE_Q.search(texts[i - 1])
        if owns(i) and (restated or _RIDE_CONTEXT.search(texts[i])):
            found = _restated_destination(texts[i])
            if found:
                return found
        # A reply straight after "where to?" answers it. People rarely reply with the
        # bare place — "grand central pls, need to be there by 6pm" is the normal shape —
        # so cut at the first clause break and drop the trailing filler.
        if answers_where and owns(i):
            head = re.split(r"[,.;!?]| but | and | need | i'?ll | im | i'?m ",
                            texts[i].strip(), maxsplit=1)[0]
            words = [w for w in head.split() if w]
            while words and words[-1].lower().strip(".,!?") in _PLACE_TRAILING | {"pls", "please"}:
                words.pop()
            if 1 <= len(words) <= 4 and all(
                    w.replace("'", "").replace("-", "").isalpha() for w in words):
                cand = _plausible_place(" ".join(words))
                if cand:
                    return _norm_place(cand)
    return None


# Only a message that is itself about a ride can donate a pickup time. Without this
# guard "the match starts at 8" two messages earlier becomes the cab time.
_RIDE_CONTEXT = re.compile(
    r"\b(?:cab|uber|ola|lyft|rapido|taxi|ride|auto|book|booking|pick\s*up|drop)\b",
    re.IGNORECASE)


_WHEN_Q = re.compile(
    r"\bwhat\s+time\b|\bwhen\s+(?:do|did|are|is|should|for|u\b|you\b)|"
    r"\bfor\s+what\s+time\b|\bwhat\s+time\s*\?", re.IGNORECASE)


def _latest_time(hist: list) -> Optional[dict]:
    """Pickup time from the window, newest first, ride messages only.

    Mirrors _latest_amount: the recorded request is often not the message carrying the
    detail. "book a cab to hsr at 6pm" ... "actually make it koramangala" ... "ok
    booking" hands back the second message, which has no time in it.

    The ride-context guard stops "the match starts at 9" from becoming the cab time.
    The one exception is a direct answer to "what time?" — "around 8" mentions no cab
    but is unambiguously the pickup time, exactly like "whitefield" answering "where to?".
    """
    texts = [msg.get("text", "") for msg in hist]
    for i in range(len(texts) - 1, -1, -1):
        answers_when = i > 0 and _WHEN_Q.search(texts[i - 1])
        if not answers_when and not _RIDE_CONTEXT.search(texts[i]):
            continue
        t = _extract_time(texts[i].lower(), ["ride"])
        if t:
            return t
    return None


def _since(hist: list, ts: float) -> list:
    """Window trimmed to messages after `ts` — the previous fire for this intent.

    Everything that recovers a slot from the window has to be scoped, or it reaches
    back into a conversation that was already settled. Reported from live testing:

        "can you send me hundred bucks"
        "I can only do 60$ man, is that fine?"    <- counter-offer
        "Sure sending now"                        -> fires $60, correct
        "Can you send 40$ for my lunch please"    <- a NEW, unrelated request
        ...movie chat...
        "about the lunch money let send you now"  -> fired $60. Should be $40.

    The anchor is the PREVIOUS FIRE, not the request that triggered this one. Scoping
    to the request looks right and is not: the most recent request-shaped message is
    often an acceptance carrying an incidental figure ("Remaining $40 I'll arrange"),
    and anchoring there hides the real counter-offer that came before it. Once a
    payment has fired, everything up to that point is settled — that is the line.
    """
    if not ts:
        return hist
    return [m for m in hist if m.get("ts", 0) > ts]


# "how much?" — the reply is a bare number with no currency and no verb, so
# _extract_amount alone never sees it. Mirrors _WHERE_Q on the ride side.
_HOW_MUCH_Q = re.compile(
    r"\bhow\s+much\b|\bhow\s+many\b|\bwhat'?s?\s+the\s+(?:amount|total|damage)\b|"
    r"\bhow\s+much\s+(?:do\s+)?(?:you|u)\s+need\b", re.IGNORECASE)


def _latest_amount(hist: list) -> Optional[str]:
    """Most recently stated amount in the window, newest first.

    `take()` hands back the most RECENT request-shaped message, which is often not the
    one carrying the figure:

        "can you lend me 500? my wallet got stolen"   <- has the amount
        "how do you want me to get it to you?"
        "just send it to my upi"                      <- also request-shaped, no amount
        "ok, sending now"                             <- fires, triggered_by = the upi one

    so the payment sheet opened with no amount at all. Reading the window directly
    finds the figure wherever it was actually said.
    """
    texts = [m.get("text", "") for m in hist]
    for i in range(len(texts) - 1, -1, -1):
        amt = _extract_amount(texts[i])
        if amt:
            return amt
        # A bare figure answering "how much?" — "can you send me some money" /
        # "how much" / "450" / "ok sending".
        if i > 0 and _HOW_MUCH_Q.search(texts[i - 1]):
            bare = re.fullmatch(r"\s*(?:\$|₹)?\s*(\d[\d,]*(?:\.\d{1,2})?)\s*[.!]?\s*",
                                texts[i])
            if bare:
                return f"${bare.group(1).replace(',', '')}"
    return None


# A number followed by one of these is a count, not a sum: "paying attention to 3
# things" was yielding $3. Only guards the bare-number path — an explicit "$3 things"
# still reads as money.
_COUNT_NOUNS = (r"things?|times?|people|persons?|guys|kids|days?|weeks?|months?|years?|"
                r"hours?|hrs?|mins?|minutes?|seconds?|secs?|items?|ways?|reasons?|"
                r"options?|places?|stops?|bags?|rooms?|seats?|tickets?|km|kms|miles?")


def _extract_amount(text: str) -> Optional[str]:
    patterns = [
        r'(\$[\d,]+(?:\.\d{1,2})?)',
        r'(₹[\d,]+(?:\.\d{1,2})?)',
        r'(\d+(?:\.\d{1,2})?)\s*(?:dollars?|bucks?)',
        r'(\d+(?:\.\d{1,2})?)\s*(?:rupees?|rs\.?|inr)',
        r'(\d+(?:\.\d{1,2})?)\s*(?:ringgit|rm|myr)',
        r'(\d+(?:\.\d{1,2})?)\s*\$',
        r'(\d+(?:\.\d{1,2})?)\s*₹',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            # "how about 3000, thats all i have" captured "3000," and the comma went
            # onto the payment sheet.
            val = m.group(0).strip().rstrip(".,;:!?")
            # "40$" and "500₹" put the symbol after the number. Prefixing blindly gave
            # "$40$", which is what the payment sheet displayed.
            trailing = re.match(r'^([\d,]+(?:\.\d{1,2})?)\s*([$₹])$', val)
            if trailing:
                val = trailing.group(2) + trailing.group(1)
            elif re.match(r'^\d', val) and not re.search(r'(?:dollar|buck|rupee|rs\.?|inr|ringgit|rm|myr)', val, re.IGNORECASE):
                val = '$' + val
            return val
    number_words = {
        "ten": "10", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
        "hundred": "100", "thousand": "1000", "five": "5", "fifteen": "15",
    }
    for word, num in number_words.items():
        m = re.search(r'\b' + word + r'\s*(dollars?|bucks?|rupees?|rs\.?|ringgit|rm)\b',
                      text.lower())
        if m:
            # "fifty rupees" was coming back as $50. Keep the currency the user used.
            unit = m.group(1)
            symbol = "₹" if unit.startswith(("rupee", "rs")) else "$"
            return f"{symbol}{num}"
    # Bare number in money context — "front me 50", "sent me the 30"
    # -ing forms matter: "im sending you 3000 right now" is how an offer is normally
    # worded, and `send` alone does not match it — the pattern needs whitespace
    # immediately after the verb, which "sending" does not provide.
    m = re.search(
        r'\b(?:send(?:ing)?|sent|pay(?:ing)?|paid|owes?|owed|front(?:ing)?|'
        r'spot(?:ting)?|lend(?:ing)?|lent|giv(?:e|ing)|gave|venmo(?:ing)?|'
        r'cashapp(?:ing)?|zelle|cover(?:ing|ed)?|transfer(?:ring|red)?|'
        r'put\s+in|chip(?:ping)?\s+in)'
        r'\s+(?:\w+\s+){0,3}(\d[\d,]*(?:\.\d{1,2})?)\s*(?!\s*(?:' + _COUNT_NOUNS + r')\b)',
        text, re.IGNORECASE)
    if m:
        return f"${m.group(1).replace(',', '')}"
    # A per-head share carries no currency and no verb next to the figure:
    # "i covered dinner last night, 450 each" put nothing on the payment sheet.
    # This is how a split gets written in chat, so it is worth its own pattern.
    m = re.search(
        r'\b(\d[\d,]*(?:\.\d{1,2})?)\s*(?:each|apiece|per\s+(?:head|person|pax)|'
        r'a\s+(?:head|piece|pop))\b', text, re.IGNORECASE)
    if m:
        return f"${m.group(1).replace(',', '')}"
    # A stated TOTAL with no payment verb anywhere near it. "the trip came to 5000,
    # send me your shares" and "trip cost 5000, split 5 ways" put nothing on the sheet,
    # because every pattern above wants either a currency mark or a verb next to the
    # figure, and a total has neither.
    #
    # Two ways to qualify, because "was" and "is" are far too weak on their own —
    # "he was 25" and "the queue was 20 people" are not payments:
    #   - an unambiguous cost verb (cost / came to / totalled), or
    #   - a weak verb, but anchored to something that has a bill.
    # The trailing guard keeps "the trip was 3 days" and "dinner was 2 hours" out.
    _BILL_NOUN = (r'(?:trip|bill|dinner|lunch|breakfast|meal|tab|cab|uber|ola|ride|taxi|'
                  r'tickets?|rent|groceries|food|hotel|stay|booking|total)')
    for pat in (
        r'\b(?:cost|costs|came\s+to|comes\s+to|totall?ed|totals?)\s+'
        r'(?:about\s+|around\s+|like\s+|roughly\s+)?(\d[\d,]*(?:\.\d{1,2})?)',
        rf'\b{_BILL_NOUN}\b[^.!?]{{0,24}}?\b(?:was|is|were)\s+'
        r'(?:about\s+|around\s+|like\s+|roughly\s+)?(\d[\d,]*(?:\.\d{1,2})?)',
    ):
        m = re.search(pat + r'\s*(?!\s*(?:' + _COUNT_NOUNS + r')\b)', text, re.IGNORECASE)
        if m:
            return f"${m.group(1).replace(',', '')}"
    return None


def _extract_time(text_lower: str, intents: list = None) -> Optional[dict]:
    intents = intents or []

    # For alarm intent, prefer "alarm/wake" anchored times over generic mentions
    if "alarm" in intents:
        alarm_patterns = [
            r"(?:alarm|wake\s*(?:me\s+)?up|wake\s+up)\s+(?:at\s+|for\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm))",
            r"(?:set|put)\s+(?:an?\s+)?(?:alarm|timer)\s+(?:at\s+|for\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm))",
        ]
        for p in alarm_patterns:
            m = re.search(p, text_lower)
            if m:
                return {"value": m.group(1), "type": "specific_time"}

    components = []
    for pattern, time_type in _TIME_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            val = m.group(1) if m.lastindex else m.group(0)
            components.append({"value": val, "type": time_type})

    if not components:
        return None
    if len(components) == 1:
        return components[0]
    # Combine: e.g. "wednesday at 11am" → "wednesday at 11am"
    day_part = next((c for c in components if c["type"] in ("day_of_week", "relative_day", "relative_week")), None)
    time_part = next((c for c in components if c["type"] == "specific_time"), None)
    if day_part and time_part:
        return {"value": f"{day_part['value']} at {time_part['value']}", "type": "datetime"}
    return components[0]


# Destination fallback for when the dependency parse misses it.
#
# spaCy tags an unknown lowercase word as a VERB, which turns the preceding "to" into an
# infinitive marker (PART/aux) instead of a preposition (ADP/prep). The prep loop below
# only looks at ADP/prep, so the destination vanishes:
#
#   "order a Rapido to Koramangala"  -> to/ADP/prep   Koramangala/PROPN/pobj   -> found
#   "order a rapido to koramangala"  -> to/PART/aux   koramangala/VERB/xcomp   -> lost
#
# Same sentence, and whether the destination reaches Uber depends on whether the user
# hit shift. People type lowercase in chat constantly. A "from" phrase happens to rescue
# it ("from marathahalli to tinfactory" parses fine) which is why this hid for so long.
#
# The verb list is the guard: "to" genuinely does introduce an infinitive a lot of the
# time ("book a cab to go home"), and those must not become destinations.
_INF_VERBS = {
    "go", "get", "pick", "drop", "take", "meet", "see", "come", "head", "reach",
    "catch", "grab", "leave", "visit", "collect", "bring", "carry", "send", "be",
    "make", "do", "help", "check", "find", "call", "wait", "stay", "move", "walk",
    "drive", "ride", "travel", "fly", "arrive", "return", "start", "book", "pay",
}
_TO_RE = re.compile(r"\bto\b", re.IGNORECASE)
_DEST_TAIL_RE = re.compile(
    r"\s+((?:the\s+)?[\w'&-]+(?:\s+[\w'&-]+){0,3}?)"
    r"(?:\s+(?:to|from|at|by|before|after|around|for|please|pls|asap|now|today|"
    r"tonight|tomorrow|thanks|thx|ok|okay)\b|[,.!?;]|$)",
    re.IGNORECASE)

# Filler that trails a place name in chat: "whitefield asap", "koramangala now".
_PLACE_TRAILING = {
    "asap", "now", "please", "pls", "today", "tonight", "tomorrow", "thanks",
    "thx", "ok", "okay", "soon", "quickly", "immediately", "urgently", "fast",
    "later", "too", "also", "instead", "again", "first", "next", "rn",
}


def _tidy_place(phrase: str) -> Optional[str]:
    """Trim a place phrase that ran past the actual place name.

    The dependency parse pulls whole subtrees, so a conjunction or a following verb
    ends up glued on — "talk to the driver and book a cab" yielded "Driver And Book".
    Cut at the first conjunction or action verb, then drop trailing filler.
    """
    tokens = phrase.split()
    cut = len(tokens)
    for i, tok in enumerate(tokens):
        low = tok.lower().strip(".,!?;")
        if low in ("and", "or", "then") or (i > 0 and low in _INF_VERBS):
            cut = i
            break
    tokens = tokens[:cut]
    while tokens and tokens[-1].lower().strip(".,!?;") in _PLACE_TRAILING:
        tokens.pop()
    out = " ".join(tokens).strip()
    return out or None


# Where someone is, stated as a fact rather than as a "from" phrase. "get me a cab to
# hsr, im at indiranagar" is the ordinary way to give a pickup and the dependency parse
# never saw it as one, because there is no "from".
_PICKUP_FALLBACK_RE = re.compile(
    r"\b(?:i'?m|im|we'?re|were|currently|still)\s+(?:at|in|near|outside|by)\s+"
    r"((?:the\s+)?[\w'&-]+(?:\s+[\w'&-]+){0,2}?)"
    r"(?:\s+(?:and|but|so|right|rn|now|please|pls)\b|[,.!?;]|$)",
    re.IGNORECASE)


def _pickup_fallback(text: str) -> Optional[str]:
    """Pickup stated as 'im at X', used only when no 'from' phrase was found."""
    m = _PICKUP_FALLBACK_RE.search(text)
    if not m:
        return None
    cand = _plausible_place(_tidy_place(
        re.sub(r"^(?:the|a|an)\s+", "", m.group(1).strip(), flags=re.I)))
    return _norm_place(cand)


def _destination_fallback(text: str) -> Optional[str]:
    """Regex 'to X' destination, used only when the dependency parse found none."""
    # Scan each "to" separately rather than finditer over the whole pattern: in
    # "a ride to get to work" the rejected verb match would otherwise swallow the
    # second "to" as well, and the real destination is never examined.
    for to_match in _TO_RE.finditer(text):
        tail = _DEST_TAIL_RE.match(text, to_match.end())
        if not tail:
            continue
        phrase = re.sub(r"^(?:the|a|an)\s+", "", tail.group(1).strip(), flags=re.I).strip()
        head = phrase.split()
        if not head or head[0].lower() in _INF_VERBS:
            continue
        # Same trailing-filler trim the dependency path gets, or "to the airport later"
        # comes back as the destination "Airport Later".
        phrase = _tidy_place(phrase)
        if phrase:
            return phrase
    return None


def _extract_ride_slots(text: str, prev_messages: list = None) -> dict:
    """Extract pickup and destination from ride/travel messages using dependency parsing."""
    doc = _nlp(text)
    result = {}
    _NOISE = {"me", "us", "it", "that", "this", "one", "uber", "lyft", "cab", "ride",
              "taxi", "car", "a ride", "a cab", "a taxi", "an uber", "a lyft",
              "book", "get", "grab", "take", "need", "want", "bro", "yo", "hey",
              "ola", "grab", "gojek", "auto",
              # Apostrophe-less contractions. spaCy tags bare "ill" as a PROPN, so
              # "ill book you an uber to your place" came back with pickup="ill".
              # Nobody types the apostrophe in chat, so these are common.
              "ill", "im", "ive", "id", "youre", "hes", "shes", "theyre", "were",
              "lets", "cant", "dont", "wont", "u", "pls", "plz", "thx", "ok", "okay"}
    _TIME_PREPS = {"by", "before", "after", "around", "at"}
    _VAGUE_PLACES = {"your place", "my place", "his place", "her place", "their place",
                     "your house", "my house", "his house", "her house", "their house",
                     "your apartment", "my apartment", "there", "here", "that place",
                     "your spot", "the spot", "the place"}

    from_phrases = []
    to_phrases = []

    for token in doc:
        if token.dep_ != "prep":
            continue
        prep = token.text.lower()
        pobj = next((child for child in token.children if child.dep_ == "pobj"), None)
        if not pobj:
            continue
        phrase = " ".join(t.text for t in pobj.subtree)
        # Clean trailing preposition phrases that got pulled in
        # e.g. "downtown from my place" -> just "downtown" if there's a 'from' inside
        for sub_token in pobj.subtree:
            if sub_token.dep_ == "prep" and sub_token.text.lower() in ("from", "to", "at"):
                cut_idx = sub_token.idx - doc[0].idx
                phrase_tokens = []
                for t in pobj.subtree:
                    if t.i >= sub_token.i:
                        break
                    phrase_tokens.append(t.text)
                if phrase_tokens:
                    phrase = " ".join(phrase_tokens)
                break

        phrase_lower = phrase.lower().strip()
        if phrase_lower in _NOISE or len(phrase_lower) < 2:
            continue

        if prep == "from":
            from_phrases.append(phrase.strip())
        elif prep == "to":
            to_phrases.append(phrase.strip())
        elif prep == "at" and token.head.text.lower() in ("drop", "dropped", "stop"):
            to_phrases.append(phrase.strip())

    # Also check "at X" when it follows drop/stop patterns
    for token in doc:
        if (token.dep_ == "prep" and token.text.lower() == "at"
                and token.head.text.lower() not in ("drop", "stop", "dropped")
                and token.head.dep_ not in ("prep",)):
            pobj = next((child for child in token.children if child.dep_ == "pobj"), None)
            if pobj:
                phrase = " ".join(t.text for t in pobj.subtree).strip()
                # "at" only means destination when it is not saying where the rider
                # already IS. "get me a cab to hsr, im at indiranagar" was putting
                # Indiranagar in destination as well as pickup, sending the cab to
                # the place it was supposed to collect from.
                if (phrase.lower() not in _NOISE and len(phrase) > 2 and not to_phrases
                        and not _PICKUP_FALLBACK_RE.search(text)):
                    to_phrases.append(phrase)

    # Strip articles from start
    def clean_phrase(p):
        p = re.sub(r'^(?:the|a|an)\s+', '', p, flags=re.IGNORECASE).strip()
        # Title case proper nouns, keep "home"/"my place" etc lowercase-friendly
        if p.lower() in ("home", "my place", "my house", "your place"):
            return p.lower()
        p = _tidy_place(p) or ""
        if not p:
            return ""
        return p.title() if p.islower() and len(p) > 3 else p

    if to_phrases:
        _d = clean_phrase(to_phrases[0])
        if _d:
            result["destination"] = _d
    if from_phrases:
        _p = clean_phrase(from_phrases[0])
        if _p:
            result["pickup"] = _p

    # Fallback: "X to Y" pattern without "from" — proper noun before "to" is pickup
    if "pickup" not in result and to_phrases:
        to_token = next((t for t in doc if t.text.lower() == "to" and t.dep_ == "prep"), None)
        if to_token:
            before_to = [t for t in doc if t.i < to_token.i and t.pos_ == "PROPN"
                         and t.text.lower() not in _NOISE and len(t.text) > 2]
            if before_to:
                propns = []
                for t in before_to:
                    if propns and t.i == propns[-1].i + 1:
                        propns.append(t)
                    else:
                        propns = [t]
                pickup_phrase = " ".join(t.text for t in propns).strip()
                if pickup_phrase.lower() not in _NOISE:
                    result["pickup"] = clean_phrase(pickup_phrase)

    # Fallback: lowercase place name the parser turned into a verb (see _INF_VERBS above)
    if "destination" not in result:
        fallback = _destination_fallback(text)
        if fallback and fallback.lower() not in _NOISE:
            result["destination"] = clean_phrase(fallback)

    # Fallback: pickup given as a statement of where they are, not a "from" phrase
    if "pickup" not in result:
        pk = _pickup_fallback(text)
        if pk and pk.lower() not in _NOISE:
            result["pickup"] = pk

    # Fallback: if no destination found but "home" is mentioned with movement verb
    if "destination" not in result:
        if re.search(r'\bhome\b', text.lower()) and re.search(r'\b(?:get|go|take|drop|ride|back|head)\b', text.lower()):
            result["destination"] = "home"

    # If destination is a vague reference, try to resolve from context
    dest = result.get("destination", "").lower().strip()
    if dest in _VAGUE_PLACES and prev_messages:
        resolved = _resolve_place_from_context(prev_messages)
        if resolved:
            result["destination"] = resolved
        else:
            del result["destination"]

    pickup = result.get("pickup", "").lower().strip()
    if pickup in _VAGUE_PLACES and prev_messages:
        resolved = _resolve_place_from_context(prev_messages)
        if resolved:
            result["pickup"] = resolved
        else:
            del result["pickup"]

    return result


def _resolve_place_from_context(prev_messages: list) -> str | None:
    """Scan recent context messages for a real address or place name."""
    # Look in reverse order (most recent first)
    for msg in reversed(prev_messages):
        # Street address pattern: number + street name
        addr = re.search(
            r'\b(\d{1,5}\s+[\w\s]+?(?:street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd|lane|ln|way|place|pl|court|ct|circle|cir|parkway|pkwy)(?:\s+\w+)?)\b',
            msg, re.IGNORECASE
        )
        if addr:
            return addr.group(1).strip().title()

        # Named places: "the airport", "downtown", "mall", specific names with caps
        place = re.search(
            r'\b(?:the\s+)?(airport|station|mall|hospital|university|campus|office|gym|library|hotel|restaurant|club|bar|park|beach|terminal)\b',
            msg, re.IGNORECASE
        )
        if place:
            return place.group(0).strip().title()

        # Capitalized multi-word proper nouns (likely place names)
        proper = re.findall(r'(?<!\.\s)(?:^|\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', msg)
        if proper:
            candidate = proper[-1].strip()
            skip = {"Let Me", "I Am", "I Was", "Do You", "Can You", "What Is",
                    "How Are", "Oh My", "Thank You", "No Way"}
            if candidate not in skip and len(candidate) > 4:
                return candidate

    return None


def _extract_food(text_lower: str) -> Optional[dict]:
    result = {}
    for pattern, slot_type in _FOOD_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            val = m.group(1).strip() if m.lastindex else m.group(0).strip()
            if slot_type == "food_item":
                val = re.sub(r'\s+(?:for|at|by|before|after|around)\s+(?:dinner|lunch|breakfast|tonight|today|tomorrow|later|now|the|us|me|everyone).*$', '', val).strip()
            result[slot_type] = val
    return result if result else None


def _extract_reminder_task(text_lower: str) -> Optional[str]:
    patterns = [
        r"remind\s+(?:me\s+)?to\s+(.+?)(?:\s+at\s+\d|\s+by\s+(?:\d|next|this|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|\s+on\s+\d|\s+before\s+\w+|\s+tomorrow|\s+tonight|,\s*(?:and|also|i|he|she|we|but)\s+|$)",
        r"don'?t\s+(?:let\s+me\s+)?forget\s+(?:to\s+)?(.+?)(?:\s+at\s+\d|\s+by\s+(?:\d|next|this|tomorrow)|\s+tomorrow|,\s*(?:and|also|i)\s+|$)",
        r"ping\s+me\s+(?:to|about)\s+(.+?)(?:\s+at\s+\d|\s+by\s+(?:\d|next|this)|\,\s*(?:and|also)\s+|$)",
        r"(?:gotta|need\s+to)\s+remember\s+(?:to\s+)?(.+?)(?:\s+at\s+\d|\s+by\s+(?:\d|next|this)|,\s*(?:and|also)\s+|$)",
    ]
    for p in patterns:
        m = re.search(p, text_lower)
        if m:
            task = m.group(1).strip().rstrip(".,!?")
            if len(task) > 2:
                return task
    return None


def _extract_money_note(text_lower: str) -> Optional[str]:
    patterns = [
        r"\bfor\s+(?:the\s+)?([\w\s]{3,30}?)(?:\s+and\s+|\s*$|\s*[.!?])",
    ]
    for p in patterns:
        m = re.search(p, text_lower)
        if m:
            note = m.group(1).strip()
            stop_words = {"me", "you", "him", "her", "us", "them", "it", "that",
                          "this", "back", "sure", "real", "now", "later"}
            if note and note not in stop_words and len(note) > 2:
                return note
    return None


def _extract_bill_name(text_lower: str) -> Optional[str]:
    """Extract bill type from bills intent messages using dependency parsing."""
    doc = _nlp(text_lower)
    # Look for "X bill" or "bill for X" patterns via dependency tree
    for token in doc:
        if token.text == "bill" and token.dep_ in ("dobj", "nsubj", "pobj", "attr", "ROOT"):
            # Grab compound modifiers: "electric bill", "phone bill", "internet bill"
            compounds = [child.text for child in token.children if child.dep_ in ("compound", "amod", "nmod")]
            if compounds:
                return " ".join(compounds + ["bill"]).title()
            # Check "bill for X" pattern
            for child in token.children:
                if child.dep_ == "prep" and child.text == "for":
                    pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
                    if pobj:
                        return " ".join(t.text for t in pobj.subtree).title() + " Bill"
    # "pay rent", "pay utilities", "pay insurance"
    m = re.search(r'\b(?:pay|paying)\s+(?:the\s+|my\s+)?(\w+)\b', text_lower)
    if m:
        bill = m.group(1).strip()
        _bill_types = {"rent", "utilities", "electricity", "electric", "water", "gas",
                       "internet", "wifi", "phone", "cable", "insurance", "mortgage",
                       "tuition", "subscription", "netflix", "spotify", "hulu"}
        if bill in _bill_types:
            return bill.title()
    return None


def _extract_calendar_event(text: str, text_lower: str) -> Optional[str]:
    """Extract event/activity name from calendar intent messages using dependency parsing."""
    doc = _nlp(text)
    _TIME_WORDS = {"tomorrow", "today", "tonight", "monday", "tuesday", "wednesday",
                   "thursday", "friday", "saturday", "sunday", "morning", "afternoon",
                   "evening", "night", "am", "pm", "week", "month", "noon"}
    _VERBS = {"schedule", "block", "put", "add", "set", "create", "book", "mark", "save", "calendar"}

    _JUNK = _TIME_WORDS | {"to", "from", "for", "at", "on", "by", "and", "or", "the", "a", "an", "my"}

    # Strategy 1: "for <event>" — highest priority, most explicit
    # "block 2pm for the dentist", "schedule time for team standup"
    m = re.search(r'\bfor\s+(?:the\s+|a\s+|my\s+)?(.+?)(?:\s+(?:at|on|by|from|tomorrow|today)\b|$)', text_lower)
    if m:
        event = m.group(1).strip().rstrip(" ,.")
        if event and len(event) > 1 and not re.match(r'^\d', event) and event.lower() not in _TIME_WORDS:
            return event.title()

    # Strategy 2: Find the direct object of the scheduling verb
    for token in doc:
        if token.lemma_.lower() in _VERBS and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ("dobj", "attr"):
                    phrase_tokens = []
                    for t in child.subtree:
                        if t.text.lower() in _JUNK or re.match(r'\d', t.text):
                            continue
                        if t.dep_ == "prep" and t.text.lower() in ("at", "on", "for", "from", "by"):
                            break
                        phrase_tokens.append(t.text)
                    event = " ".join(phrase_tokens).strip()
                    if event and len(event) > 1 and event.lower() not in _TIME_WORDS:
                        return event

    # Strategy 3: "X on/at <time>" — grab X
    m = re.search(
        r'(?:schedule|block|put|add|book|set\s+up)\s+(?:a\s+|an\s+|the\s+|my\s+)?'
        r'(.+?)\s+(?:on|for|at|from|by|tomorrow|today|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d)',
        text_lower
    )
    if m:
        event = m.group(1).strip().rstrip(" ,.")
        if event and len(event) > 1 and event.lower() not in _TIME_WORDS:
            return event.title()

    return None


# Phone number regex: +country code, optional spaces/dashes, 7-15 digits total
#
# The lookarounds are load-bearing. Without them search() happily pulls a 10-digit run
# out of the MIDDLE of a payment-terminal reference — "1403F9202347206" yielded
# phone=9202347206 and dragged the contact intent up to 0.93 with it. A digit run welded
# to letters, or to yet more digits, is not a phone number. Requiring a non-alphanumeric
# on both sides also kills the long-identifier case: in a 25-digit run every start
# position either has a digit behind it or a digit ahead of it, so nothing matches.
_PHONE_RE = re.compile(
    r'(?<![A-Za-z0-9])(\+?\d[\d\s\-().]{6,18}\d)(?![A-Za-z0-9])'
)

_PHONE_STOP = {"0", "100", "200", "300", "400", "500", "700", "1000"}

# Order / invoice / transaction identifiers are long digit runs that nobody can dial.
# Matched against the text immediately preceding the candidate.
_PHONE_REF_CONTEXT = re.compile(
    r'\b(?:order|invoice|inv|ref|reference|txn|transaction|receipt|ticket|tracking|'
    r'confirmation|acct|account)\b\W{0,4}(?:no\.?|number|id|#)?\W{0,4}$',
    re.IGNORECASE)


def _extract_phone(text: str) -> Optional[str]:
    """Extract phone number from text."""
    # finditer, not search: a rejected candidate must not hide a real number later in
    # the same message ("order 19101, call me on 9876543210").
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r'[^\d+]', '', m.group(1))
        bare = digits.lstrip('+')
        if len(bare) < 7 or bare in _PHONE_STOP:
            continue
        if _PHONE_REF_CONTEXT.search(text[:m.start()]):
            continue
        return digits
    return None


# Something in the message has to name a way of reaching a person. Used to decide
# whether a contact fire has any basis at all — see Phase 1c2b.
_CONTACT_SIGNAL = re.compile(
    r'\b(?:call|calling|called|text|texting|message|messaging|msg|dial|ring|ping|'
    r'phone|cell|mobile|contact|contacts|number|reach|touch|facetime|whatsapp|'
    r'telegram|dm|email|voicemail|hit\s+(?:him|her|them|up))\b', re.IGNORECASE)


def _extract_contact_name(text: str, text_lower: str) -> Optional[str]:
    """Extract name from contact-specific phrasings that _extract_recipient misses."""
    _CONTACT_NAME_STOP = {
        "the", "a", "an", "my", "your", "our", "his", "her", "this", "that",
        "me", "you", "us", "it", "some", "add", "save", "store", "put",
        "number", "contact", "contacts", "phone", "call", "name",
    }

    patterns = [
        # "as Karl Brown", "under Karl Brown", "named Karl Brown", "name Karl Brown", "called Karl"
        r'(?:as|under|named|name|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        # "Karl Brown's number/contact/phone"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'?s?\s+(?:number|contact|phone|cell)",
        # "contact for Karl Brown", "contact Karl Brown"
        r'contacts?\s+(?:for\s+|info\s+(?:for\s+)?)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        # "add Karl Brown", "save Karl Brown" when followed by number/to/phone or end
        r'(?:add|save|store|put)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\s+(?:to|in|number|phone|\+?\d)|$|[,.])',
        # "[number] Karl Brown" — name right after a phone number
        r'(?:\+?\d[\d\s\-]{6,18}\d)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        # "for Karl Brown"
        r'\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\s|$|[,.])',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            name = m.group(1).strip()
            if name.lower() not in _CONTACT_NAME_STOP and len(name) > 1:
                return name
    return None


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 — Cancel / Defer / Re-trigger
# ═══════════════════════════════════════════════════════════════════════

# Intents governed by ConversationStateMachine (api/conversation.py). IntentLifecycle
# must not track or re-trigger these — two registries for the same intent get out of
# sync and re-add stale intents to turns the state machine already resolved.
_SM_MANAGED_INTENTS = {"money", "ride"}


class IntentLifecycle:
    """
    Track active intents per room and detect cancel/defer/re-trigger.

    Scalable design:
    - Patterns are compiled once, matched generically (no per-intent hardcoding)
    - Recently cancelled intents are remembered so they can be re-triggered OR suppressed
    - Defer wins over new-intent when temporal phrases co-occur with active intents
    - Questions about cancelled intents are suppressed automatically
    """

    ACTIVE_TIMEOUT = 300
    CANCEL_MEMORY = 120   # remember cancellations for 2 minutes

    CANCEL_PATTERNS = [re.compile(p) for p in [
        r"\b(?:cancel|nevermind|never\s*mind|forget\s+(?:it|that|about\s+it))\b",
        r"\b(?:actually\s+)?(?:don'?t|do\s*n'?t)\s+(?:bother|worry|do\s+(?:it|that))\b",
        r"\b(?:nah|nvm|jk|just\s+kidding)\b",
        r"\bscrap\s+(?:it|that)\b",
        r"\bi\s+(?:changed?\s+my\s+mind|don'?t\s+(?:need|want)\s+(?:it|that))\b",
    ]]

    DEFER_PATTERNS = [re.compile(p) for p in [
        r"\b(?:not\s+(?:now|yet)|maybe\s+later)\b",
        # "hold off" postpones. "hold on" does not — in chat it means "wait a second
        # while I do this", and it attaches to the exact messages we most want to fire
        # on: "sending it now hold on", "booking an ola rn hold on", "okk transferring
        # it now hold on". Treating the two as synonyms deferred 10 of the 30 missed
        # acceptances on the Claude eval — a third of all misses — on messages the
        # model had already scored at 0.995+.
        r"\b(?:hold\s+off|in\s+a\s+bit)\b",
        r"\b(?:after|once)\s+(?:I|we|i)\s+\w+",
        r"\b(?:next\s+time|some\s+other\s+time)\b",
        r"\blet\s+me\s+think\b",
        r"\bpay\s+(?:me\s+)?later\b",
        r"\b(?:i'?ll|will)\s+(?:\w+\s+)?later\b",
        r"\badd\s+later\b",
        r"\bdo\s+(?:it|that)\s+later\b",
    ]]

    CONFIRM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
        r"^\s*(?:actually\s+)?(?:yes|yeah|yep|yup|sure|ok(?:ay)?|go\s+ahead|do\s+it|let'?s\s+(?:go|do\s+it)|confirmed?|absolutely|send\s+it|book\s+it|just\s+do\s+it)(?:\s+(?:do\s+it|go\s+ahead|please))?\s*[.!]?\s*$",
    ]]

    # Reviving something the user CANCELLED needs more than a bare agreement. A plain
    # "sure" or "ok" later in the conversation is almost always about something else,
    # and resurrecting a cancelled alarm on it is the same failure as re-firing an
    # active one. Changing your mind is deliberate, so require language that says so:
    # "actually yes", "do it anyway", "changed my mind".
    REVIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
        r"\bactually\b",
        r"\b(?:do|book|send|set)\s+it\s+anyway\b",
        r"\bchanged\s+my\s+mind\b",
        r"\bnvm\s+(?:that|it)?\s*,?\s*(?:do|go)\b",
        r"^\s*(?:go\s+ahead|do\s+it|just\s+do\s+it)\s*[.!]?\s*$",
    ]]

    def __init__(self):
        self.active: dict[str, dict] = defaultdict(dict)
        self.cancelled: dict[str, dict] = defaultdict(dict)

    def _clean_expired(self, room_id: str):
        now = time.time()
        expired = [k for k, v in self.active.get(room_id, {}).items()
                   if now - v["ts"] > self.ACTIVE_TIMEOUT]
        for k in expired:
            del self.active[room_id][k]
        expired_c = [k for k, v in self.cancelled.get(room_id, {}).items()
                     if now - v["ts"] > self.CANCEL_MEMORY]
        for k in expired_c:
            del self.cancelled[room_id][k]

    def _matches(self, patterns, text):
        return any(p.search(text) for p in patterns)

    def process(self, room_id: str, text: str, detection: dict) -> dict:
        self._clean_expired(room_id)
        tl = text.lower().strip()
        lifecycle = {}
        active_intents = list(self.active.get(room_id, {}).keys())
        recently_cancelled = list(self.cancelled.get(room_id, {}).keys())

        # 1. Cancel — clear active intents, remember them
        if active_intents and self._matches(self.CANCEL_PATTERNS, tl):
            lifecycle["action"] = "cancel"
            lifecycle["cancelled_intents"] = active_intents
            for intent in active_intents:
                self.cancelled[room_id][intent] = {
                    "ts": time.time(),
                    "original": self.active[room_id][intent].get("text", ""),
                }
                self.active[room_id].pop(intent, None)
            detection = dict(detection)
            detection["intents"] = [i for i in detection.get("intents", []) if i not in active_intents]
            detection["lifecycle"] = lifecycle
            return detection

        # 2. Defer — temporal phrases override even if the model fires the same intent
        if self._matches(self.DEFER_PATTERNS, tl):
            deferred = list(active_intents) if active_intents else []
            for intent in deferred:
                self.active[room_id][intent]["status"] = "deferred"
            # Also defer newly-fired intents (e.g. "add later" fires calendar fresh)
            fired = detection.get("intents", [])
            for intent in fired:
                if intent not in deferred:
                    deferred.append(intent)
                    self.active[room_id][intent] = {"status": "deferred", "ts": time.time(), "text": text[:80]}
            if deferred:
                lifecycle["action"] = "defer"
                lifecycle["deferred_intents"] = deferred
                detection = dict(detection)
                detection["intents"] = []
                detection["lifecycle"] = lifecycle
                return detection

        # 3. Confirm / re-trigger — ONLY for intents that were deferred or cancelled.
        #
        # "active" used to be included here, which meant an intent that had ALREADY
        # fired would fire again on any bare "sure"/"ok"/"yeah" later in the room, on a
        # completely unrelated subject:
        #
        #   "Set alarm for tomorrow 10 am"  -> alarm fires   (correct)
        #   "Sure"                          -> alarm fires AGAIN
        #
        # On that second message the model scores alarm at ~0.03; the intent came
        # entirely from here. Reported from production with `reminder`, reproduced with
        # `alarm`. Confirmation is meant to revive something that was PUT OFF —
        # "remind me later" then "actually yeah do it" — not to repeat something already
        # done. Applies to all nine intents and runs on both decision paths.
        if self._matches(self.CONFIRM_PATTERNS, tl):
            confirmed = []
            for intent, info in self.active.get(room_id, {}).items():
                if info.get("status") == "deferred":
                    confirmed.append(intent)
                    info["status"] = "confirmed"
            # Cancelled intents need an explicit change of mind, not a bare "sure".
            if not confirmed and recently_cancelled and self._matches(
                    self.REVIVE_PATTERNS, tl):
                for intent in recently_cancelled:
                    confirmed.append(intent)
                    self.active[room_id][intent] = {
                        "status": "confirmed",
                        "ts": time.time(),
                        "text": self.cancelled[room_id][intent].get("original", ""),
                    }
                    self.cancelled[room_id].pop(intent, None)
            if confirmed:
                lifecycle["action"] = "confirm"
                lifecycle["confirmed_intents"] = confirmed
                detection = dict(detection)
                if not detection.get("intents"):
                    detection["intents"] = confirmed
                # A revived intent still needs a target, or the client has no show_to
                # and shows the prompt to everyone in the room — which is what was
                # reported: both sender and receiver got the same popup. The confirming
                # message ("sure") carries no direction of its own, so target it at the
                # person who just confirmed, since they are the one acting.
                if not detection.get("target"):
                    detection["target"] = {"show_to": "sender",
                                           "reason": "confirmed_deferred_intent"}
                detection["lifecycle"] = lifecycle
                return detection

        # 4. Suppress — questions about recently cancelled intents
        fired = detection.get("intents", [])
        if recently_cancelled and fired:
            suppressed = [i for i in fired if i in recently_cancelled]
            if suppressed:
                detection = dict(detection)
                detection["intents"] = [i for i in fired if i not in suppressed]
                if not detection["intents"]:
                    lifecycle["action"] = "suppressed"
                    lifecycle["reason"] = "recently_cancelled"
                    lifecycle["suppressed_intents"] = suppressed
                    detection["lifecycle"] = lifecycle
                return detection

        # 5. Suppress questions that look like conversational ("He still hasn't paid?")
        if tl.endswith("?") and fired:
            question_suppress = [
                r"\b(?:hasn'?t|haven'?t|didn'?t|doesn'?t|won'?t|isn'?t)\b",
            ]
            has_action_verb = re.search(
                r'\b(?:send|pay|venmo|cashapp|zelle|transfer|split|book|order|call|get\s+me)\b',
                tl, re.IGNORECASE
            )
            if not has_action_verb and any(re.search(p, tl) for p in question_suppress):
                detection = dict(detection)
                detection["intents"] = []
                return detection

        # 6. Register new active intents
        # money and ride are owned by ConversationStateMachine, which already tracks
        # their pending/fired state. Registering them here too created a second,
        # disagreeing registry: after the state machine fired and consumed a ride
        # pending, this one still held ride as "active", so the next bare "Sure"
        # matched CONFIRM_PATTERNS and re-added ride to a turn the state machine had
        # already decided was no_fire. That is the duplicated-icon bug seen in the app.
        for intent in fired:
            if intent in _SM_MANAGED_INTENTS:
                continue
            self.active[room_id][intent] = {
                "status": "active",
                "ts": time.time(),
                "text": text[:80],
            }

        detection = dict(detection)
        if lifecycle:
            detection["lifecycle"] = lifecycle
        return detection

    def get_active(self, room_id: str) -> dict:
        self._clean_expired(room_id)
        return dict(self.active.get(room_id, {}))


intent_lifecycle = IntentLifecycle()


# ═══════════════════════════════════════════════════════════════════════
# Full Pipeline — ties Phase 1-4 together
# ═══════════════════════════════════════════════════════════════════════

_FOOD_SUPPRESS_PHRASES = [
    'usual order', 'the usual', 'pay the usual',
    'order of operations', 'in order to', 'out of order', 'law and order',
]

_MONEY_MEME_PHRASES = [
    'cash me outside', 'cash me ousside',
]

_SARCASM_MARKERS = re.compile(
    r'\b(?:lol|lmao|rofl|haha|hehe|jk|just\s+kidding)\b', re.IGNORECASE
)

_SARCASM_ABSURD_PATTERNS = [
    re.compile(r'\bto the (?:moon|sun|mars|heaven|void|shadow realm|another (?:dimension|planet|universe))\b', re.IGNORECASE),
    re.compile(r'\baway from (?:this|here|everything|life|reality|this (?:disaster|mess|chaos))\b', re.IGNORECASE),
    re.compile(r'\bwith my (?:bed|pillow|cat|dog|couch|sofa|wall|imaginary)\b', re.IGNORECASE),
    re.compile(r'\bwhen I (?:actually )?(?:start|begin) (?:caring|trying|giving)\b', re.IGNORECASE),
    re.compile(r'\bfor when (?:I )?(?:actually )?(?:start|begin) (?:caring|trying|giving)\b', re.IGNORECASE),
    re.compile(r'\boh wait\b', re.IGNORECASE),
    re.compile(r'\bsaid no one\b', re.IGNORECASE),
    re.compile(r'\bfind (?:better|new) (?:friends|life|brain|personality)\b', re.IGNORECASE),
    re.compile(r'\b(?:out of|from) (?:this|another) (?:conversation|meeting|class|lecture|life|reality|world|dimension|planet|universe)\b', re.IGNORECASE),
    re.compile(r'\bmy (?:emotional|existential|spiritual|mental)\b', re.IGNORECASE),
    re.compile(r'\b(?:the universe|god|nobody|my sanity|my soul|my will to live)\b', re.IGNORECASE),
    re.compile(r'\bto care\b.*\bdon\'?t\b', re.IGNORECASE),
    re.compile(r'\bdon\'?t\b.*\bto care\b', re.IGNORECASE),
    re.compile(r'\b(?:imaginary|invisible|nonexistent)\b', re.IGNORECASE),
]

_SARCASM_EXAGGERATED_AMOUNTS = re.compile(
    r'\b(?:a\s+)?(?:million|billion|trillion|gazillion|bajillion|infinity|infinite)\s*(?:dollars?|bucks?)?\b',
    re.IGNORECASE
)

_META_STATEMENT_PATTERNS = [
    re.compile(r"\bi\s+will\s+(?:text|tell|message|call|ask)\s+\w+\s+(?:that|to\s+tell)\b", re.IGNORECASE),
    re.compile(r"\bi'?ll\s+(?:text|tell|message|call|ask)\s+\w+\s+(?:that|to\s+tell)\b", re.IGNORECASE),
    re.compile(r"\bgoing\s+to\s+(?:text|tell|message|call)\s+\w+\s+(?:that|to)\b", re.IGNORECASE),
]

# Payment verb inventory. MUST stay in sync with conversation.py's _PAY_VERB family —
# this gate runs first, so any app missing here is stripped before the state machine
# ever sees it (that is how "paypal me 20" silently produced nothing).
_PV      = (r"(?:send|pay|transfer|venmo|cashapp|cash\s?app|zelle|paypal|gpay|google\s?pay|"
            r"paytm|phonepe|upi|grabpay|duitnow|give)")
_PV_ING  = (r"(?:sending|paying|transferring|venmo-?ing|cashapp-?ing|cash\s?app-?ing|"
            r"zell(?:e)?ing|paypal(?:l)?-?ing|gpay-?ing|google\s?pay-?ing|paytm-?ing|"
            r"phonepe-?ing|upi-?ing|grabpay-?ing)")
_PV_PAST = (r"(?:sent|paid|transferred|venmo(?:ed|'d|d)?|cashapp(?:ed)?|zell(?:ed|d)?|"
            r"paypal(?:ed|led)?|gpay(?:ed)?|paytm(?:ed)?|phonepe(?:d)?)")

_MONEY_HAS_DIRECTIVE = re.compile(
    rf'\b{_PV}\s+(?:me|us|him|her|them)\b'            # "venmo me", "paypal me"
    rf'|\bpay\s+(?:back|up|for)\b'                     # "pay me back", "pay up"
    rf'|\bsend\s+(?:it|that|the\s+money)\b'
    rf'|\bsend\s+\$?\d'
    rf"|\bi(?:'?ll|ll|will|'?m|m)\s+(?:gonna\s+|going\s+to\s+)?{_PV}\b"
    rf"|\b(?:let\s+me|lemme)\s+{_PV}\b"
    rf"|\bi(?:'?m|m)\s+{_PV_ING}\b"                    # "im zelling you"
    rf"|\b{_PV_ING}\s+(?:you|u|him|her|them|it|that)\b" # "sending you", "cashapping you"
    rf"|\b{_PV_PAST}\s+(?:you|u|him|her|them)\b"        # "just venmoed you"
    rf"|\bgonna\s+{_PV}\b"
    , re.IGNORECASE
)


def full_pipeline(text: str, room_id: str = None, context: list = None,
                   sender: str = None, message_id: str = None,
                   reply_to: str = None, participants: int = None) -> dict:
    """Run the complete detection pipeline: model → keywords → suppression → slots → state machine → lifecycle."""
    # Phase 1a: Model inference (standalone, no context prefix — context is for state machine)
    if context is not None:
        prev_messages = context[-3:] if context else []
    else:
        prev_messages = conversation_ctx.get_prev_messages(room_id) if room_id else None
    result = run_inference(text, prev_messages=prev_messages)
    result["_text"] = text
    if prev_messages:
        result["context_messages"] = len(prev_messages)

    # Phase 1b: Merge keyword detection — only when model sees partial signal
    # Lower gate for intents where keywords are highly unambiguous
    _KEYWORD_GATE = 0.25
    _LOW_GATE_INTENTS = {"calendar", "reminder", "alarm", "bills"}
    _NO_GATE_INTENTS = set()
    kw = fast_keyword_detect(text)
    for intent in kw.get("intents", []):
        if intent in result["intents"]:
            continue
        gate = 0.0 if intent in _NO_GATE_INTENTS else (0.10 if intent in _LOW_GATE_INTENTS else _KEYWORD_GATE)
        if result["scores"].get(intent, 0) >= gate:
            result["intents"].append(intent)
            result["scores"][intent] = max(result["scores"].get(intent, 0), kw["scores"].get(intent, 0.85))

    # Phase 1c: Post-model suppression
    tl = text.lower()
    _pre_suppress = set(result["intents"])

    if "food_order" in result["intents"] and any(p in tl for p in _FOOD_SUPPRESS_PHRASES):
        result["intents"] = [i for i in result["intents"] if i != "food_order"]

    # Phase 1c2: Ride discussion suppression — talking about ride services, not requesting
    _RIDE_DISCUSSION = ['uber surge', 'uber pricing', 'lyft pricing', 'cab fare',
                        'uber schedule', 'can uber', 'does uber', 'does lyft']
    if "ride" in result["intents"] and any(p in tl for p in _RIDE_DISCUSSION):
        result["intents"] = [i for i in result["intents"] if i != "ride"]

    # Phase 1c2b: Contact needs an actual contact signal. A bare digit run is not one.
    # "any chance we can see the status of 1403F9202347206 on ipos portal" scored
    # contact at 0.93 — entirely because a transaction reference reads like a phone
    # number. If nothing names a way of reaching someone AND no plausible number
    # survives extraction, there is nobody to contact.
    if ("contact" in result["intents"]
            and not _CONTACT_SIGNAL.search(text)
            and _extract_phone(text) is None):
        result["intents"] = [i for i in result["intents"] if i != "contact"]


    # Phase 1c3: Gratitude suppression — "thanks for covering/paying" = not a money request
    _GRATITUDE_PATTERNS = [
        re.compile(r'\bthanks?\b.*\b(?:cover|covering|pay|paying|paid|picking|getting|buying|treating)\b', re.IGNORECASE),
        re.compile(r'\b(?:cover|covering|pay|paying|paid|picking|getting|buying|treating)\b.*\bthanks?\b', re.IGNORECASE),
        re.compile(r'\bappreciate\b.*\b(?:cover|pay|paid|getting)\b', re.IGNORECASE),
        re.compile(r'\bmy\s+treat\b', re.IGNORECASE),
        re.compile(r'\bi\s+got\s+you\s+covered\b', re.IGNORECASE),
        re.compile(r'\bgot\s+you\s+covered\b', re.IGNORECASE),
        re.compile(r'\bthanks?\s+for\b.*\b(?:dinner|lunch|food|drinks?|bill|tab|ride|uber|cab)\b', re.IGNORECASE),
    ]
    _SELF_INIT_KW = re.compile(r"\bi(?:'?ll|will|'?m)\s+(?:venmo|send|pay|cashapp|zelle|transfer)", re.IGNORECASE)
    if "money" in result["intents"] and any(p.search(tl) for p in _GRATITUDE_PATTERNS):
        if not _SELF_INIT_KW.search(tl):
            result["intents"] = [i for i in result["intents"] if i != "money"]
            result["money"] = None

    # Phase 1c3b: Venue offer suppression — offering to pay at a venue, not P2P transfer
    _VENUE_OFFER_PATTERNS = [
        re.compile(r'\blet\s+me\s+(?:pay|get|cover|handle)\s+(?:for\s+)?(?:dinner|lunch|food|drinks?|the\s+bill|the\s+tab|the\s+check|this|that|it)\b', re.IGNORECASE),
        re.compile(r'\bi(?:m|\'m)\s+(?:paying|covering|getting)\s+(?:for\s+)?(?:dinner|lunch|food|drinks?|the\s+bill|the\s+tab|the\s+check|this|that|it)\b', re.IGNORECASE),
        re.compile(r'\bi(?:ll|\'ll|will)\s+(?:pay|get|cover|handle)\s+(?:for\s+)?(?:dinner|lunch|food|drinks?|the\s+bill|the\s+tab|the\s+check|this|that|it)\b', re.IGNORECASE),
        re.compile(r'\bi\s+got\s+(?:this|it|the\s+bill|the\s+tab|the\s+check)\b', re.IGNORECASE),
        re.compile(r'\b(?:this\s+one(?:\'?s)?\s+on\s+me|on\s+me\s+(?:tonight|today|this\s+time))\b', re.IGNORECASE),
        re.compile(r'\bi(?:m|\'m)\s+paying\s+for\s+(?:this|that)\s+one\b', re.IGNORECASE),
    ]
    if "money" in result["intents"] and any(p.search(tl) for p in _VENUE_OFFER_PATTERNS):
        result["intents"] = [i for i in result["intents"] if i != "money"]
        result["money"] = None

    # Phase 1c3c: Future promise suppression — "ill pay you back tomorrow" = not immediate P2P
    _FUTURE_PROMISE_PATTERNS = [
        re.compile(r'\b(?:i(?:ll|\'ll|will)|gonna|going\s+to)\s+(?:pay|send|venmo|cashapp|zelle|transfer)\b.*\b(?:tomorrow|tmrw|tmr|next\s+week|next\s+month|later|soon|tonight|this\s+(?:week|weekend|friday))\b', re.IGNORECASE),
        re.compile(r'\b(?:tomorrow|tmrw|tmr|next\s+week|next\s+month|later|tonight)\b.*\b(?:i(?:ll|\'ll|will)|gonna)\s+(?:pay|send|venmo|cashapp|zelle|transfer)\b', re.IGNORECASE),
        re.compile(r'\b(?:i(?:ll|\'ll|will)|gonna)\s+(?:pay|send)\s+(?:you|u|it|that)\s+back\s+(?:tomorrow|tmrw|later|next\s+week|soon|tonight)\b', re.IGNORECASE),
    ]
    if "money" in result["intents"] and any(p.search(tl) for p in _FUTURE_PROMISE_PATTERNS):
        result["intents"] = [i for i in result["intents"] if i != "money"]
        result["money"] = None

    # Phase 1c4: Exaggerated amount sarcasm (no marker needed) — "pay me a million/billion"
    if "money" in result["intents"] and _SARCASM_EXAGGERATED_AMOUNTS.search(tl):
        result["intents"] = [i for i in result["intents"] if i != "money"]
        result["money"] = None

    # Phase 1c4b: Amount cap — amounts above $5000 in casual chat aren't real payment requests
    _MONEY_CAP = 5000
    if "money" in result["intents"]:
        amounts = re.findall(r'[\$₹]\s*([\d,]+(?:\.\d+)?)', tl)
        amounts += re.findall(r'([\d,]+(?:\.\d+)?)\s*(?:dollars?|bucks?|rupees?|rs\.?|inr|ringgit|rm|myr)\b', tl)
        amounts += re.findall(r'\b(?:send|pay|transfer|venmo|cashapp|zelle)\s+(?:me\s+)?(?:\w+\s+)?([\d,]+)\b', tl)
        parsed = []
        for a in amounts:
            try:
                parsed.append(float(a.replace(',', '')))
            except:
                pass
        if parsed and min(parsed) > _MONEY_CAP:
            result["intents"] = [i for i in result["intents"] if i != "money"]
            result["money"] = None

    # Phase 1c4c: Price complaint suppression — reacting to a price is not a payment request
    _PRICE_COMPLAINT = re.compile(
        r'(?:damn|dude|bro|wow|yikes|oof|sheesh)?\s*\$?\d+.*(?:steep|expensive|a\s+lot|too\s+much|pricey|insane|crazy|wild|ridiculous)'
        r'|(?:steep|expensive|a\s+lot|too\s+much|pricey|insane|crazy|wild|ridiculous).*\$?\d+',
        re.IGNORECASE,
    )
    if "money" in result["intents"] and _PRICE_COMPLAINT.search(tl):
        if not any(w in tl for w in ['send', 'pay', 'venmo', 'cashapp', 'zelle', 'owe']):
            result["intents"] = [i for i in result["intents"] if i != "money"]
            result["money"] = None

    # Phase 1c5: Bill discussion suppression — "bill is too high/insane" = discussing, not paying
    _BILL_DISCUSS_PATTERNS = [
        re.compile(r'\bbill\s+is\s+(?:too\s+)?(?:high|insane|crazy|ridiculous|absurd|wild|nuts|outrageous)\b', re.IGNORECASE),
        re.compile(r'\bbill\s+(?:went|has\s+gone|keeps?\s+going)\s+up\b', re.IGNORECASE),
        re.compile(r'\bbill\s+(?:came|was|seems?|looks?)\b.*\b(?:high|off|wrong|weird|insane|crazy)\b', re.IGNORECASE),
    ]
    if "bills" in result["intents"] and any(p.search(tl) for p in _BILL_DISCUSS_PATTERNS):
        result["intents"] = [i for i in result["intents"] if i != "bills"]

    # Phase 1c6: Habitual statement suppression — "I always forget X" = not a reminder request
    _HABITUAL_PATTERNS = [
        re.compile(r'\bi\s+always\s+forget\b', re.IGNORECASE),
        re.compile(r'\bi\s+keep\s+forgetting\b', re.IGNORECASE),
        re.compile(r'\bi\s+never\s+remember\b', re.IGNORECASE),
        re.compile(r'\bi\s+always\s+(?:lose|miss|skip)\b', re.IGNORECASE),
    ]
    if "reminder" in result["intents"] and any(p.search(tl) for p in _HABITUAL_PATTERNS):
        if not re.search(r'\bremind\s+me\b', tl, re.IGNORECASE):
            result["intents"] = [i for i in result["intents"] if i != "reminder"]

    # Phase 1c7: Self-directed ride suppression — only suppress truly non-actionable self-talk
    _SELF_RIDE_PATTERNS = [
        re.compile(r'\blet\s+me\s+get\s+to\s+it\b', re.IGNORECASE),
        re.compile(r'\bi\'?ll\s+(?:figure|sort)\s+(?:it\s+out|myself)\b', re.IGNORECASE),
    ]
    if "ride" in result["intents"] and any(p.search(tl) for p in _SELF_RIDE_PATTERNS):
        result["intents"] = [i for i in result["intents"] if i != "ride"]

    # Phase 1c7b: Vague ride need — no action verb directed at another party
    _RIDE_VAGUE_NEED = [
        re.compile(r'\bneed\s+to\s+get\s+to\b', re.IGNORECASE),
        re.compile(r'\bno\s+way\s+(?:i\'?m\s+)?getting\b', re.IGNORECASE),
        re.compile(r'\bi\'?m\s+stuck\b', re.IGNORECASE),
        re.compile(r'\bstuck\s+here\b', re.IGNORECASE),
        re.compile(r'\bno\s+one\'?s?\s+(?:available|around|here)\s+to\s+(?:drive|pick)\b', re.IGNORECASE),
        re.compile(r'\bi\s+need\s+transportation\b', re.IGNORECASE),
    ]
    if "ride" in result["intents"] and any(p.search(tl) for p in _RIDE_VAGUE_NEED):
        has_action = re.search(r'\b(?:book|get\s+me|call|order|arrange|pick\s+me\s+up)\b', tl, re.IGNORECASE)
        if not has_action:
            result["intents"] = [i for i in result["intents"] if i != "ride"]

    # Phase 1c7c: Vague money need — situation description, not a request
    _MONEY_VAGUE = [
        re.compile(r'\bi\s+(?:just\s+)?need\s+\d+.*\b(?:to\s+survive|for\s+the|to\s+get\s+(?:by|through))\b', re.IGNORECASE),
        re.compile(r'\bcan\s+you\s+help\s+me\s+out\b', re.IGNORECASE),
    ]
    if "money" in result["intents"] and any(p.search(tl) for p in _MONEY_VAGUE):
        result["intents"] = [i for i in result["intents"] if i != "money"]
        result["money"] = None

    # Phase 1c7e: Sarcasm with body parts / absurd sacrifice (no marker needed)
    _SARCASM_BODY_PARTS = re.compile(r'\b(?:sell|donate|give)\s+(?:a\s+)?(?:kidney|liver|organ|blood|soul|firstborn)\b', re.IGNORECASE)
    if "money" in result["intents"] and _SARCASM_BODY_PARTS.search(tl):
        result["intents"] = [i for i in result["intents"] if i != "money"]
        result["money"] = None

    # Phase 1d: General sarcasm suppression — sarcasm marker + absurd context = not real
    if result["intents"] and _SARCASM_MARKERS.search(tl):
        if any(p.search(tl) for p in _SARCASM_ABSURD_PATTERNS):
            result["intents"] = []
            result["money"] = None

    # Phase 1d2: Sarcastic $0 suppression — "$0" with sarcasm markers is never real
    if "money" in result["intents"]:
        has_zero = bool(re.search(r'\$0(?:\.00?)?\b', tl))
        if has_zero and _SARCASM_MARKERS.search(tl):
            result["intents"] = [i for i in result["intents"] if i != "money"]
            result["money"] = None

    # Phase 1d3: Exaggerated amount + sarcasm marker = never real
    if "money" in result["intents"] and _SARCASM_MARKERS.search(tl):
        if _SARCASM_EXAGGERATED_AMOUNTS.search(tl):
            result["intents"] = [i for i in result["intents"] if i != "money"]
            result["money"] = None

    # Phase 1e: Meme phrase suppression
    if "money" in result["intents"] and any(p in tl for p in _MONEY_MEME_PHRASES):
        result["intents"] = [i for i in result["intents"] if i != "money"]
        result["money"] = None

    # Phase 1f: Meta-statement suppression — "I will text X that I'll send $20"
    # The real intent is contact, not the nested action
    if "money" in result["intents"] and any(p.search(text) for p in _META_STATEMENT_PATTERNS):
        result["intents"] = [i for i in result["intents"] if i != "money"]
        result["money"] = None

    # NOTE: a "directive verb gate" was trialled here (suppress money unless an action verb
    # was found). It fixed 5 false positives but blocked 43/44 genuine requests — phrasings
    # like "spot me the 15", "wire me the 200", "yo shoot me the 20" all scored >0.75 and
    # were overruled by the regex. Removed. Chat phrasing is unbounded; the separation has
    # to come from training data, not a verb list.

    # Phase 2: Intent targeting — who should see the popup
    if result["intents"]:
        result["target"] = detect_target(text, result["intents"], result.get("money"))

    # Phase 3: Slot extraction
    slots = extract_slots(text, result["intents"], room_id=room_id, prev_messages=prev_messages)
    if slots:
        result["slots"] = slots
        if result.get("money") and not result["money"].get("detected_amount") and slots.get("amount"):
            result["money"]["detected_amount"] = slots["amount"]

    # Phase 4: Conversation state machine (fire on response, not request)
    # For money+ride: requests stored as pending, intents fire when responder confirms.
    # Other intents pass through unchanged. Falls back to immediate mode when no sender.
    if conv_classifier is not None and sender and room_id:
        # Experimental path: the conversation model decides money/ride outright. No
        # pending store, no cooldowns, no response classification — the window is the
        # state. Every other intent passes through exactly as the model produced it,
        # so this branch changes nothing outside MANAGED_INTENTS.
        # Window size comes from the loaded model, not a constant — v1 trained at 6,
        # v2 at 10, and serving with the wrong one is a silent train/serve mismatch.
        w = conv_classifier.window
        hist = conversation_ctx.get_window(room_id, size=w)
        fired, conv_scores = conv_classifier.predict(
            build_window(hist, {"sender": sender, "text": text}, size=w))
        # Drop an echo of a fire that already happened.
        #
        # The classifier re-reads the window every message and has no memory, so a
        # conversation that commits twice fires twice:
        #   "where u going?" / "grand central" / "ok, got it"      <- fires
        #   "yup thanks" / "no prob, doing it now"                 <- fires AGAIN
        # That is two prompts for one cab, and the second one carries no metadata
        # because the request record was consumed by the first.
        #
        # A repeat only counts as an echo when there is no open request left to answer
        # and the message adds nothing new. A genuine second action names its own
        # target — "ill book you an uber to koramangala too" carries a destination and
        # scores 0.86 alone, so it survives. A restatement of the action already
        # prompted for — "yeah sure" then "sending it now" — carries neither, and would
        # otherwise put a second payment sheet under the same conversation.
        # Keyed on the PAYER, not the message sender. An offer and its acceptance come
        # from different senders but name the same payer ("shall I send you 100" / "Sure"),
        # so a sender-keyed check treated them as two unrelated payments and prompted
        # twice. _resolve_payer falls back to the sender for every ordinary request, so
        # the request -> ack path is unchanged.
        _own = result.get("slots") or {}
        _own_key = {"money": "amount", "ride": "destination"}
        _payer = _resolve_payer(room_id, sender)
        # Kept so the request-recording branch below can tell "the classifier said
        # nothing" apart from "the classifier fired and we suppressed it as an echo".
        _classifier_fired = set(fired)
        deduped = []
        for intent in fired:
            _k = _own_key.get(intent)
            _mine = _own.get(_k)
            # "Names a target" is not enough to count as a new action — it has to name a
            # DIFFERENT one. Narrating the same payment twice ("Sure" -> "Sending 100
            # now.") repeats the amount that was already prompted for, and letting that
            # through put a second sheet under the same 100 rupees.
            _prev = (request_meta.last_resolved.get(room_id, {}).get(intent) or {}).get(_k)
            adds_new = (_mine not in (None, "", [])
                        and _norm_slot(_mine) != _norm_slot(_prev))
            scores_alone = (result["scores"].get(intent, 0)
                            >= model_state["thresholds"].get(intent, 0.5))
            # recently_fired keys on the PAYER (who parts with the money); has_open keys
            # on the SENDER (is there a request left for *me* to answer). They are
            # different questions and only coincide when the accepter is the payer.
            if (request_meta.recently_fired(room_id, intent, _payer)
                    and not request_meta.has_open(room_id, intent, sender)
                    and not (scores_alone and adds_new)):
                continue
            deduped.append(intent)
        fired = deduped

        result["intents"] = [i for i in result["intents"]
                             if i not in MANAGED_INTENTS] + list(fired)
        # "ok booking to indiranagar" scores travel 0.83 on its own — the intent model
        # reads a booking verb plus a destination as trip planning. In a ride
        # conversation that is the same event, so firing both puts two prompts under
        # one message. Ride wins: the classifier read the whole window to decide it,
        # the travel score came from this message alone.
        if "ride" in fired and "travel" in result["intents"]:
            result["intents"] = [i for i in result["intents"] if i != "travel"]
        # Status MUST use the same vocabulary as the rule path. The backend gates the
        # payment prompt on status == "fired" (see the integration note sent to the
        # team), so emitting anything else here means the prompt never appears no
        # matter how well the model scores — it would look like the model detecting
        # nothing.
        #
        # "pending" has no equivalent here: the classifier keeps no pending store, it
        # re-reads the window each message. A request simply produces no_fire until
        # someone commits, which is what the backend already does nothing about. The
        # states the rule layer owns — cancelled, reminder — are likewise not emitted,
        # because this path does not model them.
        result["conversation_state"] = {
            "status": "fired" if fired else "no_fire",
            "fired_intents": list(fired),
            "decided_by": "conv_classifier",
            "scores": conv_scores,
            "history": len(hist),
        }

        # Attach triggered_by so the mobile apps can read the amount off the original
        # request. On a fire, consume the matching recorded request; otherwise, if this
        # message itself looks like a request, record it for a later fire.
        if fired:
            for intent in fired:
                # Capture the PREVIOUS fire before overwriting it — that timestamp is
                # the boundary for slot recovery (see _since). Read from settled_at,
                # not last_fire: last_fire is cleared whenever a new request arrives.
                prev_fire_ts = request_meta.settled_at.get(room_id, {}).get(intent, 0)
                continues = request_meta.continues_last(room_id, intent)
                # Recorded under the payer, matching the key the dedup check above reads.
                # Recording under the sender would never match on an accepted offer.
                request_meta.mark_fired(room_id, intent, _payer)
                src = request_meta.take(room_id, intent, sender)
                if src:
                    result["conversation_state"]["triggered_by"] = {
                        "sender": src["sender"],
                        "text": src["text"],
                        "message_id": src["message_id"],
                        "slots": src["slots"],
                    }
                    # The acceptance rarely carries the amount or destination; the
                    # request does. Merge per KEY, not per dict — extract_slots returns
                    # a dict with null values rather than omitting them, so a truthiness
                    # check on the whole dict keeps an all-null one and silently drops
                    # the real values. That is why ride came back with destination null
                    # at the top level while triggered_by had "Airport".
                    # This runs whether or not the recorded request had slots of its
                    # own. It used to be gated on `if src["slots"]:`, but the most
                    # recent request-shaped message is often the one WITHOUT the
                    # figure — "can you lend me 500" ... "just send it to my upi" ...
                    # "sending now" hands back the upi message, whose slots are empty,
                    # so the merge was skipped and the payment sheet opened blank.
                    merged = dict(result.get("slots") or {})
                    for k, v in (src["slots"] or {}).items():
                        if merged.get(k) in (None, "", []):
                            merged[k] = v

                    # A counter-offer changes the amount. "can u lend me 2000" ->
                    # "i can only do 1000" -> "cool sending now" must prompt for
                    # 1000, not the 2000 originally asked for. The request record
                    # only knows the first figure, so prefer the most recent amount
                    # anyone actually said. Pre-filling the wrong number is worse
                    # than pre-filling none — on a payment screen it invites
                    # sending double.
                    # Scope every window lookup to messages after the previous fire for
                    # this intent. Unscoped, a settled negotiation leaks into the next
                    # one — see _since().
                    scoped = _since(hist, prev_fire_ts)
                    blanked = set()      # slots deliberately cleared, see below

                    if intent == "money":
                        latest = None
                        for m in reversed(scoped):
                            a = _negotiated_amount(m["text"])
                            if a:
                                latest = a
                                break
                        if latest:
                            merged["amount"] = latest
                        elif merged.get("amount") in (None, "", []):
                            merged["amount"] = _latest_amount(scoped)

                        # A divisible request states the TOTAL; each member owes a
                        # share. Divide only when the headcount is known — from the
                        # message ("split 5 ways") or from the caller's `participants`.
                        # Skipped when the asker already stated the per-person figure
                        # ("5000, thats 1000 each"), since _negotiated_amount picks
                        # that up and dividing again would give 200.
                        if src.get("divisible") and merged.get("amount"):
                            share = _per_person_share(src.get("text") or "",
                                                      merged["amount"], participants)
                            if share:
                                merged["amount"] = share
                            elif (_TOTAL_AMT.search(src.get("text") or "")
                                  and not _PER_PERSON_AMT.search(src.get("text") or "")):
                                # Divisible, and the figure we hold is the TOTAL, but the
                                # headcount is unknown (no `participants`) or the split is
                                # uneven. Blank the field rather than pre-fill 5000 when
                                # the person owes 1000 — the whole point of computing a
                                # share is not to put the wrong number on a payment sheet.
                                merged["amount"] = None
                                # Setting None is not enough: the triggered_by sync below
                                # only overwrites with truthy values, so the total would
                                # survive from src["slots"]. Mark it for removal.
                                blanked.add("amount")

                    # Same problem, ride side: the recorded request holds the
                    # destination that was asked for, which may have been changed
                    # since ("actually make it indiranagar") or may never have been
                    # in the request at all ("where to?" -> "whitefield").
                    if intent == "ride":
                        # Only the rider can change where they are going.
                        latest_dest = _latest_destination(scoped, rider=src.get("sender"))
                        if latest_dest:
                            merged["destination"] = latest_dest
                        if merged.get("time") in (None, "", []):
                            merged["time"] = _latest_time(scoped)
                    result["slots"] = merged

                    # Keep triggered_by.slots in step with the effective values.
                    # API_MOBILE tells both clients to read triggered_by.slots
                    # FIRST and fall back to the top level, so leaving the original
                    # figure here means the negotiated one never reaches the user:
                    # $1000 agreed, $2000 pre-filled on the payment sheet. The
                    # request's own wording stays in triggered_by.text.
                    # Only the slots this intent can actually use. One message can open
                    # both ("send me 900 and book me a cab to hsr"), and without this
                    # the payment sheet came back carrying destination=hsr alongside
                    # the amount. _INTENT_SLOT_KEYS is the same map slot extraction
                    # already uses, so the two stay in step.
                    allowed = set(_INTENT_SLOT_KEYS.get(intent, []))
                    tb_slots = {k: v for k, v in (src["slots"] or {}).items()
                                if k in allowed}
                    for k, v in merged.items():
                        if k in allowed and v not in (None, "", []):
                            tb_slots[k] = v
                    for k in blanked:
                        tb_slots.pop(k, None)

                    # Continuation of the same negotiation — no new request since the
                    # last fire — so keep the figures that fire resolved to. Without
                    # this, "spot me 5000" / "how about 3000" / "yeah 3000 helps"
                    # (fires 3000) / "sending" put 5000 on the second sheet, because
                    # scoping to "after the last fire" hid the counter-offer.
                    if continues:
                        prev = request_meta.last_resolved.get(room_id, {}).get(intent, {})
                        for k, v in prev.items():
                            if k in allowed and v not in (None, "", []):
                                tb_slots[k] = v
                                merged[k] = v
                        result["slots"] = merged
                    request_meta.remember_resolved(room_id, intent, tb_slots)
                    # A real request was answered — close the negotiation off so its
                    # figures cannot leak into the next one.
                    request_meta.mark_settled(room_id, intent)
                    result["conversation_state"]["triggered_by"]["slots"] = tb_slots

                    if intent == "money":
                        amt = merged.get("amount") or (src["slots"] or {}).get("amount")
                        # result["money"] can exist as an explicit None, so
                        # setdefault is not enough.
                        money = result.get("money") or {}
                        if amt and not money.get("detected_amount"):
                            money["detected_amount"] = amt
                            result["money"] = money
                    break
        else:
            # A request is a message the intent model scores as money/ride but that the
            # classifier did not fire on — i.e. someone asking rather than committing.
            # Record under EVERY managed intent that crosses threshold. One message
            # routinely opens both — "send me 900 and book me a cab to hsr" scores
            # money 0.81 and ride 0.96. This used to `break` after the first match,
            # and because MANAGED_INTENTS is a set the winner depended on iteration
            # order: the same conversation recorded money on one run and ride on the
            # next, so the payment sheet had an amount only some of the time.
            for intent in sorted(MANAGED_INTENTS):
                # An echo the dedup just swallowed is NOT a request. `fired` was
                # reassigned to the deduped list above, so without this check a
                # suppressed restatement looks identical to "the classifier stayed
                # silent" — it got recorded as a fresh request and cleared the fire
                # record, which let the NEXT restatement through. Both reported
                # double-prompts came from this: "Sure" -> "Sending asap" (suppressed,
                # but cleared the record) -> "Got intent for sending asap" (fired).
                if intent in _classifier_fired:
                    continue
                thr = model_state["thresholds"].get(intent, 0.5)
                if result["scores"].get(intent, 0) >= thr:
                    request_meta.record(room_id, intent, sender, text, message_id,
                                        result.get("slots"),
                                        divisible=bool(_DIVISIBLE.search(text)))
                    # A fresh request reopens the intent, so the next commitment is a
                    # real fire rather than an echo of the previous one.
                    request_meta.clear_fired(room_id, intent)
        sm_result = {"action": None, "pending_intent": None}
        if "money" not in result["intents"]:
            result["money"] = None
        elif not result.get("money"):
            result["money"] = _enrich_money(text)
        if result["intents"] and not result.get("target"):
            result["target"] = detect_target(text, result["intents"], result.get("money"))
        elif not result["intents"]:
            result["target"] = None
    else:
        sm_result = conversation_sm.process(
            room_id=room_id,
            text=text,
            sender=sender,
            model_result=result,
            slots=result.get("slots"),
            message_id=message_id,
            reply_to=reply_to,
            model=model_state["model"] if model_state.get("has_response_head") else None,
            current_cls=result.get("_cls_embedding"),
        )
        result["intents"] = sm_result["intents"]
        result["conversation_state"] = sm_result.get("conversation_state")

    # If state machine changed intents, update dependent fields
    if sm_result["action"] == Action.STORE_PENDING:
        # Request stored — suppress money info and target since intent isn't firing yet
        if "money" not in result["intents"]:
            result["money"] = None
        if "ride" not in result["intents"]:
            if result.get("target", {}).get("intent") == "ride":
                result["target"] = None
    elif sm_result["action"] == Action.FIRE and sm_result.get("pending_intent"):
        # Firing a pending intent from a previous message — re-enrich if needed
        fired_intent = sm_result["pending_intent"]
        if fired_intent == "money" and not result.get("money"):
            result["money"] = _enrich_money(text)
        if result["intents"] and not result.get("target"):
            result["target"] = detect_target(text, result["intents"], result.get("money"))

    # Phase 5: Lifecycle (cancel/defer/confirm)
    if room_id:
        result = intent_lifecycle.process(room_id, text, result)
        conversation_ctx.add(room_id, text, result["intents"], result["scores"],
                             sender=sender)

    # Phase 5b: A money/ride fire that came from a RESPONSE is aimed at the responder.
    #
    # detect_target() reads the message in isolation, so on "venmo me 20" -> "sure" it
    # scores the pair as a payment request and returns show_to "others" — pointing the
    # payment sheet at the person who ASKED for the money rather than the one who just
    # agreed to send it. Both decision paths did this.
    #
    # Under two-phase firing the message that fires is by definition the acceptance —
    # but the actor is NOT always its sender. Accepting a request means you pay;
    # accepting an offer means the other person pays. Resolving that needs the previous
    # turn, so ask the window instead of assuming. This does not touch immediate fires:
    # a bare "venmo me 20" never reaches here with status "fired", and keeps "others".
    if (result.get("conversation_state") or {}).get("status") == "fired":
        if any(i in result["intents"] for i in MANAGED_INTENTS):
            if _resolve_payer(room_id, sender) != sender:
                result["target"] = {"show_to": "others", "reason": "accepted_offer"}
            else:
                result["target"] = {"show_to": "sender", "reason": "accepted_request"}

    # Phase 6: Guardrails — US compliance (PCI, FTC, BSA/AML)
    guardrails = run_guardrails(text, result["intents"])
    if guardrails:
        result["guardrails"] = guardrails
        if guardrails.get("blocked"):
            result["intents"] = []
            result["money"] = None
            result["slots"] = None
            result["target"] = None
            result["context_boosted"] = None

    # Phase 7: Only surface the intents the product actually ships.
    #
    # The model has nine heads, but only money and ride have been trained and evaluated
    # to the standard we need — the other seven were split out of the training data
    # months ago and never properly retrained, so they misfire (a bare "Sure" scoring
    # calendar, "lunch at 1?" scoring calendar, "please?" scoring food_order). Product
    # decision on 2026-08-05: money and ride only until each remaining intent gets its
    # own training round.
    #
    # This filters the OUTPUT only. Scores for all nine are left untouched so the
    # dogfood logs still record what would have fired — that is the data the later
    # per-intent training rounds will be built from. Set PAYCHAT_ACTIVE_INTENTS to
    # widen it again, e.g. "money,ride,contact".
    if ACTIVE_INTENTS is not None:
        kept = [i for i in result["intents"] if i in ACTIVE_INTENTS]
        if kept != result["intents"]:
            result["intents"] = kept
            if not kept:
                result["target"] = None
                result["money"] = None
                result["slots"] = None
                result["conversation_state"] = None
                result["lifecycle"] = None

    # Clean internal fields
    result.pop("_text", None)
    result.pop("_cls_embedding", None)

    _log_message(room_id, sender, text, result)
    return result


# ── Dogfood logging ─────────────────────────────────────────────────────────────
# Every number the model has been tuned against came from generated conversations.
# This captures REAL ones — but only from consented testers, and only while the app
# is not end-to-end encrypted. Once E2EE ships this stops being possible at all.
#
# Two ways to switch it on, both requiring an explicit expiry:
#   PAYCHAT_LOG_ROOMS=dm_1_2,dm_3_4   log only these rooms
#   PAYCHAT_LOG_ALL=1                 log every room
#
# PAYCHAT_LOG_ALL is fine while the only people using the product are the team who
# agreed to it. It stops being fine the moment anyone outside that group has an
# account — at which point switch to the room list. That is a judgement about who is
# using the app, not about the code, so it has to be made deliberately each time.
#
# The expiry is required either way and is the safeguard that actually matters:
# "we'll turn it off later" is not a safeguard, a date the code enforces is.
_LOG_ROOMS = {r.strip() for r in os.environ.get("PAYCHAT_LOG_ROOMS", "").split(",")
              if r.strip()}
_LOG_ALL = os.environ.get("PAYCHAT_LOG_ALL", "").strip() in ("1", "true", "yes")
_LOG_UNTIL = os.environ.get("PAYCHAT_LOG_UNTIL", "")      # YYYY-MM-DD, required
_LOG_PATH = Path(os.environ.get("PAYCHAT_LOG_PATH",
                                str(Path(__file__).resolve().parent / "dogfood.jsonl")))

if (_LOG_ROOMS or _LOG_ALL) and not _LOG_UNTIL:
    raise SystemExit("message logging is enabled but PAYCHAT_LOG_UNTIL is not set — "
                     "refusing to log without an expiry date")
if _LOG_ALL:
    logger.warning(f"DOGFOOD LOGGING ON for ALL ROOMS until {_LOG_UNTIL} -> {_LOG_PATH}"
                   f"  (every message is stored — only appropriate while the product "
                   f"is used solely by people who agreed to it)")
elif _LOG_ROOMS:
    logger.warning(f"DOGFOOD LOGGING ON for {len(_LOG_ROOMS)} room(s) until "
                   f"{_LOG_UNTIL} -> {_LOG_PATH}")


def _log_message(room_id, sender, text, result):
    """Append one line per message, until the expiry date."""
    if not _LOG_ALL and (not _LOG_ROOMS or room_id not in _LOG_ROOMS):
        return
    if datetime.utcnow().strftime("%Y-%m-%d") > _LOG_UNTIL:
        return
    try:
        row = {
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
            "room": room_id,
            "sender": sender,
            "text": text,
            "fired": [i for i in result.get("intents", []) if i in ("money", "ride")],
            "all_intents": result.get("intents", []),
            "scores": {k: v for k, v in (result.get("scores") or {}).items()
                       if k in ("money", "ride")},
            "conv": (result.get("conversation_state") or {}).get("scores"),
        }
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        # Logging must never break detection.
        logger.warning(f"dogfood log failed: {e}")


# ── Request/Response schemas ──
class DetectRequest(BaseModel):
    text: str
    chat_id: Optional[str] = None
    room_id: Optional[str] = None
    message_id: Optional[str] = None
    sender: Optional[str] = None
    context: Optional[list] = None
    reply_to: Optional[str] = None
    # How many people are in the room, INCLUDING the sender. Optional and additive —
    # every existing caller keeps working without it.
    #
    # Needed for one thing only: turning a total into a per-person share. "trip was
    # 5000, send me your shares" means 1000 each in a room of five and 2500 in a room
    # of two, and nothing in the message says which. Without it the prompt is left
    # blank rather than guessed, because a wrong figure on a payment screen is worse
    # than an empty one. Do NOT infer it from who has spoken — in a 5-person trip only
    # two people may be talking.
    participants: Optional[int] = None


class DetectResponse(BaseModel):
    intents: list
    scores: dict
    money: Optional[dict] = None
    slots: Optional[dict] = None
    target: Optional[dict] = None
    context_boosted: Optional[list] = None
    lifecycle: Optional[dict] = None
    guardrails: Optional[dict] = None
    conversation_state: Optional[dict] = None
    latency_ms: float
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    sender: Optional[str] = None


# ── Routes ──
@app.post("/detect", response_model=DetectResponse)
@app.post("/classify", response_model=DetectResponse)
def detect(req: DetectRequest):
    """
    Detect intents in a chat message (full pipeline).
    Available at both /detect and /classify.

    Returns:
      - intents: list of fired intents (e.g. ["money", "ride"])
      - scores: per-intent confidence scores
      - money: enrichment (amount, trigger_type, direction) if money fired
      - slots: extracted entities (recipient, amount, time, destination, etc.)
      - context_boosted: intents boosted by conversation context
      - lifecycle: cancel/defer/confirm state changes
      - latency_ms: inference time
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rid = req.room_id or req.chat_id
    result = full_pipeline(req.text, room_id=rid, context=req.context,
                           sender=req.sender, message_id=req.message_id,
                           reply_to=req.reply_to, participants=req.participants)
    return DetectResponse(
        **{k: v for k, v in result.items() if k in DetectResponse.model_fields},
        chat_id=req.chat_id,
        message_id=req.message_id,
        sender=req.sender,
    )


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time message detection.

    Backend sends:
      { "text": "msg", "message_id": "123", "sender": "alice" }

    Server returns:
      { ...original message..., "detection": { intents, scores, money, latency_ms } }
    """
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

            rid = msg.get("room_id") or msg.get("chat_id")
            ctx = msg.get("context")
            sndr = msg.get("sender")
            mid = msg.get("message_id")
            rto = msg.get("reply_to")
            result = full_pipeline(text, room_id=rid, context=ctx,
                                   sender=sndr, message_id=mid,
                                   reply_to=rto)

            response = {
                **msg,
                "detection": {
                    "intents":            result["intents"],
                    "scores":             result["scores"],
                    "money":              result.get("money"),
                    "slots":              result.get("slots"),
                    "target":             result.get("target"),
                    "context_boosted":    result.get("context_boosted"),
                    "lifecycle":          result.get("lifecycle"),
                    "guardrails":         result.get("guardrails"),
                    "conversation_state": result.get("conversation_state"),
                    "latency_ms":         result["latency_ms"],
                }
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info(f"WS client disconnected: {websocket.client}")


# ── Live Chat Room ──
# Multi-user WebSocket chat with real-time money detection
chat_rooms: dict[str, set[WebSocket]] = {}   # room_id -> set of connected sockets
chat_users: dict[WebSocket, dict] = {}        # socket -> {room, nickname, color}

COLORS = ["#00e5a0", "#7c5cfc", "#ff6b6b", "#ffd700", "#00b4d8", "#ff85a1", "#82e0aa", "#f0a500"]
_color_idx = 0


def _next_color():
    global _color_idx
    c = COLORS[_color_idx % len(COLORS)]
    _color_idx += 1
    return c


async def _broadcast(room_id: str, message: dict, exclude: WebSocket = None):
    """Send a message to all clients in a room."""
    if room_id not in chat_rooms:
        return
    dead = []
    for ws in chat_rooms[room_id]:
        if ws == exclude:
            continue
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead.append(ws)
    for ws in dead:
        chat_rooms[room_id].discard(ws)
        chat_users.pop(ws, None)


@app.websocket("/ws/chat/{room_id}")
async def ws_chat(websocket: WebSocket, room_id: str):
    """
    Multi-user live chat room with real-time money detection.

    Client sends:
      { "type": "join", "nickname": "Akash" }
      { "type": "message", "text": "you owe me $25" }

    Server broadcasts to all clients in the room:
      { "type": "join", "nickname": "Akash", "color": "#00e5a0", "members": [...] }
      { "type": "message", "nickname": "Akash", "color": "#00e5a0", "text": "...",
        "venmo_detection": { is_money, confidence, trigger_type, detected_amount, latency_ms },
        "timestamp": "3:28 AM" }
    """
    await websocket.accept()

    # Initialize room
    if room_id not in chat_rooms:
        chat_rooms[room_id] = set()
    chat_rooms[room_id].add(websocket)
    color = _next_color()
    chat_users[websocket] = {"room": room_id, "nickname": "Anonymous", "color": color}

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid JSON"}))
                continue

            msg_type = msg.get("type", "")

            if msg_type == "join":
                nickname = msg.get("nickname", "Anonymous").strip()[:20] or "Anonymous"
                chat_users[websocket]["nickname"] = nickname
                members = [chat_users[ws]["nickname"] for ws in chat_rooms[room_id] if ws in chat_users]
                # Broadcast join to everyone
                await _broadcast(room_id, {
                    "type": "join",
                    "nickname": nickname,
                    "color": color,
                    "members": members,
                })
                # Send color assignment back to joining user
                await websocket.send_text(json.dumps({
                    "type": "self",
                    "nickname": nickname,
                    "color": color,
                    "members": members,
                }))

            elif msg_type == "message":
                text = msg.get("text", "").strip()
                if not text:
                    continue

                nickname = chat_users[websocket]["nickname"]
                ts = datetime.utcnow().strftime("%-I:%M %p") if os.name != 'nt' else datetime.utcnow().strftime("%#I:%M %p")
                msg_id = f"{nickname}_{int(time.time()*1000)}"

                # Full pipeline: keyword → model context → slots → lifecycle
                detection = fast_keyword_detect(text)
                detection["_text"] = text

                # Phase 3: Slot extraction
                slots = extract_slots(text, detection["intents"], room_id=room_id)

                # Phase 4: Lifecycle (cancel/defer/confirm)
                detection = intent_lifecycle.process(room_id, text, detection)

                # Update context
                conversation_ctx.add(room_id, text, detection["intents"], detection["scores"])

                # Clean internal fields
                detection.pop("_text", None)

                # Build detection payload
                det_payload = None
                has_detection = (detection.get("intents") or
                                 detection.get("lifecycle"))
                if has_detection:
                    det_payload = {
                        "intents":         detection.get("intents", []),
                        "scores":          detection.get("scores", {}),
                        "money":           detection.get("money"),
                        "slots":           slots if slots else None,
                        "context_boosted": detection.get("context_boosted"),
                        "lifecycle":       detection.get("lifecycle"),
                        "latency_ms":      detection.get("latency_ms", 0),
                    }

                # Backward compat: venmo_detection for existing mobile clients
                venmo_det = None
                if detection.get("money"):
                    venmo_det = {
                        "is_money":        True,
                        "confidence":      detection["scores"].get("money", 0.8),
                        "trigger_type":    detection["money"]["trigger_type"],
                        "direction":       detection["money"]["direction"],
                        "detected_amount": detection["money"]["detected_amount"],
                        "latency_ms":      detection.get("latency_ms", 0),
                    }

                await _broadcast(room_id, {
                    "type": "message",
                    "msg_id": msg_id,
                    "nickname": nickname,
                    "color": color,
                    "text": text,
                    "timestamp": ts,
                    "detection": det_payload,
                    "venmo_detection": venmo_det,
                })

            elif msg_type == "typing":
                nickname = chat_users[websocket]["nickname"]
                await _broadcast(room_id, {
                    "type": "typing",
                    "nickname": nickname,
                }, exclude=websocket)

    except WebSocketDisconnect:
        nickname = chat_users.get(websocket, {}).get("nickname", "Unknown")
        chat_rooms.get(room_id, set()).discard(websocket)
        chat_users.pop(websocket, None)
        members = [chat_users[ws]["nickname"] for ws in chat_rooms.get(room_id, set()) if ws in chat_users]
        await _broadcast(room_id, {
            "type": "leave",
            "nickname": nickname,
            "members": members,
        })
        logger.info(f"Chat user '{nickname}' left room '{room_id}'")


@app.get("/health")
async def health():
    """Health check — returns model version and current accuracy."""
    return {
        "status":      "ok",
        "device":      str(DEVICE),
        "model_dir":   str(MODEL_DIR),
        "loaded_at":   model_state["loaded_at"],
        "version":     model_state["version"],
        "thresholds":  model_state["thresholds"],
        "intents":     INTENTS,
        "uptime_reqs": stats["requests"],
    }


@app.get("/metrics")
async def metrics():
    """Live inference metrics."""
    return {
        "requests":       stats["requests"],
        "detections":     stats["detections"],
        "detection_rate": round(stats["detections"] / max(stats["requests"], 1), 4),
        "avg_latency_ms": round(stats["avg_latency_ms"], 2),
        "started_at":     stats["started_at"],
    }


@app.post("/reload")
async def reload_model():
    """Hot-reload the model from disk (used by continuous learning scheduler)."""
    try:
        load_model()
        return {"status": "ok", "loaded_at": model_state["loaded_at"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BatchRequest(BaseModel):
    texts: list[str]


@app.post("/batch")
async def batch_detect(req: BatchRequest):
    """Batch detect — run full_pipeline on multiple texts."""
    results = []
    for text in req.texts:
        if not text.strip():
            results.append({"intents": [], "scores": {}, "fired": [], "text": text})
            continue
        r = full_pipeline(text.strip())
        r["fired"] = r.get("intents", [])
        r["text"] = text
        results.append(r)
    return {"results": results}


@app.get("/meta")
async def meta():
    """Model metadata — used by test.html."""
    v = model_state.get("version") or {}
    return {
        "labels": INTENTS,
        "thresholds": model_state.get("thresholds", {}),
        "model": "roberta-base-9intent",
        "device": str(DEVICE),
        "test_exact_match": v.get("test_exact_match", 0),
    }


# ── Team Chat (chat.html-compatible WebSocket) ──
# Separate room state so the old /ws/chat endpoint stays untouched
team_rooms: dict[str, dict] = {}  # room_id -> {users: {ws: name}, history: []}
_popup_cooldowns: dict[str, dict[str, float]] = {}  # room_id -> {intent: last_popup_ts}
POPUP_COOLDOWN_SECS = 30

def _team_room(room_id: str) -> dict:
    if room_id not in team_rooms:
        team_rooms[room_id] = {"users": {}, "history": []}
    return team_rooms[room_id]


async def _team_broadcast(room_id: str, frame: dict, exclude: WebSocket = None):
    room = team_rooms.get(room_id)
    if not room:
        return
    dead = []
    data = json.dumps(frame)
    for ws in list(room["users"]):
        if ws is exclude:
            continue
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        room["users"].pop(ws, None)


def _presence_frame(room_id: str, joined: str = None, left: str = None) -> dict:
    room = team_rooms.get(room_id, {"users": {}})
    return {
        "type": "presence",
        "users": list(room["users"].values()),
        **({"joined": joined} if joined else {}),
        **({"left": left} if left else {}),
    }


_INTENT_SLOT_MAP = {
    "money":      {"recipient", "amount", "note"},
    "ride":       {"destination", "pickup", "time"},
    "travel":     {"destination", "pickup", "time"},
    "food_order": {"food", "time"},
    "contact":    {"recipient", "phone", "time"},
    "alarm":      {"time"},
    "reminder":   {"task", "time"},
    "calendar":   {"event", "time"},
    "bills":      {"bill_name", "amount"},
}
_REQUIRED_SLOTS = {
    "money": {"amount"},
}


def _nest_slots(flat_slots: dict, fired: list) -> dict:
    """Nest flat slots into per-intent dicts with _required_filled flag."""
    if not flat_slots:
        return {}
    nested = {}
    for intent in fired:
        relevant = _INTENT_SLOT_MAP.get(intent, set())
        intent_slots = {k: v for k, v in flat_slots.items() if k in relevant}
        if intent_slots:
            required = _REQUIRED_SLOTS.get(intent, set())
            intent_slots["_required_filled"] = all(
                flat_slots.get(r) for r in required
            ) if required else True
            nested[intent] = intent_slots
    return nested if nested else None


def _pipeline_to_msg(text: str, sender: str, result: dict, room_id: str = None) -> dict:
    """Map full_pipeline() output to the frame chat.html expects."""
    fired = result.get("intents", [])
    scores = result.get("scores", {})
    flat_slots = result.get("slots")
    lifecycle = result.get("lifecycle")
    target = result.get("target") or {}

    nested_slots = _nest_slots(flat_slots, fired) if flat_slots else None

    top_scores = [{"intent": i, "prob": scores.get(i, 0)} for i in fired]
    if not top_scores:
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_scores = [{"intent": i, "prob": p} for i, p in top3 if p > 0.05]

    pir_changes = {}
    if lifecycle:
        for key in ("activated", "acknowledged", "completed", "cancelled"):
            items = lifecycle.get(key)
            if items:
                pir_changes[key] = items if isinstance(items, list) else [items]

    now = time.time()
    room_cd = _popup_cooldowns.setdefault(room_id or "_", {})
    should_popup = []
    for intent in fired:
        if now - room_cd.get(intent, 0) > POPUP_COOLDOWN_SECS:
            should_popup.append(intent)
            room_cd[intent] = now

    return {
        "type": "msg",
        "sender": sender,
        "text": text,
        "fired": fired,
        "slots": nested_slots,
        "needs_clarification": result.get("needs_clarification", []),
        "should_popup": should_popup,
        "ui_active": list(fired),
        "triggered": bool(fired),
        "target": target,
        "pir_changes": pir_changes,
        "pir_alive": [],
        "top_scores": top_scores,
        "ts": datetime.utcnow().isoformat(),
    }


@app.websocket("/ws/{room_id}/{user_name}")
async def ws_team_chat(websocket: WebSocket, room_id: str, user_name: str):
    """Chat WebSocket compatible with chat.html — runs full_pipeline on every message."""
    await websocket.accept()
    room = _team_room(room_id)
    room["users"][websocket] = user_name
    logger.info(f"[team-chat] {user_name} joined room {room_id}")

    # Send history
    await websocket.send_text(json.dumps({
        "type": "history",
        "messages": list(room["history"][-50:]),
    }))

    # Broadcast presence
    await _team_broadcast(room_id, _presence_frame(room_id, joined=user_name))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "reason": "bad JSON"}))
                continue

            if data.get("type") == "ping":
                continue

            if data.get("type") == "msg":
                text = (data.get("text") or "").strip()
                if not text:
                    continue

                result = full_pipeline(text, room_id=room_id, sender=user_name)
                frame = _pipeline_to_msg(text, user_name, result, room_id=room_id)

                room["history"].append(frame)
                if len(room["history"]) > 1000:
                    room["history"] = room["history"][-1000:]

                await _team_broadcast(room_id, frame)

    except WebSocketDisconnect:
        room["users"].pop(websocket, None)
        await _team_broadcast(room_id, _presence_frame(room_id, left=user_name))
        logger.info(f"[team-chat] {user_name} left room {room_id}")
        # Keep room history for summaries even after all users leave


@app.get("/summary/{room_id}/{user_name}")
async def user_summary(room_id: str, user_name: str):
    """Per-user intent summary — all intents detected for this user in this room."""
    room = team_rooms.get(room_id)
    if not room:
        return {"user": user_name, "room": room_id, "intents": [], "total_messages": 0}

    user_msgs = [m for m in room["history"] if m.get("sender") == user_name]

    # Aggregate: collect every fired intent with its best slots and latest timestamp
    intent_map = {}  # intent -> {slots, messages, first_ts, last_ts, confidence, status}
    for msg in user_msgs:
        fired = msg.get("fired", [])
        slots = msg.get("slots") or {}
        scores = {s["intent"]: s["prob"] for s in (msg.get("top_scores") or [])}
        ts = msg.get("ts")
        pir = msg.get("pir_changes", {})

        for intent in fired:
            if intent not in intent_map:
                intent_map[intent] = {
                    "intent": intent,
                    "slots": {},
                    "messages": [],
                    "first_ts": ts,
                    "last_ts": ts,
                    "confidence": scores.get(intent, 0),
                    "status": "active",
                    "count": 0,
                }
            entry = intent_map[intent]
            entry["count"] += 1
            entry["last_ts"] = ts
            entry["confidence"] = max(entry["confidence"], scores.get(intent, 0))
            entry["messages"].append(msg.get("text", ""))

            # Merge slots — later values overwrite earlier ones
            intent_slots = slots.get(intent, {}) if isinstance(slots.get(intent), dict) else {}
            for k, v in intent_slots.items():
                if not k.startswith("_") and v is not None:
                    entry["slots"][k] = v

        # Track cancellations/completions
        for status_key, status_val in [("cancelled", "cancelled"), ("completed", "completed")]:
            for intent in pir.get(status_key, []):
                if intent in intent_map:
                    intent_map[intent]["status"] = status_val

    # Build ordered list (most recent first)
    intents = sorted(intent_map.values(), key=lambda x: x["last_ts"] or "", reverse=True)

    return {
        "user": user_name,
        "room": room_id,
        "total_messages": len(user_msgs),
        "intents": intents,
    }


# ── Demo UI ──
DEMO_DIR = Path(os.getenv("DEMO_DIR", str(Path(__file__).resolve().parent.parent / "demo")))
DEMO_HTML = DEMO_DIR / "paychat_demo.html"


@app.get("/chat")
async def serve_chat():
    """Serve the team chat UI."""
    chat_html = DEMO_DIR / "chat.html"
    if chat_html.exists():
        return FileResponse(chat_html, media_type="text/html")
    raise HTTPException(status_code=404, detail="chat.html not found in demo/")


@app.get("/test")
async def serve_test():
    """Serve the test/eval UI."""
    test_html = DEMO_DIR / "test.html"
    if test_html.exists():
        return FileResponse(test_html, media_type="text/html")
    raise HTTPException(status_code=404, detail="test.html not found in demo/")


@app.get("/")
async def serve_demo():
    """Serve the PayChat demo UI."""
    if DEMO_HTML.exists():
        return FileResponse(DEMO_HTML, media_type="text/html")
    return {"message": "PayChat Money Detection API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
