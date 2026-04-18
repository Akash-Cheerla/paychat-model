"""
PayChat Money Detection API
Detects money-related messages in real-time chat and returns structured data
for triggering payment flows (Venmo, CashApp, Zelle, etc.)

Endpoints:
  POST /detect       - Detect money intent in a single message
  WS   /ws/detect    - Real-time WebSocket detection stream
  GET  /health       - Health check with model version info
  GET  /metrics      - Live inference statistics
  POST /reload       - Hot-reload model from disk (zero downtime)
"""

import json
import os
import re
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paychat")

# ── Config ──
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./saved_model"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

# Popup anti-spam windows. Stateless model stays dumb; the API holds UX policy.
#   POPUP_COOLDOWN_SECONDS    - default quiet window after a popup fires.
#   DISMISSED_COOLDOWN_SECONDS- longer window when the user actively dismissed
#                                the popup (don't re-annoy).
#   POST_PAYMENT_GRACE_SECONDS- brief suppression right after payment confirms,
#                                so victory-lap messages ("sent!", "thanks")
#                                don't trigger a second popup even if a model
#                                misfire happens.
#   TRACKER_EVICTION_SECONDS  - drop tracker entries idle for this long, so the
#                                in-memory dict can't grow unbounded.
POPUP_COOLDOWN_SECONDS     = int(os.getenv("POPUP_COOLDOWN_SECONDS",     "300"))   # 5 min
DISMISSED_COOLDOWN_SECONDS = int(os.getenv("DISMISSED_COOLDOWN_SECONDS", "900"))   # 15 min
POST_PAYMENT_GRACE_SECONDS = int(os.getenv("POST_PAYMENT_GRACE_SECONDS", "60"))    # 1 min
TRACKER_EVICTION_SECONDS   = int(os.getenv("TRACKER_EVICTION_SECONDS",   "1800"))  # 30 min

MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── State ──
model_state = {
    "model": None,
    "tokenizer": None,
    "version": None,
    "loaded_at": None,
}

stats = {
    "requests": 0,
    "money_detected": 0,
    "popups_fired": 0,
    "popups_suppressed": 0,
    "started_at": datetime.utcnow().isoformat(),
    "avg_latency_ms": 0.0,
    "_latency_sum": 0.0,
}

# Per-chat popup tracker. One entry per chat_id:
#   {
#     "state":          "cooldown" | "dismissed" | "post_payment" | "idle",
#     "last_popup_ts":  float,        # unix ts of most recent fired popup
#     "last_event_ts":  float,        # unix ts of most recent state change
#     "last_amount":    str | None,   # last detected $ amount for this chat
#     "last_trigger":   str | None,   # last trigger_type for this chat
#     "popup_count":    int,          # total popups ever fired in this chat
#     "suppression_count": int,       # total popups we suppressed
#     "reason_for_current_state": str,
#   }
popup_tracker: dict = {}


# ── Model Loading ──
def load_model(model_dir: Path = MODEL_DIR):
    """Load or hot-swap model from disk."""
    logger.info(f"Loading model from {model_dir}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    model = model.to(DEVICE)
    model.eval()

    version = None
    report_path = model_dir / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        version = {
            "trained_at": report.get("trained_at"),
            "test_accuracy": report.get("test_accuracy"),
            "test_f1": report.get("test_f1"),
        }

    model_state["model"] = model
    model_state["tokenizer"] = tokenizer
    model_state["version"] = version
    model_state["loaded_at"] = datetime.utcnow().isoformat()

    if version:
        logger.info(f"Model loaded | accuracy={version['test_accuracy']:.2%} | f1={version['test_f1']:.2%}")
    else:
        logger.info("Model loaded (no training report found)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ── App ──
app = FastAPI(
    title="PayChat Money Detection API",
    description="Detects money-related messages in chat. Returns trigger type, direction, amount, and confidence.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Amount Regex ──
AMOUNT_PATTERN = re.compile(
    r'\$[\d,]+(?:\.\d{1,2})?'    # $25, $1,000.50
    r'|\b\d+\s*\$'               # 25$, 20 $
    r'|\b\d+\s*(?:dollars?|bucks?)\b',  # 25 dollars, 50 bucks
    re.IGNORECASE,
)


def _extract_amount(text: str) -> Optional[str]:
    """Extract and normalize dollar amount from text."""
    match = AMOUNT_PATTERN.search(text)
    if not match:
        return None
    amount = match.group(0)
    # Normalize "20$" -> "$20"
    if re.match(r'^\d+\s*\$', amount):
        amount = '$' + amount.replace('$', '').strip()
    return amount


# ── Trigger Classification ──
def _classify_trigger(text: str) -> str:
    """Classify what kind of money trigger was detected."""
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


# ── Direction Classification ──
def _classify_direction(text: str) -> str:
    """
    Determine money flow direction from the sender's perspective.

    Returns:
      'request' - sender is asking for money    -> popup for recipients
      'offer'   - sender is offering to pay      -> popup for sender
      'split'   - mutual split                   -> popup for everyone
    """
    t = text.lower()

    # Check OFFER first (sender wants to pay)
    offer_patterns = [
        "i owe", "i'll pay", "i'll send", "let me pay", "let me send",
        "i can pay", "i can send", "i'll venmo", "i'll cashapp", "i'll zelle",
        "shall i send", "should i send", "want me to send", "want me to pay",
        "do i owe", "how much do i owe", "i need to pay", "paying you",
        "send you", "pay you back", "i'll cover", "let me cover",
        "i got you", "my treat", "i'll get this", "on me",
        "sending you", "lemme pay", "lemme send", "ima send", "ima pay",
        "i can venmo", "i can cashapp", "i can zelle",
        "lemme venmo", "lemme cashapp", "lemme zelle",
        "venmo you", "cashapp you", "zelle you",
        "i'll get you", "let me get you", "i'll take care of",
    ]
    for p in offer_patterns:
        if p in t:
            return "offer"

    # Check SPLIT
    split_patterns = [
        "split", "halves", "half", "divide", "each",
        "chip in", "go dutch", "share the",
    ]
    for p in split_patterns:
        if p in t:
            return "split"

    # Check REQUEST (sender wants money)
    request_patterns = [
        "you owe", "owe me", "pay me", "send me", "pay up",
        "venmo me", "cashapp me", "zelle me", "where's my",
        "give me", "front me", "spot me", "cover me",
        "you still owe", "need my money",
        "hit me with", "throw me",
    ]
    for p in request_patterns:
        if p in t:
            return "request"

    return "request"


# ── Popup Cooldown Policy ──
# The model is stateless — it only answers "is this about money?". Everything
# below is the UX layer that decides whether to actually show the Venmo popup.
# Keeps popup spam out of money-heavy conversations while staying responsive to
# genuinely new transactions and payment lifecycle events.

def _evict_stale_trackers():
    """Drop chat entries with no activity in TRACKER_EVICTION_SECONDS.
    Called opportunistically so the in-memory dict can't grow unbounded."""
    now = time.time()
    stale = [
        cid for cid, s in popup_tracker.items()
        if now - s.get("last_event_ts", 0) > TRACKER_EVICTION_SECONDS
    ]
    for cid in stale:
        popup_tracker.pop(cid, None)


def _should_show_popup(chat_id: Optional[str], amount: Optional[str]):
    """
    Decide whether this money-positive message should actually trigger a popup.

    Returns:
        (should_popup, suppressed_reason, cooldown_remaining_seconds, chat_state)
    """
    now = time.time()

    # No chat_id means we can't dedupe — let the popup through. Caller should
    # always pass chat_id; this is a safety fallback.
    if not chat_id:
        return True, None, 0, "untracked"

    _evict_stale_trackers()

    state = popup_tracker.get(chat_id)

    # First money message we've ever seen for this chat
    if state is None:
        return True, None, 0, "idle"

    current = state["state"]

    # Payment recently completed — brief grace window so "sent! / thanks"
    # messages can't accidentally re-trigger a popup.
    if current == "post_payment":
        grace_elapsed = now - state["last_event_ts"]
        if grace_elapsed < POST_PAYMENT_GRACE_SECONDS:
            return False, "post_payment_grace", int(POST_PAYMENT_GRACE_SECONDS - grace_elapsed), "post_payment"
        # Grace expired — effectively idle again
        return True, None, 0, "idle"

    # State is either "cooldown" or "dismissed". Pick the right window length.
    cooldown = DISMISSED_COOLDOWN_SECONDS if current == "dismissed" else POPUP_COOLDOWN_SECONDS
    elapsed = now - state["last_popup_ts"]

    # Cooldown expired naturally → treat as idle
    if elapsed >= cooldown:
        return True, None, 0, "idle"

    # Still in cooldown — but a distinct new amount means a new transaction
    if amount and state.get("last_amount") and amount != state["last_amount"]:
        return True, None, 0, current

    # Suppress
    remaining = int(cooldown - elapsed)
    reason = "recently_dismissed" if current == "dismissed" else "cooldown_active"
    return False, reason, remaining, current


def _record_popup_fired(chat_id: str, amount: Optional[str], trigger: Optional[str]):
    """Commit a fired popup into the tracker."""
    now = time.time()
    existing = popup_tracker.get(chat_id, {})
    popup_tracker[chat_id] = {
        "state": "cooldown",
        "last_popup_ts": now,
        "last_event_ts": now,
        "last_amount": amount or existing.get("last_amount"),
        "last_trigger": trigger or existing.get("last_trigger"),
        "popup_count": existing.get("popup_count", 0) + 1,
        "suppression_count": existing.get("suppression_count", 0),
        "reason_for_current_state": "popup_just_fired",
    }
    stats["popups_fired"] += 1


def _record_popup_suppressed(chat_id: str):
    """Bump suppression counter on the tracker entry (if any)."""
    if chat_id in popup_tracker:
        popup_tracker[chat_id]["suppression_count"] = popup_tracker[chat_id].get("suppression_count", 0) + 1
    stats["popups_suppressed"] += 1


# ── Inference ──
def run_inference(text: str) -> dict:
    """Run DistilBERT model on a single message. Returns detection result."""
    t0 = time.time()

    tokenizer = model_state["tokenizer"]
    model = model_state["model"]

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
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    confidence = float(probs[1])  # P(money)
    is_money = confidence >= CONFIDENCE_THRESHOLD

    detected_amount = _extract_amount(text) if is_money else None
    trigger_type = _classify_trigger(text) if is_money else None
    direction = _classify_direction(text) if is_money else None

    latency_ms = (time.time() - t0) * 1000

    # Update stats
    stats["requests"] += 1
    stats["_latency_sum"] += latency_ms
    stats["avg_latency_ms"] = stats["_latency_sum"] / stats["requests"]
    if is_money:
        stats["money_detected"] += 1

    return {
        "is_money": is_money,
        "confidence": round(confidence, 4),
        "trigger_type": trigger_type,
        "direction": direction,
        "detected_amount": detected_amount,
        "latency_ms": round(latency_ms, 2),
    }


# ── Schemas ──
class DetectRequest(BaseModel):
    text: str
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    sender: Optional[str] = None


class DetectResponse(BaseModel):
    is_money: bool
    confidence: float
    trigger_type: Optional[str] = None
    direction: Optional[str] = None
    detected_amount: Optional[str] = None
    # Popup decision layer — read `should_popup` in the app. The rest is
    # diagnostic so backend/frontend can explain why a popup did or didn't fire.
    should_popup: bool = False
    suppressed_reason: Optional[str] = None     # "not_money" | "cooldown_active" | "recently_dismissed" | "post_payment_grace" | null
    cooldown_remaining_seconds: int = 0
    chat_state: str = "idle"                    # "idle" | "cooldown" | "dismissed" | "post_payment" | "untracked"
    latency_ms: float
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    sender: Optional[str] = None


class PaymentCompleteRequest(BaseModel):
    """Optional audit info when the app signals a successful payment."""
    amount: Optional[str] = None
    payer: Optional[str] = None
    payee: Optional[str] = None
    method: Optional[str] = None  # "venmo" | "cashapp" | "zelle" | ...


# ── Routes ──
@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """
    Detect money-related content in a chat message.

    Send every chat message here. The model returns whether it's money-related,
    what type, who should see the payment popup (direction), and the dollar amount.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server is still starting.")

    result = run_inference(req.text)

    # Opportunistic cleanup every call so non-money chatter also prunes stale entries
    _evict_stale_trackers()

    # Popup policy layer — stateless model + stateful API decision
    if not result["is_money"]:
        should_popup = False
        suppressed_reason = "not_money"
        cooldown_remaining = 0
        chat_state = (
            popup_tracker.get(req.chat_id, {}).get("state", "idle")
            if req.chat_id else "untracked"
        )
    else:
        should_popup, suppressed_reason, cooldown_remaining, chat_state = _should_show_popup(
            req.chat_id, result["detected_amount"],
        )
        if req.chat_id:
            if should_popup:
                _record_popup_fired(req.chat_id, result["detected_amount"], result["trigger_type"])
            else:
                _record_popup_suppressed(req.chat_id)
            # Report the state AFTER this call — what app team sees should match tracker truth
            chat_state = popup_tracker.get(req.chat_id, {}).get("state", chat_state)

    return DetectResponse(
        **result,
        should_popup=should_popup,
        suppressed_reason=suppressed_reason,
        cooldown_remaining_seconds=cooldown_remaining,
        chat_state=chat_state,
        chat_id=req.chat_id,
        message_id=req.message_id,
        sender=req.sender,
    )


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """
    Real-time WebSocket detection stream.

    Send: { "text": "you owe me $25", "message_id": "123", "sender": "akash" }
    Recv: { ...original fields..., "venmo_detection": { is_money, confidence, ... } }
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

            result = run_inference(text)
            chat_id = msg.get("chat_id")

            _evict_stale_trackers()

            # Same popup policy as /detect — consistent contract for the app
            if not result["is_money"]:
                should_popup = False
                suppressed_reason = "not_money"
                cooldown_remaining = 0
                chat_state = (
                    popup_tracker.get(chat_id, {}).get("state", "idle")
                    if chat_id else "untracked"
                )
            else:
                should_popup, suppressed_reason, cooldown_remaining, chat_state = _should_show_popup(
                    chat_id, result["detected_amount"],
                )
                if chat_id:
                    if should_popup:
                        _record_popup_fired(chat_id, result["detected_amount"], result["trigger_type"])
                    else:
                        _record_popup_suppressed(chat_id)
                    chat_state = popup_tracker.get(chat_id, {}).get("state", chat_state)

            response = {
                **msg,
                "venmo_detection": {
                    "is_money": result["is_money"],
                    "confidence": result["confidence"],
                    "trigger_type": result["trigger_type"],
                    "direction": result["direction"],
                    "detected_amount": result["detected_amount"],
                    "should_popup": should_popup,
                    "suppressed_reason": suppressed_reason,
                    "cooldown_remaining_seconds": cooldown_remaining,
                    "chat_state": chat_state,
                    "latency_ms": result["latency_ms"],
                },
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info(f"WS client disconnected: {websocket.client}")


@app.get("/health")
async def health():
    """Health check. Returns model version, accuracy, and device info."""
    return {
        "status": "healthy" if model_state["model"] is not None else "loading",
        "device": str(DEVICE),
        "model_dir": str(MODEL_DIR),
        "loaded_at": model_state["loaded_at"],
        "version": model_state["version"],
        "threshold": CONFIDENCE_THRESHOLD,
        "total_requests": stats["requests"],
    }


@app.get("/metrics")
async def metrics():
    """Live inference statistics."""
    money = stats["money_detected"]
    fired = stats["popups_fired"]
    suppressed = stats["popups_suppressed"]
    return {
        "requests": stats["requests"],
        "money_detected": money,
        "detection_rate": round(money / max(stats["requests"], 1), 4),
        "popups_fired": fired,
        "popups_suppressed": suppressed,
        "suppression_rate": round(suppressed / max(fired + suppressed, 1), 4),
        "active_chat_trackers": len(popup_tracker),
        "avg_latency_ms": round(stats["avg_latency_ms"], 2),
        "started_at": stats["started_at"],
    }


@app.post("/payment-complete/{chat_id}")
async def payment_complete(chat_id: str, body: Optional[PaymentCompleteRequest] = None):
    """
    Signal that a payment has been completed for this chat.

    Call this from the app (or from a Venmo/CashApp webhook) the moment the
    transaction confirms. We clear the cooldown and enter a brief grace window
    so victory-lap messages ("sent!", "thanks") don't trigger another popup.
    After the grace window the chat is fully idle again — new money topics
    pop normally.
    """
    now = time.time()
    existing = popup_tracker.get(chat_id, {})
    previous_state = existing.get("state", "idle")

    popup_tracker[chat_id] = {
        "state": "post_payment",
        "last_popup_ts": existing.get("last_popup_ts", now),
        "last_event_ts": now,
        "last_amount": (body.amount if body else None) or existing.get("last_amount"),
        "last_trigger": existing.get("last_trigger"),
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
        "chat_state": "post_payment",
        "previous_state": previous_state,
        "grace_window_seconds": POST_PAYMENT_GRACE_SECONDS,
        "popup_count_total": popup_tracker[chat_id]["popup_count"],
    }


@app.post("/popup-dismissed/{chat_id}")
async def popup_dismissed(chat_id: str):
    """
    Signal that the user dismissed the popup without paying.

    We extend the cooldown (DISMISSED_COOLDOWN_SECONDS, default 15 min) so we
    don't keep re-annoying them every time the word 'money' shows up.
    """
    now = time.time()
    existing = popup_tracker.get(chat_id, {})
    previous_state = existing.get("state", "idle")

    popup_tracker[chat_id] = {
        "state": "dismissed",
        "last_popup_ts": now,  # restart the clock from the dismiss event
        "last_event_ts": now,
        "last_amount": existing.get("last_amount"),
        "last_trigger": existing.get("last_trigger"),
        "popup_count": existing.get("popup_count", 0),
        "suppression_count": existing.get("suppression_count", 0),
        "reason_for_current_state": "user_dismissed_popup",
    }

    return {
        "status": "ok",
        "chat_id": chat_id,
        "chat_state": "dismissed",
        "previous_state": previous_state,
        "cooldown_seconds": DISMISSED_COOLDOWN_SECONDS,
    }


@app.post("/reset-cooldown/{chat_id}")
async def reset_cooldown(chat_id: str):
    """Force-clear cooldown for a chat. Admin/debug use — not called in normal flow."""
    existed = chat_id in popup_tracker
    popup_tracker.pop(chat_id, None)
    return {"status": "ok", "chat_id": chat_id, "existed": existed, "chat_state": "idle"}


@app.get("/chat-state/{chat_id}")
async def chat_state(chat_id: str):
    """
    Inspect the popup tracker for a chat. Handy when Samyak's debugging why a
    popup did or didn't fire during integration.
    """
    state = popup_tracker.get(chat_id)
    if state is None:
        return {
            "chat_id": chat_id,
            "state": "idle",
            "message": "no tracker entry — next money message will popup",
        }

    now = time.time()
    current = state["state"]
    if current == "post_payment":
        remaining = max(0, POST_PAYMENT_GRACE_SECONDS - (now - state["last_event_ts"]))
    else:
        cooldown = DISMISSED_COOLDOWN_SECONDS if current == "dismissed" else POPUP_COOLDOWN_SECONDS
        remaining = max(0, cooldown - (now - state["last_popup_ts"]))

    return {
        "chat_id": chat_id,
        "state": current,
        "last_popup_at": datetime.utcfromtimestamp(state["last_popup_ts"]).isoformat() if state.get("last_popup_ts") else None,
        "last_event_at": datetime.utcfromtimestamp(state["last_event_ts"]).isoformat() if state.get("last_event_ts") else None,
        "last_amount": state.get("last_amount"),
        "last_trigger": state.get("last_trigger"),
        "popup_count": state.get("popup_count", 0),
        "suppression_count": state.get("suppression_count", 0),
        "cooldown_remaining_seconds": int(remaining),
        "reason_for_current_state": state.get("reason_for_current_state"),
    }


@app.post("/reload")
async def reload_model_endpoint():
    """Hot-reload model from disk without restarting. Use after retraining."""
    try:
        load_model()
        return {"status": "ok", "loaded_at": model_state["loaded_at"], "version": model_state["version"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API info."""
    return {
        "service": "PayChat Money Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
