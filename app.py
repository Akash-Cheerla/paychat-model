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
    "started_at": datetime.utcnow().isoformat(),
    "avg_latency_ms": 0.0,
    "_latency_sum": 0.0,
}


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
    latency_ms: float
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    sender: Optional[str] = None


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
    return DetectResponse(
        **result,
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

            response = {
                **msg,
                "venmo_detection": {
                    "is_money": result["is_money"],
                    "confidence": result["confidence"],
                    "trigger_type": result["trigger_type"],
                    "direction": result["direction"],
                    "detected_amount": result["detected_amount"],
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
    return {
        "requests": stats["requests"],
        "money_detected": stats["money_detected"],
        "detection_rate": round(stats["money_detected"] / max(stats["requests"], 1), 4),
        "avg_latency_ms": round(stats["avg_latency_ms"], 2),
        "started_at": stats["started_at"],
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
