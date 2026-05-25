"""
FYOE eval harness — local web app for stress-testing the model with hundreds
of messages a day. Logs every judgment to JSONL, exposes stats, and exports
failures as training data for v4.

Run from project root:
    python eval/eval_server.py

Then open http://localhost:8001 in your browser.

Endpoints:
    GET  /                    -> the test UI (eval/test.html)
    POST /detect              -> run model on text, return all 18 intent scores
    POST /judge               -> save user verdict for a tested message
    GET  /stats               -> running accuracy + per-intent precision/recall
    GET  /failures            -> list of saved failures
    GET  /seed                -> seed_tests.jsonl as JSON
    GET  /export              -> failures formatted as training-data JSONL
    POST /batch               -> run model on a list of texts at once
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

# ONNX or PyTorch — set USE_ONNX=1 env var to use ONNX Runtime
USE_ONNX = os.environ.get("USE_ONNX", "0") == "1"

if USE_ONNX:
    import onnxruntime as ort
else:
    import torch
    from transformers import AutoModelForSequenceClassification

# v4 Layer A slot extractor (deterministic regex + dateparser, ~5ms / message)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slot_filler import extract_slots, needs_clarification, SLOT_SCHEMA  # noqa: E402
from trigger_filter import should_process, matched_intents  # noqa: E402
from context_accumulator import ConversationContext, _REJECTION_EXACT, _CANCEL_PATTERNS, _PURE_FILLER  # noqa: E402
from smart_postprocess import smart_suppress  # noqa: E402

# v5 conversation context — tracks multi-turn history per chat room
CONV_CTX = ConversationContext(window_size=5, stale_seconds=1800)

# Popup cooldown — prevents the same intent re-animating every message in a convo.
# should_popup=False during cooldown (don't re-animate), ui_active=True (keep button visible).
POPUP_COOLDOWN: dict[str, dict[str, float]] = {}  # room_id -> {intent: last_popup_ts}
COOLDOWN_SECS = 30

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(ROOT / "saved_model")))
EVAL_DIR = ROOT / "eval"
JUDGMENTS_FILE = EVAL_DIR / "judgments.jsonl"
SEED_FILE = EVAL_DIR / "seed_tests.jsonl"
HTML_FILE = EVAL_DIR / "test.html"
CHAT_HTML_FILE = EVAL_DIR / "chat.html"
CHAT_LOG_FILE = EVAL_DIR / "chat_log.jsonl"   # persistent log of every chat msg

EVAL_DIR.mkdir(exist_ok=True)

# --- load model once -------------------------------------------------------
print(f"[eval] loading model from {MODEL_DIR} ...", flush=True)
print(f"[eval] backend: {'ONNX Runtime' if USE_ONNX else 'PyTorch'}", flush=True)

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)

with open(MODEL_DIR / "thresholds.json") as f:
    THRESHOLDS: dict[str, float] = json.load(f)

if USE_ONNX:
    # ONNX Runtime inference
    onnx_model_path = MODEL_DIR / "model_quantized.onnx"
    if not onnx_model_path.exists():
        onnx_model_path = MODEL_DIR / "model.onnx"
    print(f"[eval] loading ONNX model: {onnx_model_path.name}", flush=True)
    ort_session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
    mdl = None
    DEVICE = "cpu"
    # Load labels from config.json
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)
    LABELS: list[str] = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
    print(f"[eval] ONNX session ready ({onnx_model_path.stat().st_size / 1e6:.0f}MB)", flush=True)
else:
    # PyTorch inference
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device: {DEVICE}", flush=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE).eval()
    ort_session = None
    LABELS: list[str] = (
        list(mdl.config.id2label.values())
        if mdl.config.id2label
        else list(THRESHOLDS.keys())
    )
    print(f"[eval] loaded {type(mdl).__name__} ({sum(p.numel() for p in mdl.parameters())/1e6:.0f}M params)", flush=True)

try:
    with open(MODEL_DIR / "training_report.json") as f:
        REPORT = json.load(f)
        MODEL_VERSION = REPORT.get("model", "unknown")
        TEST_EXACT = REPORT.get("test_exact_match", 0)
except Exception:
    MODEL_VERSION = "unknown"
    TEST_EXACT = 0

print(f"[eval] labels: {LABELS}", flush=True)

# --- inference -------------------------------------------------------------

def _empty_result(text: str) -> dict[str, Any]:
    """Result for a message that the trigger filter skipped."""
    return {
        "text": text,
        "all_scores": [{"intent": l, "prob": 0.0, "threshold": THRESHOLDS.get(l, 0.5), "fired": False} for l in LABELS],
        "fired": [],
        "slots": {},
        "needs_clarification": [],
        "triggered": False,
        "trigger_hints": [],
    }


def _run_inference(input_ids, attention_mask) -> list[list[float]]:
    """Run inference through ONNX or PyTorch and return probabilities."""
    if USE_ONNX:
        if hasattr(input_ids, 'numpy'):
            input_ids_np = input_ids.numpy()
            attention_mask_np = attention_mask.numpy()
        else:
            input_ids_np = input_ids
            attention_mask_np = attention_mask
        logits = ort_session.run(None, {
            "input_ids": input_ids_np,
            "attention_mask": attention_mask_np,
        })[0]
        # sigmoid
        probs = (1 / (1 + np.exp(-logits))).tolist()
        return probs
    else:
        import torch as _torch
        with _torch.no_grad():
            logits = mdl(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = _torch.sigmoid(logits).cpu().tolist()
        return probs


def run_model(texts: list[str], use_trigger: bool = True) -> list[dict[str, Any]]:
    """Batched inference. Returns list of
    {text, all_scores, fired, slots, needs_clarification, triggered, trigger_hints}.

    If use_trigger=True, messages without action keywords are skipped entirely
    (casual chat like "lol", "what's up" never hits the model).
    """
    # Split into triggered vs skipped
    if use_trigger:
        to_run = []
        indices = []
        results = [None] * len(texts)
        for i, text in enumerate(texts):
            if should_process(text):
                to_run.append(text)
                indices.append(i)
            else:
                results[i] = _empty_result(text)
        if not to_run:
            return results  # type: ignore
    else:
        to_run = texts
        indices = list(range(len(texts)))
        results = [None] * len(texts)

    if USE_ONNX:
        enc = tok(to_run, return_tensors="np", truncation=True, max_length=128, padding=True)
        probs = _run_inference(enc["input_ids"], enc["attention_mask"])
    else:
        enc = tok(to_run, return_tensors="pt", truncation=True, max_length=128, padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        probs = _run_inference(enc["input_ids"], enc["attention_mask"])

    for idx, text, ps in zip(indices, to_run, probs):
        scores = []
        for i, label in enumerate(LABELS):
            thr = THRESHOLDS.get(label, 0.5)
            scores.append({
                "intent": label,
                "prob": round(ps[i], 4),
                "threshold": thr,
                "fired": ps[i] >= thr,
            })
        scores.sort(key=lambda s: -s["prob"])
        fired_raw = [s["intent"] for s in scores if s["fired"]]
        # v5.1: smart post-processing — suppress false co-fires
        fired = smart_suppress(text, fired_raw)
        # v4: deterministic slot extraction for the fired intents
        slots = extract_slots(text, fired) if fired else {}
        clar = needs_clarification(slots)
        results[idx] = {
            "text": text,
            "all_scores": scores,
            "fired": fired,
            "slots": slots,
            "needs_clarification": [{"intent": i, "missing": m} for i, m in clar],
            "triggered": True,
            "trigger_hints": matched_intents(text),
        }
    return results  # type: ignore


def run_model_with_context(text: str, context: str | None = None, use_trigger: bool = True) -> dict[str, Any]:
    """Single-message inference with optional conversation context.

    When context is provided, the tokenizer encodes (context, text) as a
    text pair — same format the v5 model was trained on:
        <s> prior msg 1 | prior msg 2 </s></s> current message </s>

    If context is provided and the current message alone doesn't pass the
    trigger filter, we STILL run the model — because the context might make
    an otherwise-ambiguous message actionable ("dominos?" after "I'm hungry").
    """
    # With context, always run the model (context can make ambiguous messages actionable)
    if use_trigger and not context:
        if not should_process(text):
            return _empty_result(text)

    ret_tensors = "np" if USE_ONNX else "pt"
    if context:
        enc = tok(context, text, return_tensors=ret_tensors, truncation=True, max_length=128)
    else:
        enc = tok(text, return_tensors=ret_tensors, truncation=True, max_length=128)

    if not USE_ONNX:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

    probs_batch = _run_inference(enc["input_ids"], enc["attention_mask"])
    probs = probs_batch[0]

    scores = []
    for i, label in enumerate(LABELS):
        thr = THRESHOLDS.get(label, 0.5)
        scores.append({
            "intent": label,
            "prob": round(probs[i], 4),
            "threshold": thr,
            "fired": probs[i] >= thr,
        })
    scores.sort(key=lambda s: -s["prob"])
    fired_raw = [s["intent"] for s in scores if s["fired"]]
    # v5.1: smart post-processing — suppress false co-fires
    fired = smart_suppress(text, fired_raw)
    slots = extract_slots(text, fired) if fired else {}
    clar = needs_clarification(slots)
    return {
        "text": text,
        "context": context,
        "all_scores": scores,
        "fired": fired,
        "slots": slots,
        "needs_clarification": [{"intent": i, "missing": m} for i, m in clar],
        "triggered": True,
        "trigger_hints": matched_intents(text),
    }


# --- FastAPI ---------------------------------------------------------------
app = FastAPI(title="FYOE eval harness")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectReq(BaseModel):
    text: str

class BatchReq(BaseModel):
    texts: list[str]

class ClassifyReq(BaseModel):
    text: str
    context: list[str] = []
    room_id: str = ""

class JudgeReq(BaseModel):
    text: str
    predicted: list[str]
    expected: list[str]
    expected_entities: dict[str, str] = {}
    verdict: str  # "correct" | "wrong_intent" | "missing_intent" | "extra_intent" | "wrong_entity"
    note: str = ""
    tag: str = ""

@app.get("/", response_class=HTMLResponse)
def index():
    if not HTML_FILE.exists():
        return HTMLResponse("<h1>test.html missing</h1>", status_code=500)
    return HTMLResponse(HTML_FILE.read_text(encoding="utf-8"))


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    """Shareable two-person chat that runs every message through the model."""
    if not CHAT_HTML_FILE.exists():
        return HTMLResponse("<h1>chat.html missing</h1>", status_code=500)
    return HTMLResponse(CHAT_HTML_FILE.read_text(encoding="utf-8"))


@app.post("/classify")
def classify(req: ClassifyReq):
    """
    Main integration endpoint (see INTEGRATION.md).
    Returns fired intents, slots, and two UI control fields:
      should_popup — fire a new popup animation (event, respects 30s cooldown)
      ui_active    — keep the action button visible (state, always true when fired)
    """
    room_id = req.room_id or "api"
    # Server-side safety net: drop context for rejections/filler even if client sends it.
    # The client SHOULD follow the context rules in INTEGRATION.md, but we double-check.
    ctx_str = None
    if req.context:
        ct = req.text.strip().lower().rstrip("!.?")
        if ct not in _REJECTION_EXACT and not _CANCEL_PATTERNS.match(ct) and ct not in _PURE_FILLER:
            ctx_str = " | ".join(req.context)
    r = run_model_with_context(req.text, ctx_str)

    # Popup cooldown logic
    now = time.time()
    room_cooldowns = POPUP_COOLDOWN.setdefault(room_id, {})
    should_popup: list[str] = []
    ui_active: list[str] = list(r["fired"])
    for intent in r["fired"]:
        if now - room_cooldowns.get(intent, 0) > COOLDOWN_SECS:
            should_popup.append(intent)
            room_cooldowns[intent] = now

    scores_list = [
        {"intent": s["intent"], "prob": s["prob"], "fired": s["intent"] in r["fired"]}
        for s in r["all_scores"]
    ]
    return {
        "fired": r["fired"],
        "slots": r["slots"],
        "needs_clarification": r["needs_clarification"],
        "scores": scores_list,
        "should_popup": should_popup,   # trigger new popup animation
        "ui_active": ui_active,         # keep action button visible
    }


# ───────────────────────────────────────────────────────────────────────
#  CHAT — WebSocket two-person chat with live model annotations.
#  Rooms are in-memory; messages are also appended to chat_log.jsonl so
#  you can mine real conversations later for v5 training data.
# ───────────────────────────────────────────────────────────────────────
ROOMS: dict[str, dict[str, Any]] = {}


def _get_room(rid: str) -> dict[str, Any]:
    if rid not in ROOMS:
        ROOMS[rid] = {"sockets": {}, "messages": []}
    return ROOMS[rid]


async def _broadcast(room: dict[str, Any], payload: dict[str, Any]) -> None:
    dead: list[str] = []
    for name, ws in list(room["sockets"].items()):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(name)
    for n in dead:
        room["sockets"].pop(n, None)


def _append_chat_log(room_id: str, msg: dict[str, Any]) -> None:
    rec = {"room": room_id, **msg}
    try:
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[chat] failed to append log: {e}", flush=True)


@app.websocket("/ws/{room_id}/{user_name}")
async def ws_chat(websocket: WebSocket, room_id: str, user_name: str) -> None:
    await websocket.accept()
    room = _get_room(room_id)

    # If the same name is already connected, kick the old socket so reconnects work.
    old = room["sockets"].pop(user_name, None)
    if old is not None:
        try:
            await old.close()
        except Exception:
            pass
    room["sockets"][user_name] = websocket

    # Send history + current presence to the joiner.
    await websocket.send_json({
        "type": "history",
        "messages": room["messages"][-50:],
        "you": user_name,
        "room": room_id,
    })
    await _broadcast(room, {
        "type": "presence",
        "users": list(room["sockets"].keys()),
        "joined": user_name,
    })

    try:
        while True:
            data = await websocket.receive_json()
            kind = data.get("type", "msg")

            if kind == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if kind == "msg":
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                # v5: get conversation context for this room
                # Pass current text so accumulator skips context for rejections/filler
                ctx_str = CONV_CTX.get_context_string(room_id, current_text=text)
                # Run model with context in threadpool to avoid blocking.
                r = await asyncio.to_thread(run_model_with_context, text, ctx_str)
                # Record the turn so future messages have context
                CONV_CTX.record(room_id, text, sender=user_name,
                                fired=r["fired"], slots=r.get("slots", {}))
                # Popup cooldown — should_popup=event, ui_active=state
                now = time.time()
                room_cooldowns = POPUP_COOLDOWN.setdefault(room_id, {})
                should_popup: list[str] = []
                ui_active: list[str] = list(r["fired"])
                for intent in r["fired"]:
                    if now - room_cooldowns.get(intent, 0) > COOLDOWN_SECS:
                        should_popup.append(intent)
                        room_cooldowns[intent] = now
                msg = {
                    "type": "msg",
                    "sender": user_name,
                    "text": text,
                    "fired": r["fired"],
                    "slots": r["slots"],
                    "needs_clarification": r["needs_clarification"],
                    "should_popup": should_popup,   # fire new popup animation (event)
                    "ui_active": ui_active,          # keep action button visible (state)
                    "triggered": r.get("triggered", True),
                    "context_used": ctx_str is not None,
                    # only the top-5 scores so the payload stays small
                    "top_scores": r["all_scores"][:5] if r.get("triggered", True) else [],
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                room["messages"].append(msg)
                if len(room["messages"]) > 200:
                    room["messages"] = room["messages"][-200:]
                _append_chat_log(room_id, msg)
                await _broadcast(room, msg)
                continue

            # Unknown frame — just echo back as an error
            await websocket.send_json({"type": "error", "reason": f"unknown frame {kind!r}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[chat] {user_name}@{room_id} error: {e}", flush=True)
    finally:
        if room["sockets"].get(user_name) is websocket:
            room["sockets"].pop(user_name, None)
        await _broadcast(room, {
            "type": "presence",
            "users": list(room["sockets"].keys()),
            "left": user_name,
        })

@app.get("/meta")
def meta():
    return {
        "model": MODEL_VERSION,
        "labels": LABELS,
        "thresholds": THRESHOLDS,
        "test_exact_match": TEST_EXACT,
        "device": DEVICE,
        "slot_schema": SLOT_SCHEMA,
        "trigger_filter": True,
    }

@app.post("/detect")
def detect(req: DetectReq):
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    return run_model([req.text])[0]

@app.post("/batch")
def batch(req: BatchReq):
    if not req.texts:
        raise HTTPException(400, "empty list")
    return {"results": run_model(req.texts)}

@app.post("/judge")
def judge(req: JudgeReq):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "text": req.text,
        "predicted": req.predicted,
        "expected": req.expected,
        "expected_entities": req.expected_entities,
        "verdict": req.verdict,
        "note": req.note,
        "tag": req.tag,
    }
    with open(JUDGMENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, "saved": rec["ts"]}

def _load_judgments() -> list[dict[str, Any]]:
    if not JUDGMENTS_FILE.exists():
        return []
    out = []
    with open(JUDGMENTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out

@app.get("/stats")
def stats():
    js = _load_judgments()
    if not js:
        return {"total": 0, "today": 0, "today_correct": 0, "per_intent": {}, "by_tag": {}, "recent_failures": []}

    today = datetime.now(timezone.utc).date().isoformat()
    today_js = [j for j in js if j["ts"].startswith(today)]

    # Per-intent precision/recall from judgments where expected was annotated
    per_intent = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    for j in js:
        pred, exp = set(j["predicted"]), set(j["expected"])
        for label in LABELS:
            in_pred, in_exp = label in pred, label in exp
            if in_pred and in_exp:
                per_intent[label]["tp"] += 1
            elif in_pred:
                per_intent[label]["fp"] += 1
            elif in_exp:
                per_intent[label]["fn"] += 1

    for label, m in per_intent.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        m["precision"] = round(tp / (tp + fp), 3) if tp + fp else None
        m["recall"] = round(tp / (tp + fn), 3) if tp + fn else None
        m["support"] = tp + fn

    # By tag
    by_tag: dict[str, dict[str, int]] = {}
    for j in js:
        t = j.get("tag") or "untagged"
        by_tag.setdefault(t, {"total": 0, "correct": 0})
        by_tag[t]["total"] += 1
        if j["verdict"] == "correct":
            by_tag[t]["correct"] += 1

    failures = [j for j in js if j["verdict"] != "correct"]

    return {
        "total": len(js),
        "total_correct": sum(1 for j in js if j["verdict"] == "correct"),
        "today": len(today_js),
        "today_correct": sum(1 for j in today_js if j["verdict"] == "correct"),
        "per_intent": per_intent,
        "by_tag": by_tag,
        "recent_failures": failures[-30:],
    }

@app.get("/failures")
def failures():
    js = _load_judgments()
    return {"failures": [j for j in js if j["verdict"] != "correct"]}

@app.get("/seed")
def seed():
    if not SEED_FILE.exists():
        return {"cases": []}
    cases = []
    with open(SEED_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return {"cases": cases}

@app.get("/export", response_class=PlainTextResponse)
def export():
    """Failures formatted as training-data JSONL — drop into training/data/."""
    js = _load_judgments()
    out_lines = []
    seen = set()
    for j in js:
        if j["verdict"] == "correct":
            continue
        key = j["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out_lines.append(json.dumps({"text": j["text"], "intents": j["expected"]}, ensure_ascii=False))
    return "\n".join(out_lines) + ("\n" if out_lines else "")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    print(f"\n[eval] FYOE eval harness on port {port}", flush=True)
    print(f"[eval]   single-message tester:  http://localhost:{port}/", flush=True)
    print(f"[eval]   shareable chat:         http://localhost:{port}/chat", flush=True)
    print(f"[eval] to share with a friend on the internet:", flush=True)
    print(f"[eval]   1) install ngrok (https://ngrok.com/download)", flush=True)
    print(f"[eval]   2) in another terminal:  ngrok http {port}", flush=True)
    print(f"[eval]   3) send the https://...ngrok-free.app/chat URL to your friend\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
