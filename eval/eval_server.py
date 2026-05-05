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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# v4 Layer A slot extractor (deterministic regex + dateparser, ~5ms / message)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slot_filler import extract_slots, needs_clarification, SLOT_SCHEMA  # noqa: E402

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "saved_model"
EVAL_DIR = ROOT / "eval"
JUDGMENTS_FILE = EVAL_DIR / "judgments.jsonl"
SEED_FILE = EVAL_DIR / "seed_tests.jsonl"
HTML_FILE = EVAL_DIR / "test.html"
CHAT_HTML_FILE = EVAL_DIR / "chat.html"
CHAT_LOG_FILE = EVAL_DIR / "chat_log.jsonl"   # persistent log of every chat msg

EVAL_DIR.mkdir(exist_ok=True)

# --- load model once -------------------------------------------------------
print(f"[eval] loading model from {MODEL_DIR} ...", flush=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[eval] device: {DEVICE}", flush=True)

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE).eval()

with open(MODEL_DIR / "thresholds.json") as f:
    THRESHOLDS: dict[str, float] = json.load(f)

LABELS: list[str] = (
    list(mdl.config.id2label.values())
    if mdl.config.id2label
    else list(THRESHOLDS.keys())
)

try:
    with open(MODEL_DIR / "training_report.json") as f:
        REPORT = json.load(f)
        MODEL_VERSION = REPORT.get("model", "unknown")
        TEST_EXACT = REPORT.get("test_exact_match", 0)
except Exception:
    MODEL_VERSION = "unknown"
    TEST_EXACT = 0

print(f"[eval] loaded {type(mdl).__name__} ({sum(p.numel() for p in mdl.parameters())/1e6:.0f}M params)", flush=True)
print(f"[eval] labels: {LABELS}", flush=True)

# --- inference -------------------------------------------------------------
@torch.no_grad()
def run_model(texts: list[str]) -> list[dict[str, Any]]:
    """Batched inference. Returns list of
    {text, all_scores, fired, slots, needs_clarification}.
    """
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=128, padding=True).to(DEVICE)
    logits = mdl(**enc).logits
    probs = torch.sigmoid(logits).cpu().tolist()
    out = []
    for text, ps in zip(texts, probs):
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
        fired = [s["intent"] for s in scores if s["fired"]]
        # v4: deterministic slot extraction for the fired intents
        slots = extract_slots(text, fired) if fired else {}
        clar = needs_clarification(slots)
        out.append({
            "text": text,
            "all_scores": scores,
            "fired": fired,
            "slots": slots,
            "needs_clarification": [{"intent": i, "missing": m} for i, m in clar],
        })
    return out

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
                # Run model in threadpool to avoid blocking the event loop.
                result = await asyncio.to_thread(run_model, [text])
                r = result[0]
                msg = {
                    "type": "msg",
                    "sender": user_name,
                    "text": text,
                    "fired": r["fired"],
                    "slots": r["slots"],
                    "needs_clarification": r["needs_clarification"],
                    # only the top-5 scores so the payload stays small
                    "top_scores": r["all_scores"][:5],
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
