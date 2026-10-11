"""The message-summary service. Its own process, deliberately.

Summaries and intents are different models doing different jobs, and nothing in this
file imports the intent pipeline. They were briefly served from the same app and that
turned out to be a mistake in three separate ways, each measured:

  1. The event loop. A card costs 3-10s of blocking torch. Served from app.py, one
     summary took /classify from 86ms to 5,571ms - a single /classify completed during
     one 5.6s summary, because every other request queued behind it. One uvicorn worker
     and nothing to absorb it.
  2. torch's thread pool. torch.set_num_threads is process-global. The summary wants 2
     (measured: 2.8s/card vs 4.0s on 4). The intent model normally runs on ONNX and is
     unaffected - but it has a PyTorch fallback, and if model.onnx is ever missing from
     an LFS pull, the summary silently halves the intent path's threads.
  3. Memory. The student peaks at 1.4GB. Sharing a container with the intent models put
     that against the same limit, so a long message could push out the path that has to
     stay up.

Run it next to the intent service, or on another box - it holds no state and shares
nothing with it:

    uvicorn summary_app:app --host 0.0.0.0 --port 8001

/classify is unaffected by anything that happens in here, including this process dying.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

import summary_engines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("summary")

app = FastAPI(title="PayChat message summaries", version="1")


def _utc_iso():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


# The student is ~1GB and takes ~35s to load, so it loads once on first use rather than
# per request or at import. None means the weights are not on this box, and
# production_summary() then serves the rules card - the feature degrades, the service
# does not break.
_STUDENT = {"loaded": False, "engine": None}

# How many cards one batch request may ask the MODEL for. At ~3s a card an unbounded
# batch is a timeout; the gate rejects ~99% of real messages so this is never reached in
# practice. Past it the rules answer instantly instead of the caller getting nothing.
_BATCH_MODEL_CALLS = int(os.environ.get("PAYCHAT_SUMMARY_BATCH_CALLS", "12"))


def _student():
    if not _STUDENT["loaded"]:
        _STUDENT["engine"] = summary_engines.default_student()
        _STUDENT["loaded"] = True
        logger.info("summary student: %s",
                    _STUDENT["engine"].model_dir
                    if _STUDENT["engine"] else "not installed, using rules")
    return _STUDENT["engine"]


class MessageSummaryRequest(BaseModel):
    """One message, or a batch of them, to summarise for the Summary tab."""
    text: Optional[str] = None
    # [{"message_id": "...", "text": "..."}] - a whole screen in one round trip, so
    # opening the tab is not N requests.
    messages: Optional[list] = None


@app.get("/summary-status")
def summary_status():
    # NOT async: the first call LOADS the model, ~35s of blocking work.
    """Did the summary model actually load on this box?

    Without this the only way to tell is to send a long message and read `source` off
    the reply. The endpoint never fails when the weights are missing - it quietly serves
    the rules card - which is the right behaviour and also the kind of thing that stays
    unnoticed for a week. One curl after a deploy answers it.

    `threads` is what the model will actually use, not the process default, so it can be
    checked against the deploy doc before the first card rather than after.
    """
    import torch
    student = _student()
    # The model's thread count is applied in its lazy load, so before the first card
    # torch.get_num_threads() is still the process default - reporting that would be a
    # lie on a fresh box, which is the one place this endpoint gets read.
    loaded = student is not None and student.is_loaded()
    return {
        "student_loaded": student is not None,
        "model_dir": student.model_dir if student else None,
        "serving": "student+rules" if student else "rules only (weights not found)",
        "threads": summary_engines.SUMMARY_THREADS,
        "threads_live": torch.get_num_threads() if loaded else None,
        "weights_resident": loaded,
        "batch_model_calls": _BATCH_MODEL_CALLS,
        "as_of": _utc_iso(),
    }


@app.post("/message-summary")
def message_summary(req: MessageSummaryRequest):
    # NOT async. A card costs 3-10s of blocking torch and an async def runs ON the event
    # loop, so it would stall every other request to this process for the whole
    # inference. A sync def is run in FastAPI's threadpool instead. Measured: as an
    # async def this held an unrelated request for 5.5s; as a sync def, 189ms.
    """A topic and a few bullets for a long message, or nothing when there is nothing
    worth showing.

    Here so that iOS and Android show the SAME card. iOS summarises on-device today
    (Apple Foundation Models) and Android does not summarise at all, which means a fix
    to the wording ships with an App Store release and never reaches Android. One
    implementation, one place to fix.

    `summary` is null when the message is too short, is mostly links, or yields nothing
    but topic labels - no card is better than a card of noise. Callers render a card
    only when `summary` is present.

    Nothing is stored. This reads the text it is given and returns.
    """
    student = _student()
    source = "student+rules" if student else "rules"
    if req.messages is not None:
        # The model costs ~3s a card, so an unbounded batch is a timeout waiting to
        # happen: 100 long messages would hold the request for five minutes. In practice
        # the gate rejects ~99% and a batch of 100 real messages needs one or two model
        # calls, so this budget is never reached - it exists for the pathological batch.
        # Past it, the rules still answer, instantly, rather than the caller getting
        # nothing.
        budget = _BATCH_MODEL_CALLS
        out = []
        for m in req.messages[:100]:
            if not isinstance(m, dict):
                continue
            use = student if budget > 0 else None
            before = budget
            card = summary_engines.production_summary(m.get("text") or "", use)
            if card is not None and use is not None:
                budget -= 1
            out.append({"message_id": m.get("message_id"), "summary": card,
                        "source": source if before > 0 else "rules"})
        return {"results": out, "model_calls_used": _BATCH_MODEL_CALLS - budget,
                "as_of": _utc_iso()}
    return {"summary": summary_engines.production_summary(req.text or "", student),
            "source": source,
            "as_of": _utc_iso()}


@app.get("/health")
def health():
    return {"ok": True, "service": "summary", "as_of": _utc_iso()}
