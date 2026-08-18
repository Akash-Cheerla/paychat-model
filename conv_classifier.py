"""
Conversation classifier — decides whether money/ride should fire on the current message.

Replaces the rule-based decision layer. Instead of judging a reply from two isolated
messages, it reads the recent conversation window and answers one question directly.

The rendering here MUST match training exactly (see conv_model/model_info.json):
  * speakers renamed so the sender of the LAST message is always "A"
  * a request spliced in from the pending store is marked "[earlier]"
  * messages joined with " | "
  * truncation from the LEFT, so the message being classified is never cut

Any drift between this and the training-time rendering silently degrades accuracy, so
render() is duplicated verbatim rather than imported from the notebook.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("paychat.conv")

try:
    import torch
    import torch.nn as nn
    from transformers import RobertaModel, RobertaConfig, AutoTokenizer
    from safetensors.torch import load_file
except ImportError:      # server can still boot without it
    torch = None

LABELS = ["money", "ride"]
MAX_LEN = 192
WINDOW = 6


class ConvFireModel(nn.Module if torch else object):
    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.roberta = backbone
        h = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fire_head = nn.Linear(h, len(LABELS))

    def forward(self, input_ids, attention_mask=None):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.fire_head(self.dropout(out.last_hidden_state[:, 0, :]))


def render(window):
    """Window -> the exact string format the model was trained on.

    window: [{"s": sender, "t": text, "earlier": bool?}, ...] oldest first,
            the message being classified LAST.
    """
    if not window:
        return ""
    ids = {window[-1]["s"]: "A"}
    nxt = 1
    for m in window:
        if m["s"] not in ids:
            ids[m["s"]] = chr(ord("A") + nxt)
            nxt += 1
    return " | ".join(
        f"{ids[m['s']]}:{' [earlier]' if m.get('earlier') else ''} {m['t']}"
        for m in window
    )


class ConversationClassifier:
    def __init__(self, model_dir):
        self.ok = False
        if torch is None:
            logger.warning("torch unavailable — conversation classifier disabled")
            return
        d = Path(model_dir)
        if not (d / "model.safetensors").exists():
            logger.warning(f"no conversation model at {d} — disabled")
            return

        info = json.loads((d / "model_info.json").read_text())
        self.thresholds = info.get("thresholds", {"money": 0.5, "ride": 0.5})
        # Override the trained thresholds without retraining or editing the model dir:
        #     PAYCHAT_CONV_THRESHOLDS="money=0.85,ride=0.80"
        #
        # The thresholds are chosen by an argmax over an F-beta curve that is nearly
        # flat, so they move a lot between runs on the same data (ride has been 0.825,
        # 0.38 and 0.95 across rounds) and they are not obviously the right operating
        # point. Money currently sits at 0.97 against ride's 0.825 while being the
        # weaker intent, so being able to sweep this against the coverage gate is the
        # difference between "we need another training round" and "we have been
        # shipping a badly chosen operating point".
        _ov = os.environ.get("PAYCHAT_CONV_THRESHOLDS", "").strip()
        if _ov:
            for part in _ov.split(","):
                if "=" in part:
                    k, _, v = part.partition("=")
                    try:
                        self.thresholds[k.strip()] = float(v)
                    except ValueError:
                        logger.warning("bad PAYCHAT_CONV_THRESHOLDS entry: %r", part)
            logger.info("conv thresholds overridden: %s", self.thresholds)
        # Read window and max_length FROM the model rather than hardcoding them. These
        # two numbers must match training exactly; a silent mismatch is what dropped the
        # intent model from 0.94 to 0.03 when context was first tried. v1 trained at
        # window 6 / 192 tokens, v2 at window 10 / 256, and the serving code must follow
        # whichever model is actually loaded.
        self.window = info.get("window", WINDOW)
        self.max_len = info.get("max_length", MAX_LEN)
        self.tok = AutoTokenizer.from_pretrained(str(d), use_fast=True)
        # Trained with left truncation so the current message survives. Must match.
        self.tok.truncation_side = "left"

        # Prefer the INT8 ONNX session — 3.5x faster than fp32 torch (77ms -> 22ms) with
        # no firing-decision change across the near-threshold probes in
        # export_conv_onnx.py. On a path that runs per message, on top of the main
        # model, that difference is the difference between shippable and not.
        self.session = None
        int8 = d / "conv_model_int8.onnx"
        if int8.exists():
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    str(int8), providers=["CPUExecutionProvider"])
                logger.info(f"conversation classifier loaded from {int8.name} "
                            f"(thresholds {self.thresholds})")
            except ImportError:
                logger.warning("onnxruntime unavailable — falling back to torch")

        if self.session is None:
            cfg = RobertaConfig.from_pretrained(str(d))
            self.model = ConvFireModel(RobertaModel(cfg))
            self.model.load_state_dict(load_file(str(d / "model.safetensors")),
                                       strict=False)
            self.model.eval()
            logger.info(f"conversation classifier loaded from safetensors "
                        f"(thresholds {self.thresholds})")
        self.ok = True

    @torch.no_grad() if torch else staticmethod
    def predict(self, window):
        """Returns (fired_intents, scores)."""
        if not self.ok or not window:
            return [], {}
        text = render(window)
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_len)
        if self.session is not None:
            import numpy as np
            out = self.session.run(None, {
                "input_ids": enc["input_ids"].numpy(),
                "attention_mask": enc["attention_mask"].numpy()})
            probs = 1 / (1 + np.exp(-out[0][0]))
        else:
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            probs = torch.sigmoid(logits)[0]
        scores = {n: round(float(probs[i]), 4) for i, n in enumerate(LABELS)}
        fired = [n for i, n in enumerate(LABELS)
                 if float(probs[i]) >= self.thresholds.get(n, 0.5)]
        return fired, scores


def build_window(history, current, pending_texts=(), size=WINDOW):
    """Assemble the window the model expects.

    history: [{"sender","text"}, ...] oldest first, NOT including current
    current: {"sender","text"}
    pending_texts: [{"sender","text"}] for requests still open but scrolled out of
        view. Splicing them in is what makes "distance does not matter" work without
        a distance rule — the model always sees why it might be firing.
    """
    ctx = list(history)[-(size - 1):] if size > 1 else []
    recent = {(m["sender"], m["text"]) for m in ctx}
    spliced = [{"s": p["sender"], "t": p["text"], "earlier": True}
               for p in pending_texts
               if (p["sender"], p["text"]) not in recent]
    return (spliced
            + [{"s": m["sender"], "t": m["text"]} for m in ctx]
            + [{"s": current["sender"], "t": current["text"]}])
