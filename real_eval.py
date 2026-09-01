"""One evaluation, on real conversations, with the label source stated for every number.

Our synthetic test set is out of distribution and we can now prove it: training
conversations run a median of 6 turns, the eval 24, real chats 15; 32% of real rooms need
more than one prompt and the eval contains not a single conversation that does. So this
measures the three things that can be measured on real data without inventing labels.

  RECALL      dogfood logs. A request is identifiable, and so is somebody agreeing to it.
              Counted only where a request was actually accepted - if nobody agreed,
              silence is correct and is not a miss.

  FALSE FIRE  real WhatsApp chats. Ordinary conversation between real people. Every
              prompt is listed so it can be read rather than trusted; the rate is the
              number that matters.

  CASES       the failures confirmed today, labelled by Akash on 2026-08-24, not by me
              and not by an LLM.

Run against two servers to compare models. Nothing here is synthetic.
"""
import argparse, json, re, sys, io, time
from pathlib import Path
from collections import Counter

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data_gen"))
# wa_study installs its own utf-8 stdout wrapper at import. Import it BEFORE ours is
# set up - the other order closed our buffer mid-run and lost a completed pass.
import wa_study                                    # noqa: E402

sys.stdout = io.TextIOWrapper(open(sys.__stdout__.fileno(), "wb", closefd=False),
                              encoding="utf-8", errors="replace", line_buffering=True)
MR = ("money", "ride")

REQ = re.compile(
    r"\b(?:can|could|will|would|pls|please)\b.{0,30}\b(?:book|send|pay|transfer|cab|uber|ola|ride)\b"
    r"|\b(?:book|send|pay|transfer)\s+(?:me|us)\b"
    r"|\bbook\s+a\s+(?:cab|uber|ola|ride)\b"
    r"|\bneed\b.{0,20}\b(?:cab|uber|ride)\b|\bi\s+need\s+\d+", re.I)
ACK = re.compile(
    r"^(?:ok|okay|k|kk|sure|yeah|yea|ya|yep|yup|yes|fine|of course|why not|will do|done|"
    r"sending|booking|on it|no problem|np|cool|sure thing|\U0001F44D)\b", re.I)

# Labelled by Akash, 2026-08-24. Each is a failure confirmed on real traffic.
CASES = [
    ("second request in a room gets a prompt", "ride", True, [
        ("11", "Hi"), ("11", "Can you book cab for me"), ("41", "Sure"),
        ("26", "Hi"), ("41", "Hi can you book a cab for me from majestic"),
        ("26", "Sure"), ("26", "can you book a cab for me from majestic to jp nagar"),
        ("41", "Cool"), ("41", "Sure")]),
    ("repeat agreement by the same person is an echo", "ride", False, [
        ("11", "Can you book cab for me"), ("41", "Sure"), ("41", "Yes"),
        ("41", "Of course"), ("41", "Sure")]),
    ("greeting after a request", "money", False, [
        ("26", "Hi"), ("26", "How are you"), ("26", "Can you send me 10$"),
        ("53", "Hi")]),
    ("discussing the app", "ride", False, [
        ("20", "How long shall we keep the request open?"),
        ("10", "May be forever for now"),
        ("20", "Let's say\nB: can you book me a cab to airport\nA: okay"),
        ("20", "So no limit and as long as it's in the window"), ("10", "Yes")]),
    ("self-initiated, in progress", "money", True, [
        ("4", "Hmm"), ("4", "I'll need to book a cab ride back home"),
        ("4", "I'm sending you one dollar")]),
    ("a need never fires", "ride", False, [
        ("4", "Hmm"), ("4", "I'll need to book a cab ride back home")]),
    ("unprompted offer waits", "ride", False, [
        ("11", "hey"), ("41", "let me book a cab")]),
    ("offer answering a request fires", "ride", True, [
        ("11", "can you book me a cab to the airport"), ("41", "let me book a cab")]),
]


def post(url, text, room, sender, mid):
    try:
        return requests.post(url, timeout=60, json={
            "text": text, "room_id": room, "sender": str(sender),
            "message_id": mid}).json()
    except Exception:
        return {}


def recall(url, tag):
    tot = hit = 0
    misses = []
    for log in sorted((ROOT / "data/eval").glob("dogfood_*.jsonl")):
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
        by = {}
        for r in rows:
            by.setdefault(r["room"], []).append(r)
        for room, rs in by.items():
            fired_at = set()
            for i, r in enumerate(rs):
                d = post(url, r["text"], f"re_{tag}_{room}", r["sender"], f"{tag}_{room}_{i}")
                if [x for x in (d.get("intents") or []) if x in MR]:
                    fired_at.add(i)
            for i, r in enumerate(rs):
                if not REQ.search(r["text"]):
                    continue
                after = rs[i + 1:i + 7]
                accepted = any(x["sender"] != r["sender"] and ACK.match(x["text"].strip())
                               for x in after)
                if not accepted:
                    continue                      # nobody agreed - silence is correct
                tot += 1
                if any(i + 1 + j in fired_at for j in range(len(after))):
                    hit += 1
                else:
                    misses.append((room, r["text"][:44],
                                   [x["text"][:20] for x in after[:3]]))
    return hit, tot, misses


def falsefire(url, tag, cap):
    chats = ["WhatsApp Chat - Sunny", "WhatsApp Chat - Gowtham CEO",
             "WhatsApp Chat - Team DFoE", "WhatsApp Chat - Windsor Telugu Association",
             "WhatsApp Chat - RIDES WINDSOR \U0001F697 GTA"]
    n = f = 0
    hits = []
    for c in chats:
        p = Path(f"C:/Users/akash/Downloads/{c}/_chat.txt")
        if not p.exists():
            continue
        msgs, _ = wa_study.parse(p)
        msgs = [m for m in msgs if m["text"].strip()][:cap]
        for i, m in enumerate(msgs):
            d = post(url, m["text"], f"ff_{tag}_{c[:12]}_{i//25}", m["who"], f"{tag}_{i}")
            got = [x for x in (d.get("intents") or []) if x in MR]
            n += 1
            if got:
                f += 1
                hits.append((c[15:30], got, m["text"][:52]))
    return f, n, hits


def cases(url, tag):
    ok = 0
    detail = []
    for ci, (name, intent, want, turns) in enumerate(CASES):
        room = f"cs_{tag}_{ci}"
        fired = False
        for ti, (spk, txt) in enumerate(turns):
            d = post(url, txt, room, spk, f"{tag}_{ci}_{ti}")
            got = [x for x in (d.get("intents") or []) if x in MR]
            if ti == len(turns) - 1:
                fired = intent in got
        good = fired == want
        ok += good
        detail.append((name, want, fired, good))
    return ok, len(CASES), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cap", type=int, default=1200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tag = f"{a.label}_{int(time.time())}"

    print(f"\n  ===== {a.label} =====")
    ok, n, detail = cases(a.url, tag)
    print(f"\n  CONFIRMED CASES (labelled by Akash)   {ok}/{n}")
    for name, want, got, good in detail:
        print(f"    {'ok  ' if good else 'FAIL'}  {name:44} want={'prompt' if want else 'quiet':6} got={'prompt' if got else 'quiet'}")

    hit, tot, misses = recall(a.url, tag)
    print(f"\n  RECALL on real dogfood traffic         {hit}/{tot} = {hit/max(tot,1)*100:.1f}%")
    for room, txt, after in misses[:8]:
        print(f"    missed  [{room}] {txt!r} -> {after}")

    f, N, hits = falsefire(a.url, tag, a.cap)
    # Save first. A finished measurement must not be destroyed by a print.
    if a.out:
        (ROOT / a.out).write_text(json.dumps({
            "label": a.label, "cases": [ok, n], "recall": [hit, tot],
            "false": [f, N], "false_hits": hits, "misses": misses}, indent=1,
            ensure_ascii=False), encoding="utf-8")
    print(f"\n  FALSE PROMPTS on real WhatsApp chats   {f} in {N} messages = {f/max(N,1)*1000:.1f} per 1000")
    for c, got, txt in hits[:10]:
        print(f"    {str(got):10} [{c}] {txt!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
