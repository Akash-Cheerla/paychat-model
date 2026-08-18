"""Pull slot phrasings and hard negatives out of two publicly licensed dialogue corpora.

Why bother, when we already generate everything: five training rounds have moved the
behavioural numbers 1-2% each. The generator has saturated — every new batch is drawn
from the same distribution the model already fits. What is missing is language nobody
on this team wrote.

Two corpora, both cleanly licensed (unlike the WhatsApp/Telegram research dumps, which
are real people's private messages and not something a payments product should train on):

  * Schema-Guided Dialogue (CC BY-SA 4.0) — Payment_1, RideSharing_1/2, Banks_1/2
  * Taskmaster-1 (Google) — 1,098 uber/lyft self-dialogs, plus 6,610 in other domains

Neither can be used as conversations: both are a human talking to an assistant, and our
whole problem is two humans negotiating until one commits. So we take two things that
survive the register change:

  PHRASINGS — the exact spans crowdworkers used for a destination, a pickup, an amount,
  a recipient. These are span-annotated, so they come out clean, and they are the part
  of the input our slot extraction actually reads.

  HARD NEGATIVES — the non-money, non-ride Taskmaster domains. Ordering a pizza is full
  of "I'll pay when it gets here", "can you send it to my address", "book me a table";
  auto repair is full of appointments and money that is never a P2P transfer. Real
  sentences, guaranteed no-fire, and confusable in exactly the way our false positives
  are. The label is free, which is the whole appeal — no rule ambiguity to argue about.

Output: data/external/phrasings.json and data/external/hard_negatives.json.
Neither is training data on its own; gen_conversations.py consumes the phrasings, and
the negatives go in as no-fire windows.
"""
import json, glob, re, sys, io
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"

# SGD slot -> our slot. Only slots we actually fill; SGD has many we do not care about.
SGD_SLOTS = {
    "destination": "destination", "to_location": "destination",
    "destination_city": "destination", "to_city": "destination",
    "from_location": "pickup", "pickup_location": "pickup", "from_city": "pickup",
    "ride_type": None, "number_of_riders": None,
    "amount": "amount", "transfer_amount": "amount",
    "receiver": "recipient", "transfer_time": None,
}
# Taskmaster annotation -> our slot.
TM_SLOTS = {
    "uber_lyft.location.to": "destination",
    "uber_lyft.location.from": "pickup",
    "uber_lyft.type.ride": None,
    "uber_lyft.time.pickup": "time",
}
NON_RIDE_MONEY_TM = ("pizza", "coffee", "movie", "restaurant", "auto")
_WS = re.compile(r"\s+")


def clean(s):
    s = _WS.sub(" ", (s or "").strip())
    return s if 1 <= len(s) <= 60 else ""


def from_sgd():
    """Span-annotated slot values from the Payment / RideSharing / Banks services."""
    out = defaultdict(Counter)
    files = sorted(glob.glob(str(EXT / "sgd" / "*" / "dialogues_*.json")))
    if not files:
        print("  sgd/ not found — skipping"); return out
    for f in files:
        for d in json.load(open(f, encoding="utf-8")):
            if not any(s.split("_")[0] in ("Payment", "RideSharing", "Banks")
                       for s in d["services"]):
                continue
            for t in d["turns"]:
                if t["speaker"] != "USER":
                    continue
                for fr in t.get("frames", []):
                    for sl in fr.get("slots", []):
                        ours = SGD_SLOTS.get(sl["slot"])
                        if not ours:
                            continue
                        v = clean(t["utterance"][sl["start"]:sl["exclusive_end"]])
                        if v:
                            out[ours][v] += 1
    return out


def from_taskmaster():
    """Slot spans from the uber/lyft dialogs, and whole utterances from the rest."""
    out, negs = defaultdict(Counter), []
    p = EXT / "taskmaster" / "TM-1-2019" / "self-dialogs.json"
    if not p.exists():
        print("  taskmaster/ not found — skipping"); return out, negs
    for conv in json.load(open(p, encoding="utf-8")):
        dom = str(conv.get("instruction_id", "")).split("-")[0].lower()
        if dom == "uber":
            for u in conv["utterances"]:
                for seg in u.get("segments", []):
                    for ann in seg.get("annotations", []):
                        ours = TM_SLOTS.get(ann["name"])
                        if ours:
                            v = clean(seg.get("text"))
                            if v:
                                out[ours][v] += 1
        elif dom in NON_RIDE_MONEY_TM:
            # USER turns only. The ASSISTANT half is a service agent, a voice that does
            # not exist in a group chat, and training on it would teach a register the
            # product never sees.
            turns = [clean(u["text"]) for u in conv["utterances"]
                     if u["speaker"] == "USER"]
            turns = [t for t in turns if t]
            if len(turns) >= 3:
                negs.append({"domain": dom, "turns": turns,
                             "source": "taskmaster-1", "fire": []})
    return out, negs


def main():
    sgd = from_sgd()
    tm, negs = from_taskmaster()

    merged = defaultdict(Counter)
    for src in (sgd, tm):
        for k, c in src.items():
            merged[k].update(c)

    phr = {k: [w for w, _ in c.most_common()] for k, c in merged.items()}
    (EXT / "phrasings.json").write_text(
        json.dumps({"source": "SGD (CC BY-SA 4.0) + Taskmaster-1",
                    "slots": phr}, ensure_ascii=False, indent=1), encoding="utf-8")
    (EXT / "hard_negatives.json").write_text(
        json.dumps(negs, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n  phrasings.json")
    for k, c in sorted(merged.items()):
        uniq = len(c)
        print(f"    {k:12} {uniq:>6} unique   e.g. " +
              ", ".join(repr(w) for w, _ in c.most_common(3)))
    print(f"\n  hard_negatives.json   {len(negs)} conversations, "
          f"{sum(len(n['turns']) for n in negs)} turns, all no-fire")
    print("    by domain:", dict(Counter(n["domain"] for n in negs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
