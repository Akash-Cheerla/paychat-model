"""Run the DeepSeek-generated split threads against invariants, not against its labels.

Akash asked (2026-10-07) for as many scenarios as possible before the team is told, and
suggested pulling some from DeepSeek. The threads it writes are genuinely good - messy,
realistic, full of clarifications and refusals. Its LABELS are not: spot-checking the
first batch found it marking a headcount as unknown in a thread that plainly settles one
("Let's just do 6 for now"), marking a stated per-person figure as unknown, and leaving
total_value at 85 in a thread that corrects the bill to 115. Scoring against those would
have manufactured failures and hidden real ones.

So nothing here trusts a label. Every check below is arithmetic on the bill alone:

  IMPOSSIBLE   the amount is MORE than half the bill. A split has at least two people in
               it, so one person's share cannot exceed total/2 - whatever the headcount
               turns out to be. No label needed, and no judgement call. These are bugs.
  IS THE TOTAL the amount equals the bill exactly. The specific failure that put $200 on
               every member's sheet on 2026-10-07, called out separately because it is the
               one we keep rediscovering.
  ODD          the amount is not the bill over any whole number of people, and is not a
               figure stated in the thread. Not necessarily wrong - printed for review.
  FINE         blank, or the bill over some whole number of people, or a figure the thread
               states outright.

    python tests/test_split_llm_scenarios.py --url http://127.0.0.1:8971/detect
"""
import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def value_of(s):
    if s in (None, "", []):
        return None
    m = NUM.search(str(s))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def figures_in(messages):
    """Every number anyone typed in the thread, as floats."""
    out = set()
    for m in messages:
        for f in NUM.findall(m["text"] or ""):
            v = value_of(f)
            if v is not None:
                out.add(round(v, 2))
    return out


# A figure the thread states as a PER-PERSON amount: showing it in full is then right,
# and must not be mistaken for showing the total.
PER_PERSON = re.compile(
    r"(?:[₹$€]\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:rs|rupees|dollars|bucks)?\s*"
    r"(?:each|per\s+(?:head|person)|a\s?piece|apiece)\b"
    r"|(?:each|per\s+(?:head|person))\s*(?:is|are|=|:)?\s*"
    r"(?:[₹$€]\s*)?(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)


def per_person_figures(messages):
    """Every figure the thread calls a per-person amount."""
    out = set()
    for m in messages:
        for a, b in PER_PERSON.findall(m["text"] or ""):
            v = value_of(a or b)
            if v is not None:
                out.add(round(v, 2))
    return out


def classify(shown, total, thread_figures, payer_shares, pp=()):
    """Label-free verdict on one prompt."""
    v = value_of(shown)
    if v is None:
        return "FINE", "blank"
    # The thread itself called this figure a per-person amount, so it is not a total.
    if round(v, 2) in pp:
        return "FINE", "a per-person figure the thread states"
    cap = total * payer_shares / 2.0
    if v > cap + 0.011:
        why = (f"{v} is more than half of {total}"
               if payer_shares == 1 else
               f"{v} is more than {payer_shares} x half of {total}")
        return ("IS THE TOTAL", why) if abs(v - total) < 0.011 else ("IMPOSSIBLE", why)
    for n in range(2, 51):
        share = total * payer_shares / n
        if abs(v - share) < 0.011 or abs(v - round(share, 2)) < 0.011 or abs(v - int(share)) < 0.011:
            return "FINE", f"{total}/{n}" + (f" x{payer_shares}" if payer_shares > 1 else "")
    if round(v, 2) in thread_figures:
        return "FINE", "a figure stated in the thread"
    return "ODD", f"not {total} over any whole headcount, and not stated in the thread"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/detect")
    ap.add_argument("--cases", default="data/eval/split_scenarios_llm.json")
    ap.add_argument("--show-odd", action="store_true", help="print every ODD prompt")
    a = ap.parse_args()

    cases = json.loads((ROOT / a.cases).read_text(encoding="utf-8"))
    tag = int(time.time())
    s = requests.Session()
    counts = {"FINE": 0, "ODD": 0, "IMPOSSIBLE": 0, "IS THE TOTAL": 0}
    bad, odd, nofire = [], [], 0
    t0 = time.time()

    for ci, c in enumerate(cases):
        f = c.get("_facts") or {}
        # total_text is validated to appear verbatim in a message; total_value is a
        # label and was wrong at least once ("1800 rupees" carrying total_value 180), so
        # the text wins and the label is only a fallback.
        total = value_of(f.get("total_text")) or float(f.get("total_value") or 0)
        shares = int(f.get("payer_shares") or 1)
        if total <= 0:
            continue
        figs = figures_in(c["messages"])
        pp = per_person_figures(c["messages"])
        room = f"group_llm{tag}_{ci}"
        body = {"room_id": room}
        if c.get("participants"):
            body["participants"] = c["participants"]
        fired = False
        for j, m in enumerate(c["messages"]):
            d = s.post(a.url, json={**body, "text": m["text"], "sender": str(m["sender"]),
                                    "message_id": f"{tag}_{ci}_{j}"}, timeout=90).json()
            if not [x for x in (d.get("intents") or []) if x in ("money", "ride")]:
                continue
            fired = True
            shown = (d.get("slots") or {}).get("amount")
            verdict, why = classify(shown, total, figs, shares, pp)
            counts[verdict] += 1
            if verdict in ("IMPOSSIBLE", "IS THE TOTAL"):
                bad.append((verdict, c["name"], f.get("total_text"), m["text"], shown, why))
            elif verdict == "ODD":
                odd.append((c["name"], f.get("total_text"), m["text"], shown, why))
        if not fired:
            nofire += 1
        if (ci + 1) % 25 == 0:
            print(f"    ... {ci+1}/{len(cases)}  ({(ci+1)/max(time.time()-t0,1e-9):.1f}/s)",
                  flush=True)

    # See the note in tests/test_split_matrix.py about why TOTAL is printed.
    print(f"\n  TOTAL {counts['FINE']}/{sum(counts.values())} prompts defensible")
    print(f"  {len(cases)} threads, {sum(counts.values())} prompts opened"
          f", {nofire} threads never fired")
    for k in ("FINE", "ODD", "IS THE TOTAL", "IMPOSSIBLE"):
        print(f"    {counts[k]:>5}  {k}")

    if bad:
        print(f"\n  {len(bad)} prompt(s) that cannot be right:")
        for verdict, name, tt, text, shown, why in bad:
            print(f"    [{verdict}] {name}")
            print(f"        bill {tt!r}, reply {text[:66]!r}")
            print(f"        showed {shown!r} - {why}")
    if odd and (a.show_odd or len(odd) <= 12):
        print(f"\n  {len(odd)} for review (not necessarily wrong):")
        for name, tt, text, shown, why in odd[:20]:
            print(f"    {name}\n        bill {tt!r}, reply {text[:66]!r} -> {shown!r}\n        {why}")
    elif odd:
        print(f"\n  {len(odd)} for review - rerun with --show-odd to list them")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
