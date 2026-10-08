"""Every split shape we can think of, with the share computed instead of hand-written.

Asked for by Akash, 2026-10-07, before the split fixes are given to the team: "test this
case as thoroughly as possible with multiple different scenarios ... i dont want any more
issues once i inform the team".

The point of this file is that the right answer is ARITHMETIC. A hand-written expectation
is only as good as the person who wrote it, and the live $200 bug got through a suite full
of hand-written ones. Here the bill, the headcount and the currency are generated, the
expected share is computed from the rules in FIRING_RULE.md 5a/5a1/5a3, and the only thing
the test asserts about the model is that the number on the payment sheet matches.

Three outcomes are reported separately, because they are not equally bad:

  WRONG-HIGH   the sheet shows MORE than the person owes. The worst thing this system can
               do: it invites sending too much money, and the payer has no way to know.
               Any single one of these fails the run.
  WRONG-LOW    the sheet shows less, or is blank when it could have been filled. An
               annoyance - the asker is short and says so. Counted, reported, not fatal
               on its own.
  NO FIRE      the reply never opened a sheet. Reported as coverage, not scored here:
               which phrasings fire is the classifier's business and is measured by
               tests/robustness_suite.py and the ship gate.

    python tests/test_split_matrix.py --url http://127.0.0.1:8971/detect
    python tests/test_split_matrix.py --url ... --cases data/eval/split_scenarios_llm.json
"""
import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

# A currency is (how the asker writes the bill, how the share comes back). The share keeps
# the mark on the side the asker put it, and "Rs"/"rupees" normalise to the symbol.
# `back` of None means "we expect a BLANK amount" - see the euro row.
CURRENCIES = [
    ("${v}",        "${s}",        "USD"),
    ("₹{v}",   "₹{s}",   "INR"),
    # A leading "Rs", "Rs." or "INR" is normalised to the symbol by slot extraction; a
    # TRAILING "rupees" is left where the asker put it.
    ("Rs {v}",      "₹{s}",   "INR"),
    ("Rs. {v}",     "₹{s}",   "INR"),
    ("INR {v}",     "₹{s}",   "INR"),
    ("{v} rupees",  "{s} rupees",  "INR"),
    # KNOWN LIMITATION, 2026-10-07: the amount extractor matches "$", the rupee symbol and
    # the words (dollars/bucks/rupees/rs/inr), but not the euro sign - so a euro bill
    # yields no amount at all and the sheet opens BLANK. Blank is the safe direction and
    # the market is India and the US, so this is recorded rather than fixed; fixing it
    # means touching money extraction globally. Pinned here so the behaviour cannot change
    # unnoticed.
    ("€{v}",   None,          "EUR"),
    ("{v}",         "${s}",        "USD"),      # a bare figure comes back marked "$"
]

# Asks that are unambiguously "this figure is the whole bill, split it".
ASKS = [
    "Dinner was {amt} total, can everyone send me their share?",
    "Dinner was {amt} total, send me your shares",
    "Can everyone send me {amt} for the dinner yesterday?",
    "The trip came to {amt}, send me your shares",
    "{amt} in all for the cab, everyone send me your share",
    "Lunch was {amt} altogether, pay up individual shares",
    "Hotel was {amt} total, everyone please send me the money",
    "{amt} for the groceries, lets split it",
]

# Replies that are KNOWN to fire (measured 2026-10-07, conv classifier >= 0.985). The
# "I'll pay / I'll cover for X" family is deliberately absent: it does not fire at all and
# that gap is recorded in FIRING_RULE.md 5a3, not papered over here.
REPLIES_ONE_SHARE = [
    "sending my share",
    "Okay. Sending my share.",
    "sending",
    "ok sending",
    "sending now",
    "Okay, sending my portion",
    "sure sending",
]
# (reply, how many shares it pays)
REPLIES_MULTI = [
    ("sending for both of us", 2),
    ("ill cover both of us", 2),
    ("sending for the 3 of us", 3),
]


def expected_share(value, n, cur_code):
    """FIRING_RULE 5a1: the share, rounded the way the ruling says, or None for blank."""
    if not n or n < 2:
        return None
    share = value / n
    if share == int(share):
        return str(int(share))
    if cur_code == "INR":
        # Half rounds DOWN - see the comment in app.py's _per_person_share.
        r = math.ceil(share - 0.5)
        return str(r) if r else None
    txt = f"{share:.2f}"
    return txt if float(txt) else None


def scale(share_txt, times):
    """FIRING_RULE 5a3: one share -> the share for `times` people."""
    if share_txt is None or times < 2:
        return share_txt
    v = float(share_txt) * times
    return str(int(v)) if v == int(v) else f"{v:.2f}"


def build():
    """Every scenario as (name, participants, [(sender, text)], expected_amount)."""
    out = []

    # ---- 1a. bill x headcount, in dollars: the arithmetic and the rounding ---------
    for value in (208, 200, 225, 90, 100, 1000, 5000, 37, 10, 5, 0.5, 12345):
        for n in (2, 3, 4, 5, 7, 8, 9, 12):
            amt = f"${_num(value)}"
            want = expected_share(value, n, "USD")
            out.append((f"grid {amt} / {n}", n,
                        [("20", ASKS[0].format(amt=amt)), ("10", "sending my share")],
                        f"${want}" if want is not None else None))

    # ---- 1b. every currency form, on the bills where rounding actually bites -------
    for value, n in ((208, 9), (1000, 7), (50, 3), (240, 4)):
        for form, back, code in CURRENCIES:
            amt = form.format(v=_num(value))
            want = expected_share(value, n, code)
            label = f"currency {amt} / {n}" + ("  [known: no euro support]" if back is None else "")
            out.append((label, n,
                        [("20", ASKS[0].format(amt=amt)), ("10", "sending my share")],
                        back.format(s=want) if (back and want is not None) else None))

    # ---- 2. every ask wording, one bill -------------------------------------------
    for ask in ASKS:
        for n in (3, 9):
            amt = f"${208}"
            want = expected_share(208, n, "USD")
            out.append((f"ask {ask[:26]!r} / {n}", n,
                        [("20", ask.format(amt=amt)), ("10", "sending my share")],
                        f"${want}"))

    # ---- 3. every firing reply wording --------------------------------------------
    for rep in REPLIES_ONE_SHARE:
        out.append((f"reply {rep!r}", 9,
                    [("20", "Dinner was $208 total, send me your shares"), ("10", rep)],
                    "$23.11"))
    for rep, times in REPLIES_MULTI:
        out.append((f"reply {rep!r} pays {times}", 9,
                    [("20", "Dinner was $90 total, send me your shares"), ("10", rep)],
                    scale(expected_share(90, 9, "USD"), times) and
                    f"${scale(expected_share(90, 9, 'USD'), times)}"))

    # ---- 4. the headcount stated IN the message, overriding the roster ------------
    for phrase, n in [("split {n} ways", 4), ("split it {n} ways", 6),
                      ("split between the {n} of us", 3), ("split among the {n} of us", 5),
                      ("{n} ways", 8)]:
        txt = "Dinner was $240, " + phrase.format(n=n)
        out.append((f"stated headcount: {phrase.format(n=n)!r} beats participants=9", 9,
                    [("20", txt), ("10", "sending my share")],
                    f"${expected_share(240, n, 'USD')}"))

    # ---- 5. thread shapes: the fixes from 2026-10-07 ------------------------------
    out += [
        ("thread: clarified headcount", 5,
         [("20", "Dinner was $90 total, send me your shares"),
          ("10", "how many of us are splitting?"),
          ("20", "just the 3 of us who ate"),
          ("30", "ok sending my share")], "$30"),

        ("thread: corrected total", 5,
         [("20", "Dinner was $90 total, send me your shares"),
          ("20", "sorry it was $120 actually"),
          ("10", "ok sending")], "$24"),

        ("thread: corrected total twice", 5,
         [("20", "Dinner was $90 total, send me your shares"),
          ("20", "sorry it was $120 actually"),
          ("20", "actually $150, i forgot the tip"),
          ("10", "ok sending")], "$30"),

        ("thread: dropout then re-split", 5,
         [("20", "Dinner was $90 total, everyone send me your share"),
          ("10", "i cant pay, im broke till friday"),
          ("20", "no worries, rest of you split it 4 ways then"),
          ("30", "sending")], "$22.50"),

        ("thread: refusal does not change the share", 5,
         [("20", "Dinner was $90 total, can everyone send me their share?"),
          ("10", "sorry i cant pay right now"),
          ("30", "sending mine")], "$18"),

        ("thread: question in between", 5,
         [("20", "Dinner was $200 total, send me your shares"),
          ("10", "was that with the drinks?"),
          ("20", "yeah everything"),
          ("30", "sending my share")], "$40"),

        ("thread: headcount asked and answered with a digit", 5,
         [("20", "Dinner was $200 total, send me your shares"),
          ("10", "how many?"),
          ("20", "the 4 of us"),
          ("30", "sending my share")], "$50"),

        # A range is an unknown headcount, so blank - never the smaller number.
        ("thread: range headcount blanks", 5,
         [("20", "Dinner was $90, lets split it between 4-5 of us"),
          ("10", "ok sending")], None),
        ("thread: range with a dash and spaces", 5,
         [("20", "Dinner was $90, split between 4 - 5 of us"),
          ("10", "ok sending")], None),
        ("thread: range written with 'or'", 5,
         [("20", "Dinner was $90, split it between 4 or 5 of us"),
          ("10", "ok sending")], None),

        # No headcount anywhere: blank, never the total.
        ("no participants, no stated count: blank", None,
         [("20", "Dinner was $90 total, send me your shares"),
          ("10", "sending my share")], None),
        ("no participants, uneven: blank", None,
         [("20", "Dinner was $208 total, send me your shares"),
          ("10", "sending my share")], None),

        # A figure already per-person is never divided again.
        ("per-person stated: $25 each", 9,
         [("20", "Can everyone send me $25 each for the dinner yesterday?"),
          ("10", "Okay. Sending my share.")], "$25"),
        ("per-person stated: thats 1000 each", 5,
         [("20", "the trip was 5000, thats 1000 each"),
          ("10", "sending")], "$1000"),
        ("per-person stated: 50 per person", 4,
         [("20", "dinner was 200, 50 per person"),
          ("10", "sending")], "$50"),
        ("per-person stated: 20 a piece", 4,
         [("20", "cab was 80, 20 a piece"),
          ("10", "sending")], "$20"),

        # Not splits at all - the figure must survive untouched.
        ("not a split: one-to-one ask", 9,
         [("20", "can you send me $200 for dinner"), ("10", "Sure")], "$200"),
        ("not a split: a debt", 9,
         [("20", "you owe me $200"), ("10", "sure sending")], "$200"),
        ("not a split: lend me", 9,
         [("20", "can you lend me $500 till friday"), ("10", "sending")], "$500"),

        # The payer types their own figure: never divided, never blanked.
        ("typed amount survives", 9,
         [("20", "Dinner was $208 total, send me your shares"),
          ("10", "sending you $25 now")], "$25"),
        ("typed amount survives a blank case", None,
         [("20", "Dinner was $208 total, send me your shares"),
          ("10", "sending you $25 now")], "$25"),

        # A headcount is not a payment: the digit in "the 3 of us" must not become $3.
        ("headcount digit is not the amount", 5,
         [("20", "Dinner was $90 total, send me your shares"),
          ("10", "sending for the 3 of us")], "$54"),
        ("headcount digit alone does not fill the sheet", 5,
         [("20", "Dinner was $200 total, send me your shares"),
          ("10", "sending for the 4 of us")], "$160"),

        # ---- the four markers the generated threads needed, 2026-10-07 -------------
        # "so we split 120?" - a figure after "split", not a headcount or "it".
        ("marker: split followed by a figure", 5,
         [("20", "The bill is 90 rupees, but i dont know how many of us are paying"),
          ("30", "Actually the total was 120 rupees"),
          ("40", "So we split 120? How many people?"),
          ("10", "Okay, sending my share")], "24 rupees"),

        ("marker: split evenly", 5,
         [("20", "Cabin rental was $150 total."),
          ("30", "Dude, we agreed to split evenly."),
          ("40", "Sending my share.")], "$30"),

        ("marker: chipping in", 5,
         [("20", "The group gift cost 1800 rupees."),
          ("30", "How many of us are chipping in?"),
          ("40", "Me too.")], "360 rupees"),

        # "my share" is not a split marker alone - it needs the figure to be a group total.
        ("marker: my share against a group total", 3,
         [("20", "The pizza was $60 total, can you guys pay me back?"),
          ("10", "Okay, sending my share.")], "$20"),

        ("marker: my share, no headcount, blanks", None,
         [("20", "Got the coffee run earlier, total was 9.50"),
          ("30", "K, sending my share now")], None),

        # ...and it must NOT turn an ordinary one-to-one ask into a split. This is the
        # regression the _MY_SHARE rule risks, so it is pinned here.
        ("marker: a one-to-one ask is not a split even with 'my share'", 9,
         [("20", "can you send me $200 for dinner"),
          ("10", "sending my share")], "$200"),
        ("marker: a debt is not a split even with 'my share'", 9,
         [("20", "you owe me $200"),
          ("10", "sending my share")], "$200"),

        # A per-person figure stated LATER is the share - not the total we were holding.
        ("per-person stated after a correction", 6,
         [("20", "The concert tickets were $260 total. Split 6 ways = $43.33 each."),
          ("30", "Wait, I thought they were $240?"),
          ("20", "Let me check... you're right, it was $240. So $40 each."),
          ("40", "Fine, sending my share now.")], "$40"),

        # A DM is two people by definition.
        # A dm_<lo>_<hi> room is two people by definition, so no participants are needed.
        # A 5th field of "dm" puts the scenario in a DM room; run() builds a fresh id each
        # time, because a fixed one would carry server state over from the previous run.
        ("dm: split between us", None,
         [("20", "dinner was 2000, split between us"), ("10", "sending")], "$1000", "dm"),
        ("dm: uneven splits in two", None,
         [("20", "dinner was 2001, lets split it"), ("10", "sending")], "$1000.50", "dm"),
        ("dm: a one-to-one ask is not halved", None,
         [("20", "can you send me $200 for dinner"), ("10", "Sure")], "$200", "dm"),
    ]

    # ---- 6. two independent bills in one room ------------------------------------
    # `want` may be a LIST: the amount expected on the 1st fire, the 2nd, and so on. A
    # second bill in the same room must divide on its own and must not inherit the first.
    out.append(("two bills in one room, each divides on its own", 4,
                [("20", "Dinner was $200 total, send me your shares"),
                 ("10", "sending my share"),
                 ("20", "Also the cab was $80 total, send me your shares"),
                 ("30", "sending my share")], ["$50", "$20"]))
    out.append(("a second bill after a blank one still divides", 4,
                [("20", "Dinner was $90 total, lets split it"),
                 ("10", "sending my share"),
                 ("20", "Cab was $200 total, send me your shares"),
                 ("30", "sending my share")], ["$22.50", "$50"]))

    return out


def _num(v):
    return str(int(v)) if float(v) == int(v) else f"{v:g}"


def _amt_value(s):
    """The numeric part of an amount string, or None."""
    if s in (None, "", []):
        return None
    digits = "".join(c for c in str(s) if c.isdigit() or c == ".")
    try:
        return float(digits)
    except ValueError:
        return None


def run(url, cases, verbose=False):
    tag = int(time.time())
    s = requests.Session()
    high, low, nofire, ok = [], [], [], 0
    t0 = time.time()
    for ci, case in enumerate(cases):
        name, parts, seq, want = case[:4]
        # The room-id matcher bounds each side to 4 digits, so the run tag is folded down
        # rather than used whole.
        room = (f"dm_{tag % 9000 + 1000}_{ci}" if len(case) > 4 and case[4] == "dm"
                else f"group_mx{tag}_{ci}")
        base = {"room_id": room}
        if parts:
            base["participants"] = parts
        fired_any = False
        wants = list(want) if isinstance(want, list) else None
        fire_no = 0
        for j, (sender, text) in enumerate(seq):
            d = s.post(url, json={**base, "text": text, "sender": sender,
                                  "message_id": f"{tag}_{ci}_{j}"}, timeout=90).json()
            got_intents = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
            if not got_intents:
                continue
            fired_any = True
            got = (d.get("slots") or {}).get("amount")
            if wants is not None:
                exp = wants[fire_no] if fire_no < len(wants) else None
                fire_no += 1
            else:
                exp = want
            if exp is None and got is None:
                continue
            if got == exp:
                continue
            gv, wv = _amt_value(got), _amt_value(exp)
            # Blank expected and a figure shown, or more than owed: the dangerous way.
            if gv is not None and (wv is None or gv > wv + 1e-9):
                high.append((name, text, exp, got))
            else:
                low.append((name, text, exp, got))
        if not fired_any:
            nofire.append(name)
        elif not any(name == h[0] for h in high) and not any(name == l[0] for l in low):
            ok += 1
        if verbose and (ci + 1) % 25 == 0:
            r = (ci + 1) / max(time.time() - t0, 1e-9)
            print(f"    ... {ci+1} of {len(cases)} done  ({r:.1f}/s)", flush=True)

    # TOTAL is what tests/run_all.py scores on, and it beats every other pattern it looks
    # for. Without it run_all matched "N/M scenarios" in the PROGRESS lines and kept the
    # last one, reporting a clean 194/195 run as 175/195.
    print(f"\n  TOTAL {ok}/{len(cases)} correct")
    print(f"  {len(cases)} scenarios")
    print(f"    {ok:>5}  correct")
    print(f"    {len(nofire):>5}  never fired (coverage, not scored here)")
    print(f"    {len(low):>5}  WRONG-LOW   (short or blank - annoying)")
    print(f"    {len(high):>5}  WRONG-HIGH  (more than owed - must be zero)")

    for label, rows in (("WRONG-HIGH", high), ("WRONG-LOW", low)):
        if not rows:
            continue
        print(f"\n  {label}:")
        seen = set()
        for name, text, want, got in rows:
            k = (name, got)
            if k in seen:
                continue
            seen.add(k)
            print(f"    {name}\n        {text!r}\n        want {want!r}, got {got!r}")
    if nofire:
        print(f"\n  never fired ({len(nofire)}):")
        for n in nofire[:25]:
            print(f"    {n}")
        if len(nofire) > 25:
            print(f"    ... and {len(nofire)-25} more")
    return 1 if high else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/detect")
    ap.add_argument("--cases", help="extra scenarios as JSON (same shape as build())")
    ap.add_argument("--limit", type=int, help="run only the first N scenarios")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cases = build()
    if a.cases:
        extra = json.loads(Path(a.cases).read_text(encoding="utf-8"))
        cases += [(c["name"], c["participants"],
                   [(m["sender"], m["text"]) for m in c["messages"]], c["expected"])
                  for c in extra]
        print(f"  + {len(extra)} scenarios from {a.cases}")
    if a.limit:
        cases = cases[:a.limit]
    return run(a.url, cases, verbose=not a.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
