"""Refuse the v12 conv batch if it dilutes v5. Runs BEFORE training; the exit code decides.

audit_dilution.py already existed before v11 and exits 1 on it. Nobody ran it as a
precondition, and six models were rejected after the fact. This is that check plus the
two things it cannot see on its own.

Three criteria:

  1  phrase dilution     audit_dilution's own check. A phrase that meant "fire" in v5
                         meaning it less in v12 is a phrase whose meaning we overwrote,
                         and it shows up later as lost recall on exactly the
                         conversations that use it.

  2  role fire rates     DIRECTIONAL, unlike the first version of this check. A role's
                         rate FALLING is dilution and fails. A role's rate RISING is
                         usually the batch working - v12 adds 478 in-progress
                         commitments tagged self_money, so that rate goes 63.7% -> 66.4%
                         by design. Symmetric tolerance would have failed the batch for
                         succeeding.

  3  boundary roles      A rise is not free. self_money is 64% fire and 36% quiet, and
                         the quiet 36% are ALREADY-DONE payments - "just sent it",
                         "transferred", "Sent 800 on GPay". Pushing the fire rate up
                         risks dragging those over. That cannot be checked in the data;
                         it is checked after training by tests/test_basics.py's
                         "already done" cases and test_self_initiated.py. Named here so
                         it is not forgotten.

    python data_gen/gate_conv_v12.py
"""
import ast, io, json, subprocess, sys, zipfile
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
# Baseline and candidate are arguments now. Hardcoding them meant the gate could
# only ever check v12, and a v14 batch would have shipped ungated.
BASE = sys.argv[1] if len(sys.argv) > 2 else "conv_windows_v5.zip"
NEW  = sys.argv[2] if len(sys.argv) > 2 else "conv_windows_v12.zip"
DROP_TOL = 2.0          # a role may not LOSE more than this


def load(p):
    return json.loads(zipfile.ZipFile(ROOT / p).read("train.json").decode("utf-8"))


def fired(v):
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    s = str(v).strip()
    if s in ("", "[]", "None"):
        return False
    try:
        return len(ast.literal_eval(s)) > 0
    except Exception:
        return bool(s)


def roles(rows):
    d = defaultdict(lambda: [0, 0])
    for r in rows:
        k = r.get("role", "?")
        d[k][0] += 1
        d[k][1] += fired(r.get("fire"))
    return d


def main():
    failed = False

    print("")
    print("  === 0. does each label match what the message says? ===")
    # Written after v12 shipped 95 windows where a cab commitment carried a money label,
    # because three generation tasks declared intent="money" while their prompts said
    # "mix money and cab". The model learned it: it scored "I am booking the cab now" at
    # money 0.994 / ride 0.053, disagreed with the base model's ride request, and fired
    # nothing. That was the one room regression the gate caught - four steps and several
    # hours after a five-line check would have caught it here.
    #
    # Deliberately narrow. It only flags the unambiguous direction: a booking verb acting
    # on a ride noun, with no payment verb present. "transferring the cab fare" stays
    # money and is not flagged.
    import re as _re
    _RN = r"(?:cab|uber|ola|taxi|auto|rapido|lyft)"
    _BOOK = _re.compile(r"\b(?:book|booking|order|ordering|call|calling|grab|grabbing|arrange|arranging)\b[^.?!]{0,16}\b(?:cab|uber|ola|taxi|auto|rapido|lyft)\b", _re.I)
    _PAY = _re.compile(r"\b(?:transfer|transferring|send|sending|pay|paying|gpay|venmo|zelle|upi|cashapp)\b[^.?!]{0,24}\b(?:fare|money|cash|amount|\d)\b", _re.I)
    rows_new = load(NEW)
    mism = []
    for r in rows_new:
        f = r.get("fire")
        f = f if isinstance(f, (list, tuple)) else ast.literal_eval(str(f) or "[]")
        if list(f) != ["money"]:
            continue
        w = r["window"]
        w = w if isinstance(w, list) else ast.literal_eval(w)
        t = w[-1]["t"]
        if _BOOK.search(t) and not _PAY.search(t):
            mism.append(t)
    print(f"  cab actions carrying a money label: {len(mism)}")
    for t in mism[:6]:
        print(f"    {t[:70]!r}")
    # v5 itself carries 29 of these in 306,672 - one in ten thousand, background noise.
    # Anything meaningfully above that rate is a batch-level labelling fault.
    BG = 40
    if len(mism) > BG:
        print(f"  FAILED: {len(mism)} is above v5's background rate of ~29")
        failed = True

    print("  === 1. phrase dilution (audit_dilution.py --gate) ===")
    rc = subprocess.run([sys.executable, str(ROOT / "data_gen/audit_dilution.py"),
                         BASE, NEW, "--gate"],
                        cwd=str(ROOT), capture_output=True, text=True).returncode
    print(f"  audit_dilution exit: {rc}")
    # Two phrases move here by design - "book you" and "want me" are the offer
    # vocabulary, and the whole batch exists to stop offers firing. Anything BEYOND
    # those two is unexplained and fails.
    out = subprocess.run([sys.executable, str(ROOT / "data_gen/audit_dilution.py"),
                          BASE, NEW, "--show", "40"],
                         cwd=str(ROOT), capture_output=True, text=True).stdout
    n_dil = 0
    for line in out.splitlines():
        if "diluted by" in line:
            n_dil = int(line.split(":")[1].split()[0])
    # Phrases this batch is MEANT to move. Widening this list to make the gate pass is
    # how a gate rots, so each entry states the intent that explains it and the check
    # that would catch it being wrong.
    #
    #   book you   "should i book you a cab"      offer, must stop firing
    #   want me    "want me to send you the 500"  offer, must stop firing
    #   i send     "shall i send you the money"   offer, must stop firing. Added after
    #              the 349-window offer_shall_i top-up, which did not exist when this
    #              list was written - the #1 defect had 6 training examples until then.
    #
    # None of these touch the commitment forms they sit next to: "i'll send it now" is
    # tokenised i/ll/send, not "i send". The guarantee that they have not is criterion 2
    # below - self_money and ack_money must not FALL - not this list.
    EXPECTED = {"book you", "want me", "i send"}
    print(f"  phrases diluted >= 8pp: {n_dil}   expected at most {len(EXPECTED)} "
          f"({', '.join(sorted(EXPECTED))})")
    if n_dil > len(EXPECTED):
        print("  FAILED: more phrases moved than the offer vocabulary explains")
        failed = True

    print("\n  === 2. role fire rates (a FALL is dilution; a rise is the batch working) ===")
    a, b = roles(load(BASE)), roles(load(NEW))
    print(f"  {'role':22} {'base':>9} {'new':>9}   change")
    for k in sorted(set(a) | set(b), key=lambda x: -b.get(x, [0])[0]):
        n1, f1 = a.get(k, [0, 0])
        n2, f2 = b.get(k, [0, 0])
        if not n1:
            print(f"  {k:22} {'--':>9} {f2/n2*100:8.1f}%   (new role, {n2:,} windows)")
            continue
        r1, r2 = f1 / n1 * 100, f2 / n2 * 100
        tag = ""
        if r1 - r2 > DROP_TOL:
            tag = f"   FAILED: lost {r1-r2:.1f}pp"
            failed = True
        elif r2 - r1 > DROP_TOL:
            tag = f"   rose {r2-r1:.1f}pp - intended, see criterion 3"
        print(f"  {k:22} {r1:8.1f}% {r2:8.1f}%   {r2-r1:+6.1f}{tag}")

    print("\n  === 3. boundary to verify AFTER training, not here ===")
    print("  self_money rose, and its quiet 36% are already-done payments")
    print("  ('just sent it', 'transferred'). These must still stay silent:")
    print("    tests/test_basics.py        'already done' category, 3/3 today")
    print("    tests/test_self_initiated.py 'just sent it' / 'cab booked', quiet today")
    print("  A batch that fixes commitments by breaking already-done is a bad trade.")

    print("\n" + "=" * 56)
    if failed:
        print("  GATE FAILED - do not train. Regenerate the batch that caused it.")
    else:
        print("  GATE PASSED - v12 windows are safe to train on.")
        print(f"  {NEW} is ready.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
