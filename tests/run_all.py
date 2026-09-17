"""Run every battery in tests/ against one or more live servers and tabulate.

run_gate.sh checks four criteria. There are eighteen batteries in this directory, and
most of them were written because something specific broke in production - the group15
miss, the Gowtham ride misfire, the echo window, the screenshots. A model that clears the
gate can still regress one of those, and nobody would know until it shipped.

This does not replace the gate. The gate decides; this shows the whole surface, so a
regression outside the four criteria is visible before a decision rather than after.

Scores are scraped from each battery's own output, because they do not share a format.
Anything this cannot parse is reported as `?` with its output kept - a test whose result
could not be read is not a test that passed.

    python tests/run_all.py --url v5=http://127.0.0.1:8930/detect \
                            --url v14=http://127.0.0.1:8962/detect
"""
import argparse, io, re, subprocess, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# Batteries that need something this runner cannot supply, and why.
SKIP = {
    "test_external_switchboard.py": "needs --swda; run separately at 150 conversations",
    "test_needs_location.py": "takes no --url",
    "run_all.py": "",
}

# Last resort ordering: the first pattern that matches wins, so put the explicit
# totals ahead of the bare fractions that appear in per-category lines.
PATTERNS = [
    re.compile(r"TOTAL\s+(\d+)\s*/\s*(\d+)"),
    re.compile(r"->\s*(\d+)\s*/\s*(\d+)"),
    re.compile(r"(?:passed|PASS(?:ED)?)\D{0,12}(\d+)\s*/\s*(\d+)"),
    re.compile(r"(\d+)\s*/\s*(\d+)\s+(?:cases|checks|scenarios)"),
]


def score(out):
    """Take the LAST match of the strongest pattern that appears at all."""
    for p in PATTERNS:
        m = p.findall(out)
        if m:
            a, b = m[-1]
            return int(a), int(b)
    # some batteries only print failures and say nothing when clean
    fails = len(re.findall(r"\b(?:FAIL|BUG|WRONG|REGRESSION|MISS)\b", out))
    if fails == 0 and out.strip():
        return None, 0          # ran, printed no failures
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", required=True,
                    help="name=url, repeatable")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default=None, help="directory for full outputs")
    a = ap.parse_args()

    arms = []
    for spec in a.url:
        name, _, url = spec.partition("=")
        if not url:
            raise SystemExit(f"  --url wants name=url, got {spec!r}")
        arms.append((name, url))

    outdir = Path(a.out) if a.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in TESTS.glob("test_*.py") if f.name not in SKIP)
    print(f"  {len(files)} batteries x {len(arms)} models\n")

    results = {}
    for f in files:
        results[f.name] = {}
        row = f"  {f.name:34}"
        for name, url in arms:
            t0 = time.time()
            try:
                r = subprocess.run([sys.executable, str(f), "--url", url],
                                   cwd=str(ROOT), capture_output=True, text=True,
                                   timeout=a.timeout, encoding="utf-8",
                                   errors="replace")
                out = (r.stdout or "") + (r.stderr or "")
                rc = r.returncode
            except subprocess.TimeoutExpired:
                out, rc = "TIMEOUT", -1
            if outdir:
                (outdir / f"{f.stem}.{name}.txt").write_text(out, encoding="utf-8")
            got, tot = score(out)
            results[f.name][name] = (got, tot, rc, time.time() - t0)
            if got is None and tot == 0:
                cell = "clean"
            elif got is None:
                cell = "?"
            else:
                cell = f"{got}/{tot}"
            row += f" {cell:>12}"
        print(row)

    print()
    hdr = f"  {'battery':34}" + "".join(f" {n:>12}" for n, _ in arms)
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    worse = []
    for fn, per in results.items():
        row = f"  {fn:34}"
        vals = []
        for name, _ in arms:
            got, tot, rc, el = per[name]
            vals.append(got if tot else None)
            cell = (f"{got}/{tot}" if got is not None and tot
                    else ("clean" if tot == 0 and got is None else "?"))
            row += f" {cell:>12}"
        # flag a battery where the LAST arm scores below the first
        if len(vals) > 1 and vals[0] is not None and vals[-1] is not None \
                and vals[-1] < vals[0]:
            row += "   <- REGRESSED"
            worse.append(fn)
        print(row)
    print("=" * len(hdr))
    if worse:
        print(f"\n  batteries where the last model scores below the first: {len(worse)}")
        for w in worse:
            print(f"    {w}")
    else:
        print("\n  no battery regressed against the first model")
    if outdir:
        print(f"\n  full output per battery in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
