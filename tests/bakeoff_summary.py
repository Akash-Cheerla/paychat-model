"""Rules against a model, on the same messages, scored the same way.

Asked for by Akash, 2026-10-08. The case for trying a model is that the two weaknesses
left in the rules - a third of real titles falling back to a generic word, and a quoted
example reading like an instruction - are the things rules are structurally worst at, and
the volume is small enough to afford one: 70 cards from 5,190 real messages.

Scored on the generated set, where every message says which parts must survive:

  KEPT      of what matters, how much reached the card
  LEAKED    of the chatter, how much reached it anyway
  CHATTER   all-noise messages correctly given no card
  TITLES    how many fell back to a generic word - the thing a model should win outright

The dogfood arm is separate and runs the RULES ONLY by default, because those are real
private conversations. A local model may be pointed at them (--local), since nothing
leaves the machine; the API proxy may never be, and refuses.

    python tests/bakeoff_summary.py --engines rules,api
    python tests/bakeoff_summary.py --engines rules --dogfood
"""
import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import summarizer as S                                      # noqa: E402
from summary_engines import (RulesEngine, LocalModelEngine, Seq2SeqEngine,   # noqa: E402
                             HybridEngine, ApiProxyEngine)

_STOP = frozenset("""
a an the this that these those it its we you they them our your their my his her i me
in on at to for of with from by as into onto over under after before once when while
until during about across through between among per via and or but not no nor if then
is are was were be been being has have had do does did will would shall should can
could may might must need needs so because there here just also still only very really
please kindly make sure
""".split())

# A title that says nothing about what the message was about.
GENERIC_TITLES = {"key points", "reminder", "update", "note", "notes", "message",
                  "summary", "info", "general", "misc", "chat"}


def content_words(text):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'\-\.]*", (text or "").lower())
            if w not in _STOP and len(w) > 1]


def coverage(phrase, points):
    want = content_words(phrase)
    if not want:
        return 1.0
    best = 0.0
    for p in points:
        have = set(content_words(p))
        best = max(best, sum(1 for w in want if w in have) / len(want))
    return best


KEEP_BAR, DROP_BAR = 0.7, 0.8


def compression(points, source):
    """How much shorter a bullet is than the sentence it came from.

    The metric this file was missing, and the reason its first verdict was wrong. KEPT%
    rewards a bullet that repeats the sentence word for word, so an engine that only
    selects sentences scores well on it while producing something nobody would call a
    summary - 58% of the rules engine's bullets are verbatim copies, averaging 89% of the
    source length (measured on real traffic, 2026-10-08).
    """
    import difflib
    sents = [x.strip(" .!?") for x in S.sentences(source)]
    if not sents:
        return 0.0, 0
    ratios, copies = [], 0
    for p in points:
        best, src = max(((difflib.SequenceMatcher(None, p.lower(), x.lower()).ratio(), x)
                         for x in sents), default=(0.0, ""))
        if best >= 0.97:
            copies += 1
        if src:
            ratios.append(len(p) / max(len(src), 1))
    return (sum(ratios) / len(ratios) if ratios else 0.0), copies


def score_generated(engine, cases, verbose=False):
    kept = missed = leaked = clean = 0
    chat_ok = chat_total = 0
    generic = titled = 0
    sizes, malformed = [], 0
    ratios, copies, bullets = [], 0, 0
    t0 = time.time()
    for i, c in enumerate(cases):
        card = engine.summarize(c["text"])
        points = (card or {}).get("points") or []
        if card:
            sizes.append(len(points))
            titled += 1
            if (card.get("title") or "").strip().lower() in GENERIC_TITLES:
                generic += 1
            malformed += sum(1 for p in points
                             if S._is_dangling(p) or len(p.split()) < 3
                             or S._is_only_mentions(p))
            ratio, cp = compression(points, c["text"])
            if ratio:
                ratios.append(ratio)
            copies += cp
            bullets += len(points)
        if not c["must_keep"]:
            chat_total += 1
            chat_ok += card is None
            continue
        for phrase in c["must_keep"]:
            if coverage(phrase, points) >= KEEP_BAR:
                kept += 1
            else:
                missed += 1
        for phrase in c["must_drop"]:
            if coverage(phrase, points) >= DROP_BAR:
                leaked += 1
            else:
                clean += 1
        if verbose and (i + 1) % 20 == 0:
            print(f"      ... {i+1}/{len(cases)}  ({(i+1)/max(time.time()-t0,1e-9):.1f}/s)",
                  flush=True)
    pct = lambda n, d: (100.0 * n / d) if d else 0.0        # noqa: E731
    return {
        "kept": pct(kept, kept + missed), "kept_n": f"{kept}/{kept+missed}",
        "leaked": pct(leaked, leaked + clean), "leaked_n": f"{leaked}/{leaked+clean}",
        "chatter": pct(chat_ok, chat_total), "chatter_n": f"{chat_ok}/{chat_total}",
        "generic": pct(generic, titled), "generic_n": f"{generic}/{titled}",
        "cards": titled, "malformed": malformed,
        "length": 100 * (sum(ratios) / len(ratios)) if ratios else 0.0,
        "verbatim": (100.0 * copies / bullets) if bullets else 0.0,
        "avg_points": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "seconds": time.time() - t0,
    }


def score_dogfood(engine, rows, limit=None):
    """No labels here - real traffic has none. Counts what can be checked structurally."""
    cards = 0
    pts = []
    generic = 0
    t0 = time.time()
    seen = 0
    for r in rows:
        text = r.get("text") or ""
        if not S.is_worth_summarizing(text):
            continue
        seen += 1
        if limit and seen > limit:
            break
        card = engine.summarize(text)
        if not card:
            continue
        cards += 1
        pts += card["points"]
        if (card.get("title") or "").strip().lower() in GENERIC_TITLES:
            generic += 1
    return {
        "qualified": seen, "cards": cards, "bullets": len(pts),
        "generic": (100.0 * generic / cards) if cards else 0.0,
        "generic_n": f"{generic}/{cards}",
        "dangling": sum(1 for p in pts if S._is_dangling(p)),
        "mentions": sum(1 for p in pts if S._is_only_mentions(p)),
        "short": sum(1 for p in pts if len(p.split()) < 3),
        "quoted": sum(1 for p in pts if p.strip() and p.strip()[0] in S._QUOTES
                      and p.strip()[-1] in S._QUOTES),
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="rules",
                    help="comma separated: rules, api, local, seq2seq, hybrid")
    ap.add_argument("--cases", default="data/eval/summary_messages.json")
    ap.add_argument("--dogfood", action="store_true",
                    help="also run the real log (rules and local only)")
    ap.add_argument("--dogfood-file", default="data/eval/dogfood_2026-10-07.jsonl")
    ap.add_argument("--limit", type=int, help="first N messages of each set")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seq2seq", default="sshleifer/distilbart-cnn-12-6")
    a = ap.parse_args()

    cases = json.loads((ROOT / a.cases).read_text(encoding="utf-8"))
    if a.limit:
        cases = cases[:a.limit]
    corpus = [c["text"] for c in cases]

    wanted = [e.strip() for e in a.engines.split(",") if e.strip()]
    engines = []
    for name in wanted:
        if name == "rules":
            engines.append(RulesEngine())
        elif name == "api":
            engines.append(ApiProxyEngine(corpus))
        elif name == "local":
            engines.append(LocalModelEngine(a.model))
        elif name == "seq2seq":
            engines.append(Seq2SeqEngine(a.seq2seq))
        elif name == "hybrid":
            engines.append(HybridEngine(a.seq2seq))
        else:
            print(f"  unknown engine {name!r}")
            return 2

    print(f"\n  GENERATED SET - {len(cases)} messages, labels say what must survive")
    print(f"  {'engine':<22}{'KEPT':>8}{'LEAKED':>9}{'CHATTER':>9}{'GENERIC TITLE':>15}"
          f"{'bullets':>9}{'VERBATIM':>10}{'LENGTH':>8}{'secs':>7}")
    results = {}
    for e in engines:
        print(f"    running {e.name} ...", flush=True)
        r = score_generated(e, cases, verbose=True)
        results[e.name] = r
        print(f"  {e.name:<22}{r['kept']:>7.1f}%{r['leaked']:>8.1f}%{r['chatter']:>8.1f}%"
              f"{r['generic']:>14.1f}%{r['avg_points']:>9.1f}"
              f"{r['verbatim']:>9.1f}%{r['length']:>7.0f}%{r['seconds']:>7.0f}")
        print(f"  {'':<22}{r['kept_n']:>8}{r['leaked_n']:>9}{r['chatter_n']:>9}"
              f"{r['generic_n']:>15}")

    if a.dogfood:
        rows = [json.loads(l) for l in
                (ROOT / a.dogfood_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n  REAL TRAFFIC - {len(rows)} messages, no labels, structural checks only")
        for e in engines:
            if isinstance(e, ApiProxyEngine):
                print(f"    {e.name}: SKIPPED - real conversations never go to an API")
                continue
            r = score_dogfood(e, rows, a.limit)
            print(f"    {e.name}: {r['cards']} cards from {r['qualified']} qualifying, "
                  f"{r['bullets']} bullets, {r['seconds']:.0f}s")
            print(f"      generic titles {r['generic_n']} ({r['generic']:.0f}%)  |  "
                  f"dangling {r['dangling']}  names-only {r['mentions']}  "
                  f"under-3-words {r['short']}  pure-quote {r['quoted']}")

    if len(results) >= 2:
        names = list(results)
        base, cand = results[names[0]], results[names[1]]
        print(f"\n  {names[1]} vs {names[0]}:")
        for key, label, better_is in (("kept", "important parts kept", "higher"),
                                      ("leaked", "chatter leaked", "lower"),
                                      ("chatter", "noise messages dropped", "higher"),
                                      ("generic", "generic titles", "lower"),
                                      ("verbatim", "bullets copied verbatim", "lower"),
                                      ("length", "bullet length vs source", "lower")):
            delta = cand[key] - base[key]
            good = (delta > 0) if better_is == "higher" else (delta < 0)
            mark = "better" if (good and abs(delta) >= 0.05) else (
                "same" if abs(delta) < 0.05 else "worse")
            print(f"    {label:<26}{base[key]:>7.1f}% -> {cand[key]:>6.1f}%  {delta:+6.1f}  {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
