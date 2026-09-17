"""Which robustness-lexicon phrasings already exist in the BASE training corpus.

A phrase the model has seen thousands of times in v16's windows ("ok", "$20") cannot
measure generalisation, so it must not sit in the held-out half. This writes the list of
such phrasings; tests/robustness_lexicon.py reads it and keeps them on the training side.

    python data_gen/lexicon_base_seen.py conv_windows_v16r.zip
"""
import io, json, re, sys, zipfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import robustness_lexicon as L   # noqa: E402

OUT = ROOT / "data/eval/lexicon_base_seen.json"
MIN_HITS = 5


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "conv_windows_v16r.zip"
    rows = json.loads(zipfile.ZipFile(ROOT / src).read("train.json"))
    finals, alltext = {}, []
    for w in rows:
        t = re.sub(r"\s+", " ", w["window"][-1]["t"].strip().lower())
        finals[t] = finals.get(t, 0) + 1
        alltext.append(" ".join(m["t"] for m in w["window"]))
    blob = "\n".join(alltext).lower()
    seen = []
    for name, items in L.LISTS.items():
        for p in items:
            low = p.lower()
            if "{n}" in low:      # amount format: does the corpus write numbers this way?
                pat = re.escape(low).replace(r"\{n\}", r"\d[\d,]*")
                hits = len(re.findall(pat, blob))
            elif "{" in low:      # request/offer template: too varied to match literally
                continue
            else:
                hits = finals.get(low, 0)
            if hits >= MIN_HITS:
                seen.append(p)
    OUT.write_text(json.dumps(sorted(seen), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {len(rows):,} base windows -> {len(seen)} lexicon phrasings already seen "
          f"(>= {MIN_HITS} times); wrote {OUT.relative_to(ROOT)}")
    for p in sorted(seen):
        print(f"    {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
