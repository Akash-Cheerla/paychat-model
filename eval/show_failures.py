import json
rpt = json.load(open("eval/baseline_report.json"))
print(f"PASSED: {rpt['passed']}/{rpt['total']} ({rpt['passed']/rpt['total']*100:.1f}%)")
print(f"IID: {rpt['test_exact_match']:.2f}%")
print()
for r in rpt["results"]:
    if not r["match"]:
        exp = ", ".join(r["expected"]) or "(none)"
        fired = ", ".join(r["fired"]) or "(none)"
        extra = ", ".join(r.get("extra", []))
        missed = ", ".join(r.get("missed", []))
        ctx = r.get("context", "")
        print(f"FAIL [{r['tag']}]: {r['text']!r}")
        if ctx:
            print(f"  context: {ctx!r}")
        print(f"  expected: {exp}")
        print(f"  fired:    {fired}")
        if extra:
            print(f"  EXTRA:    {extra}")
        if missed:
            print(f"  MISSED:   {missed}")
        top5 = r["top5"][:5]
        parts = []
        for name, prob, thr in top5:
            parts.append(f"{name}={prob:.3f}(thr={thr:.2f})")
        print(f"  top probs: {'  '.join(parts)}")
        print()
