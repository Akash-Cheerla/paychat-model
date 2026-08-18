"""Read real WhatsApp group exports for STRUCTURE, and write nothing that identifies anyone.

These are Akash's own chats, handed over so we can see how people actually coordinate
rides and money — not to be dropped into training data. Five rounds of training have
moved the behavioural numbers 1-2% each because every conversation the model has ever
seen was written by us or by an LLM we prompted. The register is wrong in ways we
cannot see from inside it.

So this file answers questions ABOUT the corpus and never reproduces it:

  * how long is a real message, how long is a real exchange
  * how does a real ride request get phrased, and what does agreement look like
  * how far apart are a request and its answer, in messages and in minutes
  * how much of a real group chat is noise between the useful bits

Everything written out is aggregate or de-identified. Names become stable pseudonyms
(A1, A2...), phone numbers, links and emails are stripped before anything is counted,
and no verbatim message is stored unless it survives the redactor and is needed as a
phrasing example. The people in these chats did not agree to be training data, and the
distinction between "learn the shape" and "copy the content" is the whole point.

  python data_gen/wa_study.py --dir "<export dir>" --report
"""
import argparse, hashlib, json, re, sys, io
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LINE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}), (\d{1,2}:\d{2}:\d{2}\s*[AP]M)\]\s*([^:]{1,60}):\s*(.*)$")
SYSTEM = re.compile(r"joined using|created this group|end-to-end encrypted|changed the|"
                    r"added|removed|left$|Disappearing messages|You joined|"
                    r"changed this group's|deleted this message|image omitted|"
                    r"video omitted|sticker omitted|document omitted|audio omitted|"
                    r"This message was deleted", re.I)

PHONE = re.compile(r"[\+\(]?\d[\d\-\(\)\s\u2011]{7,}\d")
URL   = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\S+@\S+\.\S+")

# What a ride request looks like when a real person writes one.
RIDE_REQ = re.compile(r"\b(need\s+a?\s*ride|looking\s+for\s+a?\s*ride|any\s*(?:one|body)\s+going|"
                      r"ride\s+(?:to|from)|seat[s]?\s+available|going\s+to\s+\w+|"
                      r"pick\s*up|drop\s*(?:off)?|leaving\s+(?:at|from)|"
                      r"anyone\s+(?:driving|travelling|traveling))\b", re.I)
MONEY = re.compile(r"\b(etransfer|e-transfer|interac|send\s+(?:me|you|it)|paid|payment|"
                   r"\$\s?\d+|\d+\s?(?:cad|usd|dollars|bucks|rs|inr)|venmo|zelle|"
                   r"gpay|upi|paytm|split|owe)\b", re.I)
AGREE = re.compile(r"^\s*(ok(ay)?|sure|yes|yeah|yep|ya|done|k|kk|noted|cool|fine|"
                   r"got it|will do|on it|i can|i will|ill|sounds good|perfect|"
                   r"confirmed?|booked)\b", re.I)


def redact(s):
    s = URL.sub("<link>", s)
    s = EMAIL.sub("<email>", s)
    s = PHONE.sub("<number>", s)
    return s.strip()


def parse(path: Path):
    msgs, alias, cur = [], {}, None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.replace("\u200e", "").replace("\u202a", "").replace("\u202c", "")
        m = LINE.match(raw)
        if not m:
            if cur is not None and raw.strip():
                cur["text"] += " " + raw.strip()
            continue
        date, tm, who, text = m.groups()
        if SYSTEM.search(text) or not text.strip():
            cur = None
            continue
        who = who.strip()
        if who not in alias:
            alias[who] = "A%d" % (len(alias) + 1)
        try:
            ts = datetime.strptime(f"{date} {tm.replace(chr(8239),' ')}", "%Y-%m-%d %I:%M:%S %p")
        except ValueError:
            ts = None
        cur = {"ts": ts, "who": alias[who], "text": redact(text)}
        msgs.append(cur)
    return msgs, alias


def report(msgs, alias, name):
    if not msgs:
        print(f"  {name}: nothing parsed"); return {}
    lens = [len(m["text"].split()) for m in msgs]
    lens_sorted = sorted(lens)
    med = lens_sorted[len(lens_sorted) // 2]
    dates = [m["ts"] for m in msgs if m["ts"]]
    per_person = Counter(m["who"] for m in msgs)

    ride = [m for m in msgs if RIDE_REQ.search(m["text"])]
    money = [m for m in msgs if MONEY.search(m["text"])]
    agree = [m for m in msgs if AGREE.match(m["text"])]

    # How many messages sit between a ride request and the first agreement after it.
    gaps, tgaps = [], []
    for i, m in enumerate(msgs):
        if not RIDE_REQ.search(m["text"]):
            continue
        for j in range(i + 1, min(i + 25, len(msgs))):
            if msgs[j]["who"] != m["who"] and AGREE.match(msgs[j]["text"]):
                gaps.append(j - i)
                if m["ts"] and msgs[j]["ts"]:
                    tgaps.append((msgs[j]["ts"] - m["ts"]).total_seconds() / 60)
                break

    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    print(f"  messages {len(msgs):>7}   people {len(alias):>4}   "
          f"{dates[0].date() if dates else '?'} to {dates[-1].date() if dates else '?'}")
    print(f"  median message length      {med} words   "
          f"({sum(1 for l in lens if l <= 3)/len(lens):.0%} are <=3 words)")
    print(f"  busiest person writes       {per_person.most_common(1)[0][1]/len(msgs):.0%} of all messages")
    print(f"  ride-request shaped        {len(ride):>6}  ({len(ride)/len(msgs):.1%})")
    print(f"  money-shaped               {len(money):>6}  ({len(money)/len(msgs):.1%})")
    print(f"  bare agreements            {len(agree):>6}  ({len(agree)/len(msgs):.1%})")
    if gaps:
        gs = sorted(gaps)
        print(f"  request -> agreement       median {gs[len(gs)//2]} messages apart, "
              f"{sum(1 for g in gaps if g > 1)/len(gaps):.0%} not adjacent")
    if tgaps:
        ts_ = sorted(tgaps)
        print(f"                             median {ts_[len(ts_)//2]:.0f} min apart, "
              f"{sum(1 for t in tgaps if t > 60)/len(tgaps):.0%} over an hour")
    return {"messages": len(msgs), "people": len(alias), "ride": len(ride),
            "money": len(money), "agree": len(agree),
            "median_len": med, "gaps": gaps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {}
    for d in a.dir:
        p = Path(d) / "_chat.txt"
        if not p.exists():
            print(f"  missing {p}"); continue
        msgs, alias = parse(p)
        out[Path(d).name.replace("WhatsApp Chat - ", "")] = report(msgs, alias, Path(d).name)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
