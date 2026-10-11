"""A day in a group chat, through the live endpoint, exactly as the app would.

The rehearsal before deploy: Samyak calls POST /message-summary as each message arrives
and stores the card; the Summary tab shows the cards that came back. This sends a
realistic day - long status updates, bug reports with numbers and owners, small talk, and
a decision reached across several short replies - and records what the tab would show.

The last of those is deliberately included. A decision spread over short messages is the
known limitation of per-message summarisation, and the rehearsal should show it honestly
rather than only the cases that work.
"""
import io, json, os, sys, time
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
# The SUMMARY service, which is a different process from the intent one on 8971 -
# see summary_app.py for why. Override with PAYCHAT_SUMMARY_URL.
URL = os.environ.get("PAYCHAT_SUMMARY_URL",
                     "http://127.0.0.1:8972") + "/message-summary"

# (sender, time, text)
DAY = [
    ("Gowtham", "09:02", "morning all"),
    ("Akash", "09:03", "morning"),
    ("Yash", "09:11",
     "Finished the regression pass on build 61 last night. Overall it's in much better "
     "shape than 60. Login works on both platforms now, and the chat list scrolls "
     "properly. Three things still open: the split amount shows blank when participants "
     "isn't sent, the ride prompt fires twice if you tap send quickly, and the Hindi "
     "translation on the payment sheet is cut off. I've logged all three in the sheet "
     "with screenshots."),
    ("Samyak", "09:14", "ok"),
    ("Samyak", "09:14", "will look at the participants one"),
    ("Brahma", "09:31",
     "iOS side — I've pushed the fix for the summary card showing duplicate bullets. It "
     "was the on-device model repeating itself under beam search. Also noticed the card "
     "titles are losing build numbers, so \"Build 52\" was coming out as \"App Build "
     "Testing\". That's fixed too. Both are in TestFlight build 61.3, please pull it "
     "before you test anything else today."),
    ("Rupesh", "09:40", "pulled it"),
    ("Rupesh", "09:52", "cab booking works from current location now"),
    ("Gowtham", "10:15",
     "Team — the investor demo moved up to Thursday 2pm instead of next Monday. That "
     "gives us two days. I need the money flow and the ride flow both working end to end "
     "with zero errors on the happy path. Anything cosmetic we can live with, but a "
     "failed payment or a cab that doesn't book is not something I can show. Akash can "
     "you own the money side and Andril the ride side? Let's sync at 4pm today and again "
     "tomorrow morning."),
    ("Akash", "10:16", "yes will do"),
    ("Andril", "10:17", "on it"),
    ("Yash", "10:44", "anyone else getting a 500 on the room summary endpoint"),
    ("Samyak", "10:45", "checking"),
    ("Samyak", "10:51", "fixed, was a bad deploy. redeployed"),
    ("Yash", "10:52", "working now"),
    ("Rupesh", "11:30", "lunch?"),
    ("Brahma", "11:31", "give me 20 mins"),
    ("Rupesh", "11:31", "the place downstairs is closed btw"),
    ("Akash", "11:32", "the new one near the metro is decent"),
    ("Andril", "13:20",
     "Ride side update for the demo. Current location to a saved place works, named "
     "places work if the roster is sent, and \"my current location to your current "
     "location\" resolves correctly now. The one case still broken is when someone says "
     "a place we don't have saved — it falls back to asking, which is correct behaviour "
     "but looks like a bug in the demo. I'd suggest we pre-save Gowtham's home and "
     "office before Thursday so that path never triggers."),
    ("Gowtham", "13:22", "good idea"),
    ("Gowtham", "13:22", "do that"),
    ("Akash", "14:05",
     "Money side. The split bug from the 7th is fixed and deployed — $208 across 9 people "
     "now shows $23.11 each instead of $208. I've also fixed the case where someone "
     "questions an amount and gets corrected, which was showing the wrong figure. Both "
     "passed the gate with real traffic unchanged. The one thing I'd avoid in the demo is "
     "a split where the group size isn't sent, because that deliberately shows a blank "
     "amount rather than guessing."),
    ("Samyak", "14:07", "participants is being sent on all group rooms now"),
    ("Akash", "14:07", "then we're fine"),
    ("Yash", "15:40", "QA-114 and QA-115 both pass on 61.3"),
    ("Yash", "15:41", "QA-116 still fails"),
    ("Brahma", "15:44", "that's mine, looking"),
    ("Gowtham", "16:02", "sync in 5"),
    ("Rupesh", "17:55", "that storm is wild"),
    ("Brahma", "17:58", "lost power for an hour yesterday"),
]


def main():
    s = requests.Session()
    out = []
    t0 = time.time()
    for i, (who, when, text) in enumerate(DAY):
        t = time.time()
        r = s.post(URL, json={"text": text}, timeout=180).json()
        dt = time.time() - t
        out.append({"sender": who, "time": when, "text": text,
                    "card": r.get("summary"), "source": r.get("source"),
                    "seconds": round(dt, 2)})
        mark = "CARD" if r.get("summary") else "  - "
        print(f"  {mark} {when} {who:<9} {dt:5.1f}s  {text[:58]}")
    carded = [o for o in out if o["card"]]
    print(f"\n  {len(DAY)} messages -> {len(carded)} cards in {time.time()-t0:.0f}s")
    print(f"  model time only on the {len(carded)} that carded; "
          f"the other {len(DAY)-len(carded)} returned in "
          f"{sum(o['seconds'] for o in out if not o['card'])/max(len(DAY)-len(carded),1):.2f}s avg")
    json.dump(out, io.open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
