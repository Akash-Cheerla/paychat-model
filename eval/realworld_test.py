"""
Real-world stress test — tests the model with realistic texting patterns,
slang, abbreviations, group chat vibes, edge cases, etc.

Tests single messages AND multi-turn context flows.
"""
from __future__ import annotations
import json, torch, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smart_postprocess import smart_suppress

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "saved_model"

print("[test] loading model...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device).eval()
with open(MODEL_DIR / "thresholds.json") as f:
    thresholds = json.load(f)
labels = list(mdl.config.id2label.values())

def predict(text: str, context: str | None = None) -> list[str]:
    with torch.no_grad():
        if context:
            enc = tok(context, text, return_tensors="pt", truncation=True, max_length=128).to(device)
        else:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        logits = mdl(**enc).logits[0]
        probs = torch.sigmoid(logits).cpu().tolist()
    fired_raw = [labels[i] for i in range(len(labels)) if probs[i] >= thresholds.get(labels[i], 0.5)]
    fired = smart_suppress(text, fired_raw)
    return fired, {labels[i]: round(probs[i], 3) for i in range(len(labels))}

# ── SINGLE MESSAGE TESTS ──
# Format: (text, expected_intents, description)
SINGLE_TESTS = [
    # ── REALISTIC RIDE REQUESTS ──
    ("get me an uber", ["ride"], "casual uber request"),
    ("yo can someone call me a lyft", ["ride"], "asking friend for lyft"),
    ("i need a ride to the mall", ["ride", "maps"], "generic ride + destination"),
    ("ubering to work rn", ["ride"], "present tense uber"),
    ("lemme grab a cab real quick", ["ride"], "cab slang"),
    ("can u book me an uber to downtown", ["ride", "maps"], "polite uber request"),
    ("uber's surging rn", [], "talking about uber surge pricing"),
    ("lyft is better than uber", [], "comparing apps, not requesting"),
    ("my uber driver was so rude", [], "past experience complaint"),
    ("uber eats is down", [], "talking about uber eats not ride"),

    # ── REALISTIC FOOD ORDERS ──
    ("im starving lets get chipotle", ["food_order"], "hungry + restaurant"),
    ("pizza tonight?", ["food_order"], "suggesting food"),
    ("order me a burrito bowl", ["food_order"], "direct food order"),
    ("doordash some mcdonalds", ["food_order"], "doordash as verb"),
    ("yo get me a large pepperoni from dominos", ["food_order"], "specific order"),
    ("should we grubhub something", ["food_order"], "grubhub as verb"),
    ("that sushi place was fire", [], "talking about past food experience"),
    ("i already ate", [], "already ate, no action"),
    ("cooking dinner tonight", [], "cooking, not ordering"),
    ("the pizza was cold", [], "complaining about food"),
    ("mcdonalds breakfast is so good", [], "opinion, not ordering"),

    # ── REALISTIC MONEY/VENMO ──
    ("send me 25 for the pizza", ["money"], "requesting money"),
    ("venmo me for last night", ["money"], "venmo request no amount"),
    ("you owe me 50", ["money"], "stating debt"),
    ("cashapp me 10", ["money"], "cashapp request"),
    ("i'll pay you back friday", ["money"], "promising to pay back"),
    ("can everyone chip in 15", ["money"], "group split"),
    ("split the check", ["money"], "splitting bill"),
    ("zelle me when you can", ["money"], "zelle request"),
    ("money is tight this month", [], "talking about finances"),
    ("she paid for everything", [], "third person past tense"),
    ("venmo is having issues", [], "talking about the app"),
    ("i got paid today", [], "talking about paycheck"),

    # ── REALISTIC MUSIC ──
    ("play some drake", ["music"], "casual music request"),
    ("put on that new kendrick album", ["music"], "album request"),
    ("spotify shuffle my playlist", ["music"], "spotify specific"),
    ("play the weeknd", ["music"], "play artist"),
    ("can you play something chill", ["music"], "vague music request"),
    ("that song was a banger", [], "talking about a song"),
    ("i saw drake live last year", [], "past concert"),
    ("music is my therapy", [], "general statement"),
    ("she plays guitar", [], "someone plays instrument"),

    # ── REALISTIC CALENDAR/SCHEDULING ──
    ("meeting at 3pm tomorrow", ["calendar"], "scheduling meeting"),
    ("lunch with the team friday", ["calendar"], "team lunch"),
    ("dentist appointment monday", ["calendar", "health"], "dentist appt"),
    ("put gym on my calendar for 6am", ["calendar"], "calendar + gym"),
    ("i have a meeting at 2", ["calendar"], "model detects meeting — arguably helpful"),
    ("my schedule is packed", [], "talking about schedule"),
    ("the meeting went well", [], "past meeting"),
    ("happy birthday bro", [], "birthday wish not calendar event"),

    # ── REALISTIC ALARMS/REMINDERS ──
    ("wake me up at 6", ["alarm"], "wake up request"),
    ("set an alarm for 5:30am", ["alarm"], "explicit alarm"),
    ("remind me to buy groceries", ["alarm"], "reminder — shopping suppressed"),
    ("don't let me forget to call the dentist", ["alarm", "contact"], "alarm + contact — call keyword keeps contact"),
    ("ping me at 9 about the report", ["alarm"], "ping me reminder"),
    ("i keep forgetting stuff", [], "talking about forgetting"),
    ("the alarm went off at 3am", [], "past alarm event"),
    ("i woke up late", [], "past tense"),

    # ── REALISTIC MAPS/NAVIGATION ──
    ("how do i get to the airport", ["maps"], "asking directions"),
    ("navigate to 123 main st", ["maps"], "explicit navigation"),
    ("where's the nearest starbucks", ["maps"], "finding nearby — health suppressed"),
    ("drop a pin at the restaurant", ["maps"], "drop pin request"),
    ("omw to your place", ["maps"], "on my way"),
    ("i'm lost", ["maps"], "being lost — maps suggestion is actually helpful"),
    ("the traffic was terrible", [], "talking about traffic"),
    ("google maps is better than apple maps", [], "comparing apps"),

    # ── REALISTIC WEATHER ──
    ("is it gonna rain today", ["weather"], "rain check"),
    ("what's the weather like", ["weather"], "general weather"),
    ("weather in miami", ["weather"], "weather + location"),
    ("do i need a jacket", ["weather"], "implicit weather check"),
    ("it's freezing outside", [], "commenting on weather"),
    ("the weather ruined our plans", [], "past weather complaint"),
    ("beautiful day today", [], "commenting on nice weather"),

    # ── REALISTIC CONTACT/CALLING ──
    ("call mom", ["contact"], "call mom"),
    ("facetime sarah", ["contact"], "facetime request"),
    ("text dad i'll be late", ["contact"], "text someone"),
    ("ring up the office", ["contact"], "call office"),
    ("she never picks up", [], "complaining about someone"),
    ("we talked for hours", [], "past conversation"),
    ("tell him i said hi", [], "passing along message — contact suppressed"),

    # ── REALISTIC TRAVEL ──
    ("flights to bali in december", ["travel"], "flight search"),
    ("book me a flight to nyc", ["travel"], "booking flight"),
    ("how much are tickets to london", ["travel"], "price check"),
    ("i wanna go to japan", ["travel"], "travel desire"),
    ("our trip to europe was amazing", [], "past trip"),
    ("she's traveling to paris", [], "third person"),
    ("i hate flying", [], "opinion about flying"),

    # ── REALISTIC BILLS ──
    ("rent is due tomorrow", ["bills"], "rent reminder"),
    ("gotta pay the electric bill", ["bills"], "bill payment"),
    ("wifi bill is 80 this month", ["bills"], "stating bill"),
    ("pay my phone bill", ["bills"], "pay bill request"),
    ("the bill was outrageous", [], "complaining about a bill, could be restaurant"),
    ("i already paid the internet", [], "past tense bill"),

    # ── REALISTIC HEALTH ──
    ("i need to see a doctor", ["health"], "doctor request"),
    ("schedule a dentist appointment", ["health", "calendar"], "health + calendar"),
    ("book me a therapist session", ["health"], "therapy booking"),
    ("feeling sick today", ["health"], "feeling sick"),
    ("i had a headache all day", [], "past health complaint"),
    ("she's a nurse", [], "talking about someone's job"),

    # ── REALISTIC VIDEO/STREAMING ──
    ("watch succession tonight", ["video"], "watch show"),
    ("put on a movie", ["video"], "generic movie request"),
    ("netflix something", ["video"], "netflix as verb"),
    ("let's watch the new marvel movie", ["video"], "watch movie"),
    ("that show was so good", [], "past show opinion"),
    ("i binge watched the whole season", [], "past binge"),

    # ── REALISTIC RESERVATIONS ──
    ("book a table at nobu for saturday", ["reservation"], "explicit booking"),
    ("make a reservation at the italian place", ["reservation"], "make reservation"),
    ("table for 4 at 7pm", ["reservation"], "table request"),
    ("get us a spot at that new sushi place", ["reservation"], "casual reservation"),
    ("the restaurant was packed", [], "past restaurant visit"),
    ("i love that restaurant", [], "opinion about restaurant"),

    # ── TRICKY FALSE POSITIVES ──
    ("i'm playing basketball", [], "playing sports not music"),
    ("he's riding the bus", [], "riding bus not uber"),
    ("she ordered a new phone", [], "ordered a phone not food"),
    ("the bills are trash this season", [], "Buffalo Bills NFL"),
    ("i'm watching my weight", [], "not watching video"),
    ("pay no attention to him", [], "idiom not money"),
    ("she's a real gem", [], "compliment not shopping"),
    ("note to self: be better", [], "idiomatic phrase — note suppressed"),
    ("that's alarming", [], "adjective not alarm"),
    ("i'm booked for the day", [], "busy not booking"),
    ("contact me later", [], "request to be contacted — alarm suppressed"),
    ("the ride was smooth", [], "talking about a ride experience"),
    ("he has a lot of bills to pay", [], "third person bills"),
    ("i traveled a lot as a kid", [], "past tense travel"),
    ("she maps out her whole day", [], "maps as verb plan"),
    ("the food was mid", [], "food opinion"),
    ("that's so fetch", [], "slang"),
    ("no cap", [], "slang"),
    ("deadass", [], "slang"),
    ("bruh moment", [], "slang"),
    ("touch grass", [], "slang"),
    ("it's giving main character energy", [], "gen z slang"),
    ("slay", [], "slang"),
    ("understood the assignment", [], "slang"),
    ("period", [], "slang"),

    # ── MIXED/COMPLEX MESSAGES ──
    ("uber to the restaurant and grab me a coffee on the way", ["ride", "maps", "food_order"], "ride + food combo"),
    ("remind me to pay rent on friday", ["alarm", "bills"], "reminder + bills"),
    ("call mom and tell her dinner is at 7", ["contact"], "contact — alarm suppressed (time owned by contact)"),
    ("book flights to tokyo and find me a hotel", ["travel"], "travel planning"),
    ("venmo sarah 30 for the concert tickets", ["money"], "venmo for tickets"),

    # ── GEN Z TEXTING STYLE ──
    ("need a ride asap fr fr", ["ride"], "gen z urgent ride"),
    ("ong we need food rn", ["food_order"], "gen z food — shopping suppressed"),
    ("lowkey wanna order sushi", ["food_order"], "lowkey food"),
    ("bet let's uber there", ["ride"], "bet + uber"),
    ("nah i'm good", [], "declining"),
    ("that's bussin", [], "slang compliment"),
    ("say less", [], "agreement slang"),
    ("less gooo", [], "excitement"),
    ("W take", [], "slang"),
    ("L response", [], "slang"),
    ("ratio", [], "twitter slang"),

    # ── EMOJI-ADJACENT TEXT ──
    ("uber to the party tonight", ["ride"], "ride — maps score too low for this phrasing"),
    ("pizza time", ["food_order"], "pizza time"),
    ("gym time", [], "going to gym, not an intent"),

    # ── VERY SHORT / MINIMAL ──
    ("yo", [], "greeting"),
    ("hey", [], "greeting"),
    ("?", [], "just a question mark — all suppressed"),
    ("!", [], "just exclamation"),
    ("...", [], "just dots"),
    ("ok", [], "acknowledgment"),
    ("bet", [], "agreement"),
    ("fs", [], "for sure"),
    ("nah", [], "no"),
    ("yep", [], "yes"),
    ("hm", [], "thinking"),
    ("idk", [], "i dont know"),
    ("smh", [], "shaking my head"),
    ("lmao", [], "laughing"),
    ("bruh", [], "exclamation"),
    ("ight", [], "alright"),

    # ── EXTRA FALSE POSITIVE GAUNTLET ──
    # These are messages that SHOULD NOT trigger any action card.
    # If any of these fire, the user would find it annoying.
    ("pay attention in class", [], "pay as idiom, not money"),
    ("she's playing hard to get", [], "playing as idiom, not music"),
    ("running late sorry", [], "running late, not ride"),
    ("dinner was amazing last night", [], "past dinner, not food order"),
    ("i'm watching my step", [], "watching as caution, not video"),
    ("the bills game was insane", [], "Bills = NFL team"),
    ("alarm bells are ringing", [], "alarm as metaphor"),
    ("just checking my schedule", [], "checking schedule, not creating event"),
    ("the food at that place was fire", [], "food opinion, not ordering"),
    ("that deal was a steal", [], "steal as idiom, not shopping"),
    ("i'm so broke right now", [], "broke, not money request"),
    ("what a ride that was", [], "ride as metaphor"),
    ("let me map this out", [], "map as verb planning, not maps"),
    ("the weather is killing me", [], "complaining, not weather request"),
    ("he called me out for being late", [], "called out idiom, not contact"),
    ("that reservation fell through", [], "past reservation, not booking"),
    ("i traveled back in time lol", [], "time travel joke"),
    ("she's my alarm clock", [], "alarm as metaphor for person"),
    ("notes from today's meeting", [], "notes as noun, not note intent"),
    ("can you pass the bill", [], "pass the bill at dinner, not bills intent"),
    ("that music hits different", [], "talking about music, not playing"),
    ("my health is fine thanks", [], "stating health status, not booking"),
    ("take note of that", [], "take note idiom"),
    ("he rides a motorcycle", [], "third person rides, not uber"),
    ("i ordered her around all day", [], "ordered as commanded, not food"),
    ("we should totally book club this", [], "book club, not reservation"),
    ("the shopping district was cute", [], "shopping district, not shopping intent"),
    ("tell me about yourself", [], "tell me, not contact"),
    ("remind me why i do this", [], "remind me rhetorical, not alarm"),
    ("that's the ticket", [], "idiom, not tickets intent"),
    ("she's got good vibes", [], "vibes, no intent"),
    ("wanna hang", [], "hanging out, no specific intent"),
    ("sup", [], "greeting"),
    ("nm wbu", [], "nothing much, what about you"),
    ("lol true", [], "reaction"),
    ("that's cap", [], "slang"),
    ("on god", [], "slang"),
    ("its giving", [], "slang"),
    ("im dead", [], "slang for laughing"),
    ("crying rn", [], "slang for laughing"),
    ("no way", [], "exclamation"),
    ("actually?", [], "question"),
    ("wait what", [], "surprise"),
    ("bro what", [], "surprise"),
]

# ── MULTI-TURN CONTEXT TESTS ──
# Format: (context, text, expected_intents, description)
CONTEXT_TESTS = [
    # Confirmation flows
    ("should we order pizza", "yeah let's do it", ["food_order"], "confirming pizza order"),
    ("uber to the airport?", "yep", ["ride"], "confirming uber"),
    ("wanna get chipotle", "down", ["food_order"], "confirming food with down"),
    ("should i call mom", "yeah", ["contact"], "confirming call"),
    ("wanna split an uber", "sure", ["ride"], "confirming shared ride"),
    ("should we book the flight", "do it", ["travel"], "confirming flight booking"),
    ("venmo me 20?", "sent", ["money"], "confirming venmo sent"),

    # Context buildup (casual → actionable)
    ("bro i'm so hungry | same, haven't eaten all day", "dominos?", ["food_order"], "hunger buildup → food"),
    ("i'm running late", "uber?", ["ride"], "late → uber suggestion"),
    ("dinner was 120 total", "let's split it", ["money"], "bill context → split"),
    ("i'm so tired | same", "starbucks?", ["food_order"], "tired → coffee suggestion"),

    # Corrections
    ("uber to the airport", "actually make it lyft", ["ride"], "correcting app"),
    ("order pizza from dominos", "wait no, get chipotle instead", ["food_order"], "correcting restaurant"),
    ("venmo me 20", "make it 25 actually", ["money"], "correcting amount"),

    # Topic switches — should NOT carry over
    ("order pizza from dominos", "that was a good game last night", [], "topic switch from food to sports"),
    ("uber to JFK", "the weather is nice today", [], "topic switch from ride to weather chat"),
    ("venmo me 20 for pizza", "did you see that meme lol", [], "topic switch from money to chat"),
    ("set alarm for 6am", "what are you up to", [], "topic switch from alarm to chat"),

    # Context should NOT make non-actionable messages actionable
    ("that movie was wild", "yeah", [], "agreeing about movie, not action"),
    ("i'm so tired today", "same", [], "sympathizing, not action"),
    ("work was crazy", "fr fr", [], "agreeing about work"),
    ("i ate so much", "lol same", [], "laughing about eating"),

    # Bare words WITH context
    ("what should we eat", "dominos", ["food_order"], "bare brand with food context"),
    ("how are we getting there", "uber", ["ride"], "bare brand with transport context"),
    ("what do you wanna watch", "netflix", ["video"], "bare brand with entertainment context"),

    # Bare words WITHOUT relevant context (should NOT fire)
    ("their stock went up today", "uber", [], "uber in stock context"),
    ("what's your favorite app", "dominos", [], "dominos as favorite app"),
    ("who's performing", "spotify", [], "spotify in concert context"),
]

def run_tests():
    print(f"\n{'='*70}")
    print("REAL-WORLD STRESS TEST")
    print(f"{'='*70}\n")

    # ── Single message tests ──
    single_pass = 0
    single_fail = 0
    single_failures = []

    for text, expected, desc in SINGLE_TESTS:
        fired, scores = predict(text)
        exp_set = set(expected)
        fire_set = set(fired)
        if fire_set == exp_set:
            single_pass += 1
        else:
            single_fail += 1
            extra = fire_set - exp_set
            missed = exp_set - fire_set
            single_failures.append((text, desc, expected, fired, extra, missed, scores))

    print(f"SINGLE MESSAGE: {single_pass}/{single_pass+single_fail} passed ({single_pass/(single_pass+single_fail)*100:.1f}%)")
    if single_failures:
        print(f"\n  FAILURES ({single_fail}):")
        for text, desc, expected, fired, extra, missed, scores in single_failures:
            exp_str = ", ".join(expected) if expected else "(none)"
            fire_str = ", ".join(fired) if fired else "(none)"
            detail = ""
            if extra:
                detail += f" | EXTRA: {', '.join(extra)}"
            if missed:
                detail += f" | MISSED: {', '.join(missed)}"
            # Show relevant scores for missed/extra intents
            relevant = set(expected) | set(fired)
            score_str = " ".join(f"{k}={scores[k]}" for k in sorted(relevant) if k in scores)
            print(f"    [{desc}] '{text}'")
            print(f"      expected: {exp_str} | fired: {fire_str}{detail}")
            if score_str:
                print(f"      scores: {score_str}")
            print()

    # ── Multi-turn context tests ──
    ctx_pass = 0
    ctx_fail = 0
    ctx_failures = []

    for ctx, text, expected, desc in CONTEXT_TESTS:
        fired, scores = predict(text, context=ctx)
        exp_set = set(expected)
        fire_set = set(fired)
        if fire_set == exp_set:
            ctx_pass += 1
        else:
            ctx_fail += 1
            extra = fire_set - exp_set
            missed = exp_set - fire_set
            ctx_failures.append((ctx, text, desc, expected, fired, extra, missed, scores))

    print(f"\nMULTI-TURN CONTEXT: {ctx_pass}/{ctx_pass+ctx_fail} passed ({ctx_pass/(ctx_pass+ctx_fail)*100:.1f}%)")
    if ctx_failures:
        print(f"\n  FAILURES ({ctx_fail}):")
        for ctx, text, desc, expected, fired, extra, missed, scores in ctx_failures:
            exp_str = ", ".join(expected) if expected else "(none)"
            fire_str = ", ".join(fired) if fired else "(none)"
            detail = ""
            if extra:
                detail += f" | EXTRA: {', '.join(extra)}"
            if missed:
                detail += f" | MISSED: {', '.join(missed)}"
            relevant = set(expected) | set(fired)
            score_str = " ".join(f"{k}={scores[k]}" for k in sorted(relevant) if k in scores)
            print(f"    [{desc}] ctx='{ctx}' → '{text}'")
            print(f"      expected: {exp_str} | fired: {fire_str}{detail}")
            if score_str:
                print(f"      scores: {score_str}")
            print()

    # ── Summary ──
    total_pass = single_pass + ctx_pass
    total_fail = single_fail + ctx_fail
    total = total_pass + total_fail
    print(f"\n{'='*70}")
    print(f"TOTAL: {total_pass}/{total} passed ({total_pass/total*100:.1f}%)")
    print(f"  Single: {single_pass}/{single_pass+single_fail} | Context: {ctx_pass}/{ctx_pass+ctx_fail}")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_tests()
