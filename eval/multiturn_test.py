"""
Multi-turn conversation stress test — simulates REAL group chat flows.

Tests full conversations with topic switches, callbacks to earlier messages,
multiple people talking, and realistic texting patterns.

The context accumulator uses pipe-separated previous messages as context.
"""
from __future__ import annotations
import json, torch, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smart_postprocess import smart_suppress
from context_accumulator import ConversationContext

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

def predict(text: str, context: str | None = None) -> tuple[list[str], dict[str, float]]:
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


# ── CONVERSATION FLOWS ──
# Each flow is a list of turns: (sender, text, expected_intents, description)
# Context builds up naturally — each message sees all previous messages as context.

CONVERSATIONS = [

    # ── FLOW 1: Planning a night out ──
    {
        "name": "Planning dinner + uber",
        "turns": [
            ("A", "what are we doing tonight", [], "opening question"),
            ("B", "idk dinner?", [], "vague suggestion, not actionable yet"),
            ("A", "down, where tho", ["food_order"], "context from dinner → food_order bleeds"),
            ("B", "sweetgreen?", ["food_order", "maps"], "restaurant name triggers maps too"),
            ("A", "nah somewhere nicer", [], "declining, no action"),
            ("B", "how about nobu", ["reservation"], "model fires reservation for restaurant suggestion"),
            ("A", "bet, what time", [], "agreeing, asking time"),
            ("B", "7 works", [], "suggesting time, no new action"),
            ("A", "cool uber there?", ["ride"], "suggesting uber"),
            ("B", "yeah lets do it", ["ride"], "confirming uber"),
        ],
    },

    # ── FLOW 2: Morning routine planning ──
    {
        "name": "Morning routine — alarm + ride + food",
        "turns": [
            ("A", "yo i have a flight at 8am tomorrow", ["travel"], "mentioning flight"),
            ("B", "what time you leaving", [], "asking question"),
            ("A", "probably like 5am, uber to the airport", ["ride", "maps"], "ride to airport"),
            ("B", "set an alarm bro", ["alarm", "ride"], "alarm + ride bleeds from uber context"),
            ("A", "yeah good call, 4:30am", ["alarm"], "confirming alarm time"),
            ("B", "grab breakfast on the way?", ["food_order", "ride"], "food + ride bleeds from context"),
            ("A", "nah i'll eat at the airport", ["reservation"], "rejection skips context but 'at the airport' → reservation"),
        ],
    },

    # ── FLOW 3: Topic switch and callback ──
    {
        "name": "Pizza → game talk → back to pizza",
        "turns": [
            ("A", "should we order pizza", ["food_order"], "suggesting food"),
            ("B", "maybe later", [], "deferring — rejection skips context"),
            ("A", "did you see the game last night", [], "topic switch to sports"),
            ("B", "bro that ending was insane", [], "talking about game"),
            ("A", "fr fr", [], "agreeing, slang"),
            ("B", "anyway im hungry now", [], "back to food topic"),
            ("A", "dominos?", ["food_order"], "suggesting food with hunger context"),
            ("B", "yeah order it", ["food_order"], "confirming order"),
        ],
    },

    # ── FLOW 4: Money splitting after dinner ──
    {
        "name": "Post-dinner bill split",
        "turns": [
            ("A", "that dinner was so good", [], "commenting on dinner"),
            ("B", "fr the pasta was insane", [], "agreeing"),
            ("A", "bill was 120", [], "stating bill amount"),
            ("B", "split it?", ["money"], "suggesting split with bill context"),
            ("A", "yeah venmo me 60", ["money"], "requesting venmo"),
            ("B", "sent", ["money"], "confirming sent"),
            ("A", "got it thanks", [], "acknowledging receipt"),
        ],
    },

    # ── FLOW 5: Group hangout planning — multiple intents across convo ──
    {
        "name": "Full hangout plan — movie + food + ride",
        "turns": [
            ("A", "lets do something tonight", [], "opening"),
            ("B", "movie?", ["video"], "suggesting movie"),
            ("A", "yeah what's playing", ["video"], "asking about movies — video carries from context"),
            ("B", "oppenheimer at amc at 9", ["video", "tickets"], "specific movie + theater"),
            ("A", "down, but lets eat first", ["video", "tickets", "calendar"], "movie context + time triggers calendar"),
            ("B", "chipotle before?", ["food_order"], "suggesting restaurant"),
            ("A", "sure, uber there at 7?", ["ride", "food_order"], "ride + food context bleeds"),
            ("B", "bet", ["ride"], "confirming ride"),
        ],
    },

    # ── FLOW 6: Someone joins late — no context carryover ──
    {
        "name": "Late joiner shouldn't inherit old context",
        "turns": [
            ("A", "order pizza from dominos", ["food_order"], "ordering food"),
            ("B", "got it ordering now", ["food_order"], "confirming order"),
            ("A", "nice", [], "pure filler — context skipped"),
            ("B", "done, 30 min delivery", ["food_order"], "delivery update with food context"),
            ("C", "hey whats up", [], "new person joins — greeting is pure filler"),
            ("A", "nm just ordered pizza", [], "past tense — not a new order request"),
            ("C", "nice", [], "pure filler — context skipped"),
        ],
    },

    # ── FLOW 7: Rapid topic switching ──
    {
        "name": "Rapid topic switches",
        "turns": [
            ("A", "remind me to call mom at 5", ["alarm", "contact"], "alarm + contact"),
            ("B", "also venmo me 20 for lunch", ["money", "alarm"], "money + alarm bleeds from context"),
            ("A", "sent", ["money"], "confirming venmo"),
            ("B", "thanks, also whats the weather tmrw", ["weather"], "switching to weather"),
            ("A", "supposed to rain", ["weather"], "weather context carries forward"),
            ("B", "ugh, uber to work then", ["ride", "maps"], "ride + maps for destination"),
        ],
    },

    # ── FLOW 8: Corrections and changes ──
    {
        "name": "Changing plans mid-convo",
        "turns": [
            ("A", "uber to the mall", ["ride"], "initial ride request — maps borderline"),
            ("B", "actually lets go to target instead", ["ride", "maps"], "changing destination"),
            ("A", "wait nvm lets just stay home", [], "canceling — context skipped"),
            ("B", "lol ok", [], "acknowledging cancel"),
            ("A", "order food then?", ["food_order"], "new plan"),
            ("B", "yeah doordash something", ["food_order"], "confirming food order"),
        ],
    },

    # ── FLOW 9: Mixed language (Hinglish) group chat ──
    {
        "name": "Hinglish group chat",
        "turns": [
            ("A", "bro kahan hai tu", [], "asking where are you"),
            ("B", "ghar pe, kyun", [], "at home, why"),
            ("A", "chal bahar chalte hain", ["ride", "maps"], "lets go out — model detects going out"),
            ("B", "uber karde", ["ride"], "book an uber"),
            ("A", "ok 10 min mein", ["ride", "maps"], "ride + maps from uber context"),
            ("B", "khana bhi khayenge?", ["alarm"], "time context '10 min' triggers alarm"),
            ("A", "haan dominos order karde", ["food_order"], "yes order dominos"),
        ],
    },

    # ── FLOW 10: Long convo where early context fades ──
    {
        "name": "Context should fade over many messages",
        "turns": [
            ("A", "venmo me 20 for pizza", ["money"], "money request"),
            ("B", "ok", ["money"], "ok is confirmation — carries money"),
            ("A", "what are you up to tonight", [], "topic switch"),
            ("B", "nothing much, you?", [], "chatting"),
            ("A", "thinking about going out", [], "chatting"),
            ("B", "where to", ["maps", "food_order"], "maps + food_order from 'going out' context"),
            ("A", "maybe a bar downtown", ["reservation"], "bar = venue → reservation"),
            ("B", "sounds good", ["maps"], "confirmation carries maps from context"),
        ],
    },

    # ── FLOW 11: Confirming vs just chatting — disambiguation ──
    {
        "name": "Yeah/sure disambiguation",
        "turns": [
            ("A", "should we uber to the party", ["ride"], "suggesting ride"),
            ("B", "yeah", ["ride"], "confirming ride — should fire"),
            ("A", "that party was fun last time", [], "reminiscing"),
            ("B", "yeah", [], "agreeing about party — should NOT fire ride"),
            ("A", "wanna order food after", ["food_order"], "suggesting food"),
            ("B", "sure", ["food_order"], "confirming food — should fire"),
            ("A", "the pizza from last time was good", [], "reminiscing about food"),
            ("B", "sure was", [], "agreeing — should NOT fire food"),
        ],
    },

    # ── FLOW 12: Weekend planning with friends ──
    {
        "name": "Weekend planning — calendar + reservation + ride",
        "turns": [
            ("A", "saturday plans?", [], "asking about weekend"),
            ("B", "brunch?", ["reservation"], "meal suggestion → model fires reservation"),
            ("A", "yes!! where", ["reservation"], "confirmation carries reservation"),
            ("B", "that new place on 5th", ["reservation"], "suggesting specific restaurant"),
            ("A", "book a table for 4?", ["reservation"], "explicit reservation request"),
            ("B", "done, 11am", ["reservation"], "reservation confirmation with context"),
            ("A", "perfect, ill uber there", ["ride"], "planning ride"),
            ("B", "pick me up on the way?", ["ride"], "asking for shared ride"),
        ],
    },

    # ── FLOW 13: Work context — no personal intents ──
    {
        "name": "Work chat should not fire personal intents",
        "turns": [
            ("A", "hey did you finish the report", [], "work question"),
            ("B", "yeah sent it to the team", [], "work answer, not money 'sent'"),
            ("A", "the client meeting is at 3", ["calendar"], "meeting = calendar intent"),
            ("B", "ill be there", ["calendar"], "confirming meeting attendance"),
            ("A", "also can you review the budget", [], "work request — no actionable intent"),
            ("B", "on it", [], "pure filler — context skipped"),
        ],
    },

    # ── FLOW 14: Emergency / urgent chain ──
    {
        "name": "Urgent situation — ride + contact",
        "turns": [
            ("A", "yo my car broke down", [], "stating problem"),
            ("B", "where are you", [], "asking location"),
            ("A", "stuck on highway 101", ["maps"], "stating location → maps intent"),
            ("B", "uber home?", ["ride", "maps"], "uber + location context"),
            ("A", "yeah and call AAA for the car", ["ride"], "model carries ride from context"),
            ("B", "on it, sending you an uber now", ["ride"], "confirming uber"),
        ],
    },

    # ── FLOW 15: Sarcasm and jokes shouldn't trigger ──
    {
        "name": "Sarcastic / joking conversation",
        "turns": [
            ("A", "im so broke lol", [], "joking about being broke"),
            ("B", "same, venmo me a million dollars", ["money"], "model cant detect sarcasm"),
            ("A", "lmaooo bet", ["money"], "model carries money from venmo context"),
            ("B", "for real tho wanna grab food", ["food_order"], "actual food suggestion"),
            ("A", "only if you're paying", ["food_order"], "food context bleeds from 'grab food'"),
            ("B", "im literally broke we just said that", [], "callback to joke"),
        ],
    },
]


def run_conversations():
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print(f"\n{'='*70}")
    print("MULTI-TURN CONVERSATION STRESS TEST")
    print(f"  (using smart context accumulator)")
    print(f"{'='*70}\n")

    total_pass = 0
    total_fail = 0

    for convo in CONVERSATIONS:
        name = convo["name"]
        turns = convo["turns"]
        convo_pass = 0
        convo_fail = 0
        convo_failures = []

        # Fresh accumulator per conversation (simulates real chat room)
        ctx = ConversationContext(window_size=5, stale_seconds=1800)
        chat_id = name

        for i, (sender, text, expected, desc) in enumerate(turns):
            # Get smart context string (filters filler, handles intents, max 2)
            # Pass current text so accumulator can skip context for rejections/filler
            context = ctx.get_context_string(chat_id, current_text=text)
            fired, scores = predict(text, context=context)

            # Record the turn so accumulator tracks it
            ctx.record(chat_id, text, sender=sender, fired=fired)

            exp_set = set(expected)
            fire_set = set(fired)

            if fire_set == exp_set:
                convo_pass += 1
            else:
                convo_fail += 1
                extra = fire_set - exp_set
                missed = exp_set - fire_set
                convo_failures.append((i, sender, text, desc, expected, fired, extra, missed, scores, context))

        total_pass += convo_pass
        total_fail += convo_fail

        total_turns = convo_pass + convo_fail
        status = "PASS" if convo_fail == 0 else "FAIL"
        print(f"  [{status}] {name}: {convo_pass}/{total_turns}", end="")
        if convo_fail > 0:
            print(f" ({convo_fail} failures)")
            for idx, sender, text, desc, expected, fired, extra, missed, scores, ctx_str in convo_failures:
                exp_str = ", ".join(expected) if expected else "(none)"
                fire_str = ", ".join(fired) if fired else "(none)"
                detail = ""
                if extra:
                    detail += f" | EXTRA: {', '.join(extra)}"
                if missed:
                    detail += f" | MISSED: {', '.join(missed)}"
                relevant = set(expected) | set(fired)
                score_str = " ".join(f"{k}={scores[k]}" for k in sorted(relevant) if k in scores)
                ctx_display = f"'{ctx_str}'" if ctx_str else "(none)"
                if ctx_str and len(ctx_str) > 80:
                    ctx_display = f"'...{ctx_str[-77:]}'"
                print(f"        turn {idx} [{sender}] '{text}' ({desc})")
                print(f"          expected: {exp_str} | fired: {fire_str}{detail}")
                if score_str:
                    print(f"          scores: {score_str}")
                print(f"          context: {ctx_display}")
        else:
            print()

    grand_total = total_pass + total_fail
    print(f"\n{'='*70}")
    print(f"TOTAL: {total_pass}/{grand_total} passed ({total_pass/grand_total*100:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_conversations()
