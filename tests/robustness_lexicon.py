"""Phrase lists for the robustness suite, and the split that keeps training honest.

Every list is split ONCE, deterministically, into
  TRAIN     phrasings a training-data generator may use
  HELDOUT   phrasings reserved for measurement only
Phrasings seen in real dogfood traffic are forced into HELDOUT: a model only gets credit
for them if it generalises, and real chat text never becomes training data.

The split is stratified BY PATTERN FAMILY (see `family`), not by a plain hash of the
phrase. A plain hash put every "number then symbol" amount ("10$", "10₹", "10 rs") in
HELDOUT while TRAIN had only "$10"-style prefixes - so the model could not have learned
that a symbol may follow the number, and the suite would have been measuring a hole in the
training data rather than the model's ability to generalise. Within each family the split
is still by hash, so adding a phrase never reshuffles the others.

Labels follow FIRING_RULE.md, including Gowtham's 2026-09-13 and Akash's 2026-09-15 rulings.
English only.
"""
import hashlib
import re

HELDOUT_SHARE = 0.4

# Seen in dogfood logs 2026-08/09 - always held out.
FROM_DOGFOOD = {
    "oh ok", "yep fr", "why not", "sure thing homie", "ok sure", "yes please, thanks",
    "sure, how much do you need?", "sure where would you like to be picked up?",
    "can you give me {amt}", "can you spot {amt} till then?", "hi can you send me {amt}",
    "shall i paypal you the amount?", "can i pay you {amt}", "i’ll transfer it now ?",
    "would you like me to proceed with the booking?", "{n}₹", "ofcourse i gotchu homie",
    # from the batteries and the 82-case set (real rooms / ruling examples) - test only
    "i'll send it", "i'll be sending you that dollar that i owe you", "okay i'll get you a cab for it then",
    "you owe me {amt}", "im short {amt} this month", "im sorry, will send",
}

MONEY_REQUESTS = [
    "can you send me {amt}", "send me {amt}", "can you give me {amt}", "could you lend me {amt}",
    "spot me {amt}", "can you spot {amt} till then?", "can you transfer {amt} to me", "can you pay me {amt}",
    "venmo me {amt}", "gpay me {amt}", "can you upi me {amt}", "zelle me {amt} when you can",
    "i need {amt}, can you send it", "can you cover {amt} for me", "hi can you send me {amt}",
    "can u send {amt} pls", "could you help me out with {amt}", "please send me {amt} for lunch",
    "can you paytm me {amt} for the tickets", "can i borrow {amt}", "mind sending me {amt}?",
    "can you wire me {amt}", "hey can you lend me {amt} till friday", "send {amt} my way?",
]

AMOUNTS = [
    # prefix symbol/code                     number then symbol/code            number then word
    "${n}", "₹{n}", "rs {n}", "rs.{n}", "inr {n}", "USD {n}",
    "{n}$", "{n}₹", "{n} rs", "{n} inr", "{n}rs", "{n} USD", "{n}/-",
    "{n} dollars", "{n} bucks", "{n} rupees", "{n} quid",
]
NUMBERS = ["10", "20", "45", "100", "250", "500", "1,500", "2000"]

RIDE_REQUESTS = [
    "can you book me a cab to {place}", "book me an uber to {place}", "can someone get me a ride to {place}",
    "could you order me an ola from {a} to {b}", "i need a cab to {place}, can you book it",
    "please book a taxi from {a} to {b}", "can you call me an uber to {place}", "get me a lyft to {place}",
    "can you book a rapido for me to {place}", "hey can u book me a ride home", "can you get me a cab from {a}",
    "book a cab for me from {a} to {b} please", "anyone free to book me an uber to {place}?",
]
PLACES = ["the airport", "koramangala", "downtown", "the station", "indiranagar", "work", "the mall", "whitefield"]

ACCEPT = [
    # plain                                     abbreviations
    "ok", "okay", "sure", "yes", "yeah", "ya", "yep", "yup", "k", "kk", "okie", "okk", "kay",
    # commitment idioms
    "bet", "on it", "got you", "i got you", "gotchu", "say less", "can do", "will do", "i can do that",
    "yeah i can do that", "consider it done",
    # emphatic
    "of course", "ofc", "absolutely", "definitely", "for sure", "sure thing", "happy to",
    "yes of course", "no problem", "no worries", "sure np", "yeah no problem",
    # rhetorical question = agreement ("why not" is held out; these teach the shape)
    "why not", "why wouldn't i", "no reason not to", "of course i will", "why would i say no",
    # discourse marker first ("oh ok" is held out; these teach the shape)
    "oh ok", "ah ok", "well ok", "oh sure", "hmm ok", "oh yeah sure",
    # slang tail ("yep fr" / "sure thing homie" are held out; these teach the shape)
    "yep fr", "yeah fr", "ok fr", "sure thing homie", "sure thing man", "i gotchu bro",
    "sure bro", "ok bro", "yessir", "aight", "alright", "alrighty",
    # agreement with a hedge word that still commits
    "ok fine", "ok sure", "okay sure", "yes sure", "ya sure", "sure, one sec",
    "ofcourse i gotchu homie",
]
ACCEPT_MONEY_ACTION = ["sending now", "ok sending", "sure, sending", "doing it now", "sending it rn", "transferring now"]
ACCEPT_RIDE_ACTION = ["booking now", "ok booking it", "sure, booking", "booking it rn", "on it, booking now"]

# Ruling 1 (2026-09-15): agreement + a detail question fires right away.
ACCEPT_WITH_QUESTION_MONEY = [
    "sure, how much?", "sure, how much do you need?", "ok, what's the amount?", "yeah how much",
    "sure, how much exactly?", "yes, how much should i send?", "ok sure, how much is it",
    "ok how much do i send?", "yes, what's the total?", "sure, how much you need",
    "yeah, amount?", "ok, how much again?", "sure — how much is it?",
]
ACCEPT_WITH_QUESTION_RIDE = [
    "sure, where to?", "sure where would you like to be picked up?", "yeah, where are you going?",
    "ok where should i book it to?", "sure, what time?", "ok, pickup from where?",
]

OFFERS_MONEY = [
    "shall i send you the money now?", "should i send you {amt}?", "want me to send you {amt}?",
    "can i pay you {amt}", "i’ll transfer it now ?", "do you want me to venmo you?", "shall i paypal you the amount?",
    "i can send it now if you want", "let me send you {amt}", "should i transfer the {amt} now?",
]
OFFERS_RIDE = [
    "should i book the cab for you?", "want me to book you an uber?", "would you like me to proceed with the booking?",
    "shall i get you a cab?", "i can book you a ride if you want", "should i order an ola for you?",
]
OFFER_ACCEPT = [
    "yes please", "yes", "sure", "please do", "that would help", "go ahead", "yeah do it", "ok",
    "if you don't mind", "that'd be great", "yes pls", "yes please, thanks", "sure go ahead", "yeah please",
]

INTERRUPT_BY_ACCEPTOR = ["hi", "hey", "hmm", "one sec", "hi yash", "oh hey"]
INTERRUPT_BY_REQUESTER = ["i need it urgently", "will pay you back friday", "if it's possible", "thanks in advance", "hello?"]

# Must stay quiet after a real request.
DECLINE = [
    "can't right now", "sorry i'm broke", "nah", "not today", "i don't have it", "no",
    "maybe later", "ask someone else", "sorry can't", "not possible rn", "nope", "i'm short too",
]
QUESTION_ONLY_MONEY = ["for what?", "why?", "what's it for?", "how much?", "what happened?", "why do you need it?"]
QUESTION_ONLY_RIDE = ["where are you going?", "why can't you book it?", "what time?", "where are you now?"]
FUTURE = ["i'll send it friday", "will pay you next week", "i'll book it tomorrow morning", "i'll do it after work",
          "let me check my balance first", "i'll send once i get paid"]
DONE = ["sent it already", "just paid you", "already booked it", "done, sent", "booked it for you already", "paid"]
HEDGE = ["maybe i should pitch in", "i might be able to", "not sure i can", "let me see", "possibly later", "i'll think about it"]
# Look like agreement, are not a commitment. The guard against pushing recall too far.
HARD_NEGATIVE = [
    "yeah no", "sure, if i had money lol", "ok but not now", "sure, next week", "i would but i can't",
    "haha no", "ok, ask someone else", "sure... after i get paid", "oh ok, never mind then",
    "why not ask someone else", "lol no", "ok maybe tomorrow", "yeah right", "sure, in your dreams",
    "okay but i'm broke rn", "yes but not today", "alright, i'll see later", "ok let me think",
]
REQUESTER_FOLLOWUP =["ok", "please", "i need it urgently", "thanks", "hello?", "please reply", "you there?"]
NON_REQUEST_OPENERS = [
    "i'm heading out now", "the movie was great", "traffic is crazy today", "just finished work",
    "the dinner place was packed", "my phone is almost dead", "it's raining here",
]

# ---- added 2026-09-17 for v17b: behaviours v17 lost, and Akash's 2026-09-17 rulings ----
# Every phrase here is DISTINCT from the lists above: the held-out set is global, so reusing
# a phrase ("sure", "ok") would let a new list move an old phrase between TRAIN and HELDOUT
# and silently change every earlier baseline. Bare words appear in the suite as fixed context.

# FIRING_RULE s3: in progress, no request needed - fires. (Offers and future stay quiet.)
SELF_SEND = [
    "i'll send it", "i'll be sending you that dollar that i owe you", "sending you the {amt} now",
    "im sending you {amt} now", "transferring the {amt} to you now", "paying you back now",
    "sending you the money i owe you", "i will transfer {amt} to you", "sending over the {amt} i owe you",
    "i'm paying you now", "gonna send it now", "sending the rent now", "i'll pay you the {amt} now",
    "ok i'll send the money", "i'll transfer it to you now", "here, sending it now",
]
SELF_BOOK = [
    "okay i'll get you a cab for it then", "booking your cab now", "i'll book you an uber now",
    "getting you an ola now", "im booking you a cab to {place}", "i'm ordering your uber now",
    "i'll get you a cab to {place}", "booking the taxi for you now", "ordering you a rapido now",
]
# Ruling 8 (2026-09-17): a debt stated to the debtor asks for the money.
DEBT_STATEMENTS = [
    "you owe me {amt}", "you still owe me {amt}", "dont forget you owe me {amt}", "u owe me {amt} bro",
    "you owe me {amt} for the tickets", "remember you owe me {amt}", "you still haven't paid me the {amt}",
    "hey you owe me {amt} from last week", "you never paid me back the {amt}",
]
DEBT_PAY_AGREE = [
    "yeah let me pay", "oh right, sending now", "my bad, sending it now", "ok paying now",
    "yes will send it now", "oops sorry, transferring now", "ok ok sending it", "sure, paying you now",
    "right, sending it over",
]
DEBT_ADMIT = [
    "yeah i know", "i remember", "i know i know", "i haven't forgotten", "noted", "yes i'm aware",
    "yeah yeah i know", "don't worry i remember",
]
# FIRING_RULE s1a: a shortfall is not a request; a polite noise after it fires nothing.
SHORTFALL = [
    "im short {amt} this month", "rent is due and im {amt} short", "im broke till payday",
    "i only have {amt} left for the month", "ugh im short on cash this week", "i'm {amt} short for the trip",
    "my card got declined again", "money's really tight right now",
]
SHORTFALL_POLITE = ["damn", "that sucks", "oh no", "rip", "same here", "i feel you", "that's rough", "ugh sorry man"]
SHORTFALL_COMMIT = [
    "im sorry, will send", "sending you {amt} now", "i got you, sending some now",
    "here, i'll send you {amt} right now", "let me help, sending {amt}", "i'll transfer you {amt} now",
]
# FIRING_RULE s3c: an offer-shaped reply that ANSWERS a request is a commitment and fires.
# Ruling 6/7 (2026-09-17): after such a clear take, a third person's bare ok is quiet.
CLEAR_TAKE_MONEY = [
    "i'll send it right now", "let me send it", "on it, sending now", "i'll transfer it right away",
    "ok i'll pay it", "i'll cover it, sending now", "let me pay it",
]
CLEAR_TAKE_RIDE = [
    "let me book a cab", "i'll book it", "on it, getting the cab", "let me get you an uber",
    "i'll book one for you", "let me order the ola", "i'll book it right now",
]
THIRD_EXPLICIT = ["i can book one too", "i can send it too", "i can also book one", "i'll send it as well",
                  "i can pay too", "i'll book one as well"]


def _hash_share(phrase):
    return int(hashlib.sha1(phrase.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def family(phrase):
    """The PATTERN a phrase belongs to. The split runs inside each family, so training
    always carries the shape a held-out phrase is testing."""
    p = phrase.lower()
    if "{n}" in p:                                        # amount formats
        if re.match(r"^[$₹]|^(rs\.?|inr|usd)\s*\{n\}", p):
            return "amt_prefix"
        if re.search(r"\{n\}\s*([$₹]|rs|inr|usd|/-)", p):
            return "amt_postfix_symbol"
        return "amt_postfix_word"
    if "{amt}" in p or "{place}" in p or "{a}" in p:       # request / offer templates
        if re.match(r"^(shall|should|want me|would you|can i|do you want|i can|let me|i’ll|i'll)", p):
            return "offer"
        return "request"
    if re.search(r"\b(why not|why wouldn'?t|no reason not|of course i will|why would i say no)\b", p):
        return "ack_rhetorical"
    if re.match(r"^(oh|ah|well|hmm)\b", p):
        return "ack_discourse_marker"
    if re.search(r"\b(fr|bro|homie|man|sir|dude)\b", p) or p.endswith("sir"):
        return "ack_slang_tail"
    if re.search(r"\b(how much|amount|total|what time|where|which|pickup|address)\b", p):
        return "ack_with_question"
    if len(p.split()) <= 2 and re.match(r"^[a-z]{1,5}$", p.replace(" ", "")):
        return "ack_abbrev"
    return "other"


def _base_seen():
    """Phrasings the BASE corpus already contains - they cannot test generalisation.
    Written by data_gen/lexicon_base_seen.py; absent file just means no exclusions."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data/eval/lexicon_base_seen.json"
    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _compute_heldout():
    seen = _base_seen() - FROM_DOGFOOD     # dogfood phrasings stay held out regardless
    held = set(FROM_DOGFOOD)
    groups = {}
    for name, items in LISTS.items():
        for p in items:
            groups.setdefault((name, family(p)), []).append(p)
    for (_, _), items in groups.items():
        ranked = [p for p in sorted(set(items), key=_hash_share) if p not in seen]
        n_held = max(1, round(len(ranked) * HELDOUT_SHARE)) if len(ranked) > 1 else 0
        # dogfood phrasings already count towards the held-out quota of their family
        forced = [p for p in ranked if p in FROM_DOGFOOD]
        for p in ranked:
            if len([x for x in ranked if x in held]) >= max(n_held, len(forced)):
                break
            if p not in held:
                held.add(p)
    return held


LISTS = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, list)}
HELDOUT = _compute_heldout()


def is_heldout(phrase):
    return phrase in HELDOUT


def split(items):
    return [p for p in items if p not in HELDOUT], [p for p in items if p in HELDOUT]
