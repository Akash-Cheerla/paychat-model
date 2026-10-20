"""What happened in this room, for a window of it - not one message at a time.

Per-message cards reach the long messages and nothing else. On real traffic that is about
3% of what gets said, while another ~10% carries a deadline, an owner, a decision or a
problem in messages too short to ever qualify. The clearest case in the logs is dm_13_70
on 2026-08-30: 323 messages, the whole of which is a money-and-ride thread - 500 revised
down to 300, metro stuck, a lift falling through, "literally at 200 until Friday", "need
to leave by 4" - and the Summary tab shows NOTHING, because no single message is long
enough. That is the gap this closes.

ONE thing changes against the shipped per-message summariser: the unit. Same scoring, same
vocabulary, same chatter rules, applied to every message in a window rather than to single
long messages. Nothing in summarizer.py is modified - this reads it.

The engine is DELIBERATELY the rules, not the student, which is the opposite of the
per-message split. Measured 2026-10-16 on three busy room-days, the student on a day
window failed three different ways:

    dm_13_70  "Meeting Rescheduled", "First/Second/Third meeting canceled"
              - no meeting exists anywhere in that conversation
    group_12  started on the real topic then drifted into unrelated traffic
              from later in the day
    group_12  thirteen fragments with no selection - a dump, not a summary

Two causes, neither fixable by prompting: every one of those days is 1,770-2,467 tokens
against the student's 512-token window, so it sees about the first fifth and generates the
rest from its priors; and it was distilled on single message -> card, so a multi-speaker
window is a different object it was never shown.

The rules cannot do either of those things. Every point is a span of something somebody
actually said, carried with its sender and timestamp, so the fabrication rate is zero by
construction rather than by measurement. A model-written version needs (window -> card)
teacher pairs and a longer context, which is a different base model, not a prompt change.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

import sensitive
import summarizer as S

# A day of a working group repeats itself. The dogfood log is the extreme - "Can you book
# cab for me" fifteen times - but every real room does it.
NEAR_DUPLICATE = 0.75

# The same bar a per-message card has to clear, so the two features agree about what
# counts. Below it, a point is an observation rather than something to remember.
POINT_THRESHOLD = 2

# A window with nothing in it gets no card at all, the same way a short message does not.
# A recap is for a window with something in it. Below this the window draws nothing
# rather than a card of one or two lines - Akash, 2026-10-19: "cant keep 1-2 lines and
# for each small topic and just 1-3 points we can have sumary for each, thats not
# professional". The published guidance agrees: recaps are built for busy windows, and a
# quiet one is not expected to produce one.
MIN_POINTS = 4

# A ceiling, not a target. The count follows the content - the median window produces
# three points and a busy day produces twelve - but the ceiling has to be low, because
# the selection is score-ordered and the tail is chatter. At 18 the freed slots recruited
# "Would you rather live near a beach or mountain?" and "Hang in there" into dm_13_70;
# at 12 that day is the money-and-ride thread and nothing else. A day summary longer than
# about twelve lines has stopped being a summary anyway.
MAX_POINTS = 12

# Below this many messages a window is not a conversation yet, and per-message cards
# already cover anything long enough to matter in it.
MIN_MESSAGES = 8


@dataclass
class RoomSummary:
    room: str
    points: list = field(default_factory=list)
    messages: int = 0
    window_from: Optional[str] = None
    window_to: Optional[str] = None
    # {"requests": 4, "deadlines": 2} - what kinds of thing the window holds.
    shape: dict = field(default_factory=dict)
    # [{"heading": "decisions", "points": [...]}, ...] - what a recap is read for,
    # kept apart so the reader does not have to interpret it.
    cards: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"room": self.room, "messages": self.messages,
                "window_from": self.window_from, "window_to": self.window_to,
                "shape": dict(self.shape),
                "sections": [{"heading": c["heading"],
                              "points": [dict(p) for p in c["points"]]}
                             for c in self.cards],
                # The same points as one flat chronological list, for a caller that
                # wants the conversation's own order rather than per-thread cards.
                "points": [dict(p) for p in self.points]}


def _content(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", (text or "").lower())
            if w not in S._FUNCTION_WORDS and len(w) > 2}


def _is_near_duplicate(words: set, seen: Sequence[set]) -> bool:
    if not words:
        return True
    for prev in seen:
        if prev and len(words & prev) / max(len(words | prev), 1) >= NEAR_DUPLICATE:
            return True
    return False


# A point has to stand on its own. "Can you bring it up?", "Already did can't find it",
# "Actually we can move it?" all score the same 3 as "Ohh btw can you send me 500 ?" - they
# match the request pattern - but the reader cannot act on them without scrolling back to
# find what "it" was, which is the thing a summary exists to save them. Raising the score
# floor does not separate these (measured: identical list at 3, and 25 fewer windows get a
# summary at all); what separates them is whether the object is ever named.
#
# A point is refused when its only object is a bare pronoun. Numbers, names, @mentions and
# concrete nouns all rescue it, so "send me 500" and "I need to leave by 4" stay.
_BARE_ANAPHORA = re.compile(
    r"^(?:\W*(?:ok|okay|yes|yeah|no|actually|also|and|but|so|well|hmm|oh+|btw)\b\W*)*"
    r"(?:(?:can|could|would|will|shall)\s+(?:you|we|i|someone)|i|we|you|it|that|this|"
    r"they|he|she|already|just)\b"
    # The tail must allow the closing punctuation, or the pattern can never match a
    # question - and most of these are questions ("Can you bring it up?").
    r"[^.!?]*?\b(?:it|this|that|one|them|those|these|some|there)\b[^.!?]*[.!?]*\s*$",
    re.IGNORECASE)

# What rescues a point from the test above: anything that names the thing. Two patterns,
# because one of them must be case-SENSITIVE - a single pattern with re.IGNORECASE makes
# "[A-Z][a-z]{2,}" match any word at all, which silently rescued every point and made this
# whole filter a no-op when it was first written.
_CONCRETE = re.compile(
    r"\d|@\w+|"
    r"\b(?:cab|ride|car|taxi|money|cash|rupees?|bucks?|card|payment|bill|split|share|"
    r"build|app|screen|page|button|link|doc|file|log|test|release|branch|merge|deploy|"
    r"server|model|api|bug|crash|address|location|home|office|airport|station|metro|"
    r"flight|train|lunch|dinner|meeting|call|demo|ticket|issue)\b",
    re.IGNORECASE)

# A proper noun names the thing too - but only away from the start of the sentence, where
# every point begins with a capital ("Can", "Actually", "Already").
_PROPER_NOUN = re.compile(r"(?!^)(?<=[a-z,]\s)[A-Z][a-z]{2,}")


def _stands_alone(point: str) -> bool:
    point = (point or "").strip()
    if _CONCRETE.search(point) or _PROPER_NOUN.search(point):
        return True
    return not _BARE_ANAPHORA.search(point)


# Credentials and identifiers are defined once, in sensitive.py, and shared with the
# per-message path - the audit that found them found the same leaks in both.
_SENSITIVE = sensitive.SECRET


# A point that opens on a conjunction reads as half a sentence - "But june's is still
# due", "So need something asap" - which was 5.6% of all points. The content is fine; the
# opener is not, so it is removed rather than the point refused. summarizer._LEAD_INS does
# this for per-message cards; short messages never went through it, because they never
# used to become points at all.
_OPENER = re.compile(
    r"^(?:and|but|or|so|then|also|because|cos|coz|cause|which|that|plus|anyway|"
    r"actually|basically|well|hmm+|oh+|ohh+|btw|ok|okay|yeah|yes|same|"
    # How people open a message TO the room. These arrived on cards reading as content:
    # "Guys dinner last night came to 2400 total", "Quick one - the investor demo moved
    # up", "Same, 18000 split three ways so 6000 each".
    r"guys|team|folks|everyone|quick\s+one|small\s+ask|hi\s+team|hey\s+team"
    # The lookahead is not optional. Without it "that" matches inside "that's", because
    # \b sits between the word and the apostrophe, and "that's mine, i'll look today"
    # was being rewritten to "'s mine, i'll look today" - the opener stripper corrupting
    # the text it was meant to tidy.
    r")\b(?!['’])[\s,:-]*",
    re.IGNORECASE)


# Someone's private life is not a summary point. The ask patterns cannot tell a social
# ask from a work one - "will you smoke with me?" is the same shape as "will you send me
# 500?" and scores higher - and two work-vocabulary patterns misfire on personal news:
# "My salary got delayed" scores on the "delay" rule meant for a delayed build, and
# "Can't quit fb, only snapchat" on the "cannot" rule meant for a broken feature.
#
# Found by auditing all 253 summaries the logs produce. The corpus for this is the
# anonymised WhatsApp exports (cr_/rw_/nw_ rooms), which is real human conversation and
# therefore the right population to check - PayChat is a consumer chat app, so most rooms
# will look like these rather than like group_12.
#
# Rescued when the point also carries money or a commitment, because that is the feature's
# whole purpose: "drinks came to 1200, can you send me 300" has to survive.
_PRIVATE = re.compile(
    r"\b(?:smoke|smoking|cigarettes?|cigs?|vape|weed|drunk|hangover|"
    r"period|periods|pregnan\w*|sick|ill|fever|hospital|doctor|surgery|medicine|meds|"
    r"depress\w*|anxiety|therapy|divorce|breakup|broke\s+up|girlfriend|boyfriend|"
    r"crush|dating|salary|payslip|appraisal|fired|laid\s+off|resign\w*|"
    # "they let me go on friday, still figuring out what to do" reached a card once the
    # weekday signal started firing - losing a job said in confidence, in a summary.
    r"let\s+(?:me|him|her|them)\s+go|letting\s+(?:me|him|her|them)\s+go|"
    r"made\s+redundant|notice\s+period|severance)\b",
    re.IGNORECASE)

# What rescues a point from the test above - a transaction or a commitment, which is what
# the feature exists to surface.
_ACTIONABLE = re.compile(
    r"[$£€₹]\s?\d|\b\d+\s?(?:rs|rupees?|bucks?|dollars?)\b|"
    r"\b(?:send|sent|pay|paid|transfer|owe|owes|split|share|cab|ride|book(?:ed|ing)?|"
    r"pickup|pick\s+up|drop)\b",
    re.IGNORECASE)


def _is_private(point: str) -> bool:
    return bool(_PRIVATE.search(point or "")) and not _ACTIONABLE.search(point or "")


def _tidy(point: str) -> str:
    """Strip leading conjunctions and filler, repeatedly, then recapitalise."""
    text = (point or "").strip()
    for _ in range(4):
        stripped = _OPENER.sub("", text, count=1).strip()
        if stripped == text or len(stripped.split()) < 3:
            break
        text = stripped
    return text[:1].upper() + text[1:] if text else text


# What something cost is the premise of every split, and it scored 1 - below the bar.
# "dinner last night came to 2400 total", "dinner was 208 total", "drinks came to 1200",
# "the bill is 2400" all scored only the +1 that any digit earns, so a room whose whole
# subject was a shared bill produced no summary (mt_buried, 2026-10-19). The per-message
# path never noticed because those messages are far too short to earn a card of their own,
# which is exactly the gap this feature exists to close.
#
# An amount alone is not enough - "Build 61" and "QA-116" are numbers too - so a spend
# word has to be next to it.
_SPEND_WORDS = (r"total|each|split|splits|share|shares|bill|cost|costs|came|comes|"
                r"wants|want|charges|charged|quoted|asking|asks|"
                r"owe|owes|owed|paid|pay|pays|send|sending|sent|transfer|"
                r"dinner|lunch|breakfast|drinks|cab|ride|trip|rent|fare|fees?|"
                r"ticket|tickets|groceries|petrol|fuel")
_AMOUNT = (r"[$£€₹]\s?\d|\b\d[\d,]*(?:\.\d+)?\s?(?:rs|rupees?|bucks?|dollars?|k)\b"
           r"|\b\d[\d,]{1,}")
# Either order: "dinner was 208 total" and "cab fare was 350" both count.
_MONEY_POINT = re.compile(
    rf"(?:{_AMOUNT})(?=[\s\S]{{0,40}}?\b(?:{_SPEND_WORDS})\b)"
    rf"|\b(?:{_SPEND_WORDS})\b[\s\S]{{0,40}}?(?:{_AMOUNT})",
    re.IGNORECASE)

# Enough to clear POINT_THRESHOLD on its own, since that is the whole point.
_MONEY_BONUS = 2

# summarizer.importance was built for long work messages and scores ordinary conversation
# at nothing. Measured on realistic rooms (2026-10-19) it dropped every one of these:
#
#   0  the investor demo moved up            0  the kitchen tap is leaking again
#   1  it's thursday 2pm now instead ...     0  i'll call the plumber
#   1  rent is due on the 5th                1  also the plumber wants 600 for the visit
#   0  the electricity bill came too         1  2400 for last month
#
# A flatmates' rent-and-plumber conversation produced 2 points from 26 messages. The
# patterns want "by Friday", "blocker", "crash", "QA-116" - work vocabulary - and a
# consumer chat app is mostly not that. These are room-path only; the per-message card is
# untouched, and it never saw these lines anyway because they are far too short to card.
_ROOM_SIGNALS = [
    # A date or time in any of the shapes conversation actually uses. The shipped pattern
    # needs by/before/due IMMEDIATELY followed by a digit, so "due on the 5th" and a bare
    # "thursday 2pm" both scored nothing.
    (3, re.compile(r"\b(?:due|deadline|by|before|on|from)\s+(?:the\s+)?"
                   r"(?:\d{1,2}(?:st|nd|rd|th)?\b|mon|tues|wednes|thurs|fri|satur|sun)",
                   re.IGNORECASE)),
    (3, re.compile(r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b"
                   r"(?:\s+(?:morning|afternoon|evening|night|\d))?", re.IGNORECASE)),
    (3, re.compile(r"\b\d{1,2}\s?(?:am|pm)\b|\b\d{1,2}:\d{2}\b", re.IGNORECASE)),
    # A plan or a date that CHANGED, which is the thing people scroll back to check.
    (3, re.compile(r"\b(?:moved?|shifted|pushed|postponed|rescheduled|brought)\s+"
                   r"(?:it\s+)?(?:up|back|forward|to|from)\b|\binstead\s+of\b",
                   re.IGNORECASE)),
    # Something is wrong, in the words people use at home rather than in a bug tracker.
    (3, re.compile(r"\b(?:leak(?:s|ing|ed)?|blocked|clogged|flooded|burst|jammed|"
                   r"not\s+work(?:ing)?|stopped\s+work(?:ing)?|broke(?:n|\s+down)?|"
                   r"dirty|smell(?:s|ing|y)?|noisy|too\s+(?:high|low|slow|hot|cold)|"
                   # how a blank or missing field gets reported, which scored nothing:
                   # "the login screen is blank on android too"
                   r"blank|empty|missing|greyed\s+out|stuck\s+on|"
                   r"keeps?\s+\w+ing)\b", re.IGNORECASE)),
    # Somebody taking something on. "I'll call the plumber" is the single most useful
    # line in that whole conversation and it scored zero.
    (3, re.compile(r"\b(?:i'?ll|i\s+will|we'?ll|we\s+will|let'?s|let\s+me)\s+"
                   r"(?:call|book|order|arrange|sort|handle|take|do|send|pay|check|ask|"
                   r"tell|get|fix|bring|collect|organise|organize|"
                   # "let's sync at 3 tomorrow" is the commonest way a team commits to a
                   # time and it scored 1
                   r"sync|meet|catch\s+up|review|look|start|finish|ship|deploy|discuss)\b",
                   re.IGNORECASE)),
    # A release or a version being available, which is how this team hands work over.
    (2, re.compile(r"\b(?:is\s+)?(?:up|live|out|released|shipped|deployed|available)\s+"
                   r"(?:on|in|at)\b|\bon\s+(?:testflight|play\s*store|app\s*store)\b",
                   re.IGNORECASE)),
    # A bill or a cost without a spend verb next to it - "2400 for last month".
    (2, re.compile(r"\b(?:rent|bill|bills|electricity|water|gas|wifi|internet|"
                   r"maintenance|deposit|fees?)\b", re.IGNORECASE)),
]


# Money and ride REQUESTS do not belong in a summary - the Intent Log already has them,
# and duplicating them is what made the first version read like a transcript. Akash,
# 2026-10-19: "regarding intents we dont need them in summaries we already have intent
# logs, summaries are for something that is important or we have many discussions or info
# on".
#
# The line is request versus fact. "can you send me 500" and "book me a cab to the office"
# are intents and come out. "dinner last night came to 2400 total" and "18000 split three
# ways so 6000 each" are information somebody will scroll back for and stay - the intent
# log records that a sheet opened, not what the bill was or how it was divided.
_INTENT_SHAPED = re.compile(
    r"^(?:\s*(?:ok|okay|yes|also|and|but|so|hey|hi|btw|actually|please)\b[\s,]*)*"
    r"(?:(?:can|could|would|will|cud)\s+(?:you|u|someone|anyone|somebody)\b"
    r"|(?:gimme|give\s+me|send\s+me|lend\s+me|transfer\s+me|pay\s+me)\b"
    r"|(?:i'?ll|i\s+will|let\s+me)\s+(?:send|transfer|pay|book|order)\b"
    r"|(?:sending|transferring|paying|booking)\b"
    r"|(?:book|order)\s+(?:me|us|a|an)\b)",
    re.IGNORECASE)

# ... unless the line also carries something the intent log does not: what it cost, how it
# was split, who is owed, or a date attached to it.
_INTENT_EXCEPTION = re.compile(
    r"\b(?:total|each|split|ways|share|shares|net|owes?|owed|owe|due|rent|bill)\b"
    r"|\b(?:by|before)\s+(?:\d|mon|tues|wednes|thurs|fri|satur|sun)",
    re.IGNORECASE)


def _is_intent(point: str) -> bool:
    """Is this the Intent Log's job rather than the summary's?"""
    text = (point or "").strip()
    return bool(_INTENT_SHAPED.search(text)) and not _INTENT_EXCEPTION.search(text)


# What a recap is actually read for. The published guidance on summary products is
# consistent about this: decisions, action items with owners, and open questions, kept
# apart so the reader does not have to interpret the thing. A flat list of quoted lines is
# a record rather than a tool, which is what the first version was.
_SECTIONS = [
    ("decisions", re.compile(
        r"\b(?:decided|agreed|approved|rejected|confirmed|going\s+with|sticking\s+with|"
        r"final(?:ised|ized)?|signed\s+off|locked|"
        r"moved?\s+(?:to|up|back|forward)|instead\s+of|rescheduled|postponed|cancelled|"
        r"is\s+due\b|is\s+now\b)", re.IGNORECASE)),
    ("action items", re.compile(
        r"\b(?:let'?s\s+\w+|"
        r"i'?ll|we'?ll|i\s+will|we\s+will|let\s+me|please|needs?\s+to|"
        r"has\s+to|have\s+to|must|will\s+(?:check|call|tell|ask|do|send|fix)|"
        r"taking|owns?|assigned|pull\s+it|verify)\b", re.IGNORECASE)),
    ("open questions", re.compile(
        r"\?\s*$|^\s*(?:is|are|does|do|did|can|could|should|who|what|when|where|why|how)\b",
        re.IGNORECASE)),
]


def _section_of(point: str) -> str:
    for name, rx in _SECTIONS:
        if rx.search(point or ""):
            return name
    return "information"


def _candidates(message: dict):
    """The point(s) one message contributes.

    A message long enough for its own card is cut into that card's points first, so a
    nine-sentence status update contributes its deadline rather than its throat-clearing.
    Everything else is scored whole - which is the entire point of this feature, because
    "QA-116 still fails" is four words and will never earn a card of its own.
    """
    text = (message.get("text") or "").strip()
    if not text:
        return []
    if S.is_worth_summarizing(text):
        card = S.summarize(text)
        return list(card.points) if card else []
    return [S._tidy_point(text)]


# NO TITLE, and this is a finding rather than an omission.
#
# Measured over 255 room-day summaries from every log:
#
#   summarizer.title_for, as the prototype used it          88.2% weak
#     "Key Points" x91, "Bug" x47, "Payment" x30, "Call" x7
#   the same read from the highest-scoring points instead   88.6% weak
#     which fixed the two examples that prompted it, "Stuck" and "Call",
#     and moved the aggregate the wrong way
#   keeping only titles that pass a specificity test        12.9% named
#     and those were "Karam It's", "Book I'll", "Friends Right",
#     "Account Paypal" - worse than the generic ones
#
# title_for is earliest-topic-keyword-wins, which is right for a message, where what it
# opens with is what it is about. A multi-speaker window has no such opening, so the
# function falls through to its generic fallback or latches onto a stray word pair. No
# ordering or filter fixes that; a usable day title would have to be generated, which
# means the model, which is the thing that fabricated "Meeting Rescheduled".
#
# A window does not need an invented subject anyway: the room and the dates already say
# what it is, and `shape` below says whether it is worth opening. That is what Zoom and
# Teams show too - a recap and a date, not a guessed topic.


# What kinds of thing the window contained. Factual, so it is never wrong about the
# window the way a guessed topic is - "4 requests, 2 deadlines" tells the reader whether
# to open it, which is all a heading is for.
# Order matters: the first match wins, so the most specific category comes first. Money
# leads because it is what this product is for - a shared bill is the premise of every
# split, and without a money category "Dinner last night came to 2400 total" fell through
# to no category at all (measured on the four-topic room, 2026-10-19).
_SHAPE = [
    ("money", _MONEY_POINT),
    ("deadlines", re.compile(r"\b(?:by|before|due|until|no\s+later\s+than)\s+"
                             r"(?:\d|mon|tue|wed|thu|fri|sat|sun|today|tomorrow|tonight|"
                             r"eod|noon|midnight|next\s+\w+)|\b(?:eod|asap|deadline)\b|"
                             # "the demo moved to Thursday 2pm" is a deadline too
                             r"\bmoved?\s+(?:to|up\s+to)\s+"
                             r"(?:\d|mon|tue|wed|thu|fri|sat|sun|next\s+\w+)|"
                             r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\s+\d",
                             re.IGNORECASE)),
    ("requests", re.compile(r"\b(?:can|could|would|will)\s+(?:you|someone|anyone|we)\b|"
                            r"\bplease\b|@\w+|\bgimme\b|\bneed\s+(?:you|someone)\b",
                            re.IGNORECASE)),
    ("problems", re.compile(r"\b(?:crash(?:es|ed|ing)?|bugs?|broken|fail(?:s|ed|ing)?|"
                            r"errors?|not\s+working|stuck|wrong|missing|issues?|"
                            r"can(?:not|n't|'t))\b|"
                            # how a blank field gets reported, which had no category
                            r"\b(?:blank|empty|does\s?n'?t|do\s?n'?t|wo\s?n'?t|"
                            r"no\s+log|dropped|silently)\b", re.IGNORECASE)),
    ("decisions", re.compile(r"\b(?:assigned|approved|rejected|merged|released|deployed|"
                             r"agreed|decided|confirmed|cancelled|postponed|fixed|"
                             r"going\s+with|sticking\s+with)\b", re.IGNORECASE)),
]


def _kind_of(point: str) -> Optional[str]:
    """Which of the shape categories this point belongs to, strongest first.

    Carried on every point so a client can GROUP by it. Points come out in timestamp
    order, which is deliberate - a money thread reads "send me 500" then "actually send
    300" and reordering it loses the correction - but in a room running several threads at
    once that means the list switches subject almost every line (measured: 10 of 11
    switches on a four-topic room). Grouping is a presentation choice, so the category
    goes in the payload and the client decides.
    """
    for name, rx in _SHAPE:
        if rx.search(point or ""):
            return name
    return None


def shape(points: Sequence[dict]) -> dict:
    """How many of each kind of point the window holds, strongest category first."""
    counts = {}
    for name, rx in _SHAPE:
        n = sum(1 for p in points if rx.search(p["text"]))
        if n:
            counts[name] = n
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def summarize_room(messages: Sequence[dict], room: str = "",
                   limit: int = MAX_POINTS) -> Optional[RoomSummary]:
    """The things worth remembering from one window of one room, oldest first.

    `messages` is the window the caller wants summarised, each {"text", "sender", "ts"}.
    Returns None when there is nothing worth showing - a quiet window draws no card, the
    same rule the per-message tab follows.
    """
    messages = [m for m in messages if (m.get("text") or "").strip()]
    if len(messages) < MIN_MESSAGES:
        return None

    scored = []
    for m in messages:
        for raw in _candidates(m):
            if not raw or sensitive.is_sensitive(raw) or _is_private(raw):
                continue                  # no credentials, no numbers, no private life
            if _is_intent(raw):
                continue                  # the Intent Log already has this
            c = _tidy(raw)
            # A shared cost is substantive even when it reads as dangling. summarizer's
            # _is_dangling treats a trailing "each" as a point stopping mid-phrase, which
            # is right for prose and wrong here: "18000 split three ways so 6000 each",
            # "6000 for rent plus 800 for electricity each" and "yes 6800 each" were ALL
            # rejected, and that is the most useful sentence shape in a bill-splitting
            # app. The score threshold and the stands-alone test still apply.
            if not c:
                continue
            # The strongest conversational signal only, not the sum - stacking them let a
            # chatty line with a day name and a vague plan outscore a real deadline.
            room_signal = max((w for w, rx in _ROOM_SIGNALS if rx.search(c)), default=0)
            is_money = bool(_MONEY_POINT.search(c))
            # A line that fired a conversational signal IS substantive, by definition, and
            # so is a shared cost. summarizer._is_substantive wants a bare function word
            # or a digit, which ordinary speech often has neither of once contracted:
            # "That's mine, i'll look today" - a person taking a bug - has function words
            # only inside "that's" and "i'll", and was rejected. _is_dangling separately
            # rejects every "X each" line, which is the shape of every split.
            # The exemption bypasses the function-word and dangling tests, NOT the length
            # one. Without this line "Friday", "Wifi", "Now wifi", "Around 3:30PM" and
            # "$556 sent" all became points, because a weekday, a clock time or a bill
            # word is enough to fire a signal on its own.
            if len(c.split()) < S.MINIMUM_POINT_WORDS:
                continue
            if not (S._is_substantive(c) or is_money or room_signal):
                continue
            score = S.importance(c)
            if is_money:
                score += _MONEY_BONUS      # what it cost is the premise of the split
            score = max(score, room_signal)
            if score < POINT_THRESHOLD or not _stands_alone(c):
                continue
            scored.append({"sender": m.get("sender"), "ts": m.get("ts"),
                           "text": c, "score": score, "kind": _kind_of(c)})

    # Strongest first, so when a near-duplicate is dropped the better-scoring copy of it
    # is the one that survives.
    scored.sort(key=lambda s: (-s["score"], s["ts"] or ""))
    kept, seen = [], []
    for s in scored:
        words = _content(s["text"])
        if _is_near_duplicate(words, seen):
            continue
        seen.append(words)
        kept.append(s)
        if len(kept) >= max(MIN_POINTS, min(limit, MAX_POINTS)):
            break
    if len(kept) < MIN_POINTS:
        return None

    kept.sort(key=lambda s: s["ts"] or "")
    stamps = [s["ts"] for s in messages if s.get("ts")]
    grouped = sections(kept)
    return RoomSummary(room=room, points=kept, messages=len(messages),
                       window_from=min(stamps) if stamps else None,
                       window_to=max(stamps) if stamps else None,
                       shape=shape(kept), cards=grouped)


# Two points belong to the same thread when they talk about the same things. Tuned on
# the realistic rooms: at 0.15 the dinner split and the build thread merged, at 0.30 the
# dinner split broke into three one-point cards.
_SAME_THREAD = 0.22

# Words that are in every conversation and name nothing, so they must never become a
# thread's headline. "Too" and "I'll Plumber" were the first version's output.
_TITLE_STOP = frozenset("""
also then just only still even really quite very much more most some sure yeah okay
said says tell told think thought know knew want wants need needs going gonna
today tomorrow yesterday tonight morning evening night week month year time
thing things stuff guys folks everyone anyone someone nothing anything something
actually honestly basically literally obviously maybe perhaps probably
""".split())

# A thread of one point is not a thread. Those are collected into a final catch-all card
# rather than shown as a row of single-bullet cards.
_MIN_THREAD = 2


def _link(a: dict, b: dict) -> float:
    wa, wb = _content(a["text"]), _content(b["text"])
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def threads(points: Sequence[dict]) -> List[List[dict]]:
    """Split the window's points into the threads being discussed.

    A window summary of a real room is several conversations at once - a build to test, a
    dinner split, a demo date moving - and one flat chronological list of all of them is
    what made the first version read badly: it changed subject almost every line. Each
    thread becomes its own card instead.

    Single-link clustering on shared content words, which is enough here because the
    points are already the few things worth remembering rather than raw chat, and because
    threads in a chat room genuinely do use different vocabulary - rent and plumber, cab
    and repayment. Chronology is preserved WITHIN a thread, which is what matters: "send
    me 500" then "actually send 300" has to stay in that order.
    """
    groups: List[List[dict]] = []
    for p in points:
        # AVERAGE linkage, not single. Single-link chains: the dinner split joined the
        # build thread because one point shared "600" with it and that point shared a
        # word with the next, so three unrelated threads merged into one card. A point
        # has to look like at least half of a thread to belong to it.
        joined = [g for g in groups
                  if sum(1 for q in g if _link(p, q) >= _SAME_THREAD) * 2 >= len(g)]
        if not joined:
            groups.append([p])
            continue
        # And when a point bridges two threads, it joins the one it matches best rather
        # than welding them together.
        best = max(joined, key=lambda g: sum(_link(p, q) for q in g) / len(g))
        best.append(p)
    for g in groups:
        g.sort(key=lambda p: p["ts"] or "")
    # Biggest thread first, then earliest - a card with five things in it matters more
    # than one with two, and a stable order beats whatever the clustering happened to do.
    groups.sort(key=lambda g: (-len(g), g[0]["ts"] or ""))
    real = [g for g in groups if len(g) >= _MIN_THREAD]
    loose = [p for g in groups if len(g) < _MIN_THREAD for p in g]
    if loose:
        loose.sort(key=lambda p: p["ts"] or "")
        real.append(loose)
    return real


def thread_title(points: Sequence[dict]) -> Optional[str]:
    """A headline for ONE thread.

    Titling a whole mixed window was hopeless - summarizer.title_for gave "Key Points" 91
    times in 255 summaries, because a window has no single subject to read off. A thread
    does, which is the point of splitting them: this is the same job as titling one
    message, which the existing title_for already does.
    """
    if not points:
        return None
    text = " . ".join(p["text"] for p in points)
    title = S.title_for(text)
    if title in ("Key Points", "Nothing to report"):
        # Name it after what the thread is mostly about: the content words it shares.
        # Name it after the words the thread keeps coming back to. Counted rather than
        # intersected, because an intersection over five points is usually empty, and
        # restricted to real words - the first version produced "Too" and "I'll Plumber".
        counts: dict = {}
        for p in points:
            for w in _content(p["text"]):
                if len(w) >= 4 and w not in _TITLE_STOP and not w.endswith("'ll"):
                    counts[w] = counts.get(w, 0) + 1
        best = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
        pick = [w for w, n in best if n >= 2][:2] or [w for w, _ in best[:2]]
        return " ".join(w.capitalize() for w in pick) if pick else None
    return title


def sections(points):
    """Group the window's points into what a recap is read for.

    Topic clustering was tried first and abandoned: word overlap on 3-10 word points
    merged the dinner split into the build thread through a shared "600", average linkage
    did not fix it, and naming each cluster produced "Too", "I'll Plumber" and "Saturday
    Kitchen". Sections need no clustering and no generated headline, and they are what the
    guidance on summary products says people actually want.

    Order is deliberate: decisions, then what somebody has to do, then what is still open,
    then the rest. Chronology is kept within a section, because "send me 500" then
    "actually send 300" has to stay in that order.
    """
    order = ["decisions", "action items", "open questions", "information"]
    buckets = {name: [] for name in order}
    for p in points:
        buckets[_section_of(p["text"])].append(p)
    out = []
    for name in order:
        rows = sorted(buckets[name], key=lambda p: p["ts"] or "")
        if rows:
            out.append({"heading": name, "points": rows})
    return out


def window(messages: Sequence[dict], since: Optional[str] = None,
           hours: Optional[float] = None) -> List[dict]:
    """The slice of a room to summarise.

    Two shapes, because the apps can supply different things. `since` is an ISO timestamp -
    "what was said after I last looked", which needs the client to know its own read
    position. `hours` is a lookback and needs nothing from the client. With neither, the
    whole list is the window, which is what a caller that has already sliced it wants.
    """
    msgs = sorted((m for m in messages if (m.get("text") or "").strip()),
                  key=lambda m: m.get("ts") or "")
    if since:
        msgs = [m for m in msgs if (m.get("ts") or "") > since]
    elif hours:
        latest = max((m.get("ts") or "" for m in msgs), default="")
        if latest:
            try:
                end = datetime.fromisoformat(latest.replace("Z", "+00:00"))
            except ValueError:
                return msgs
            cut = (end - timedelta(hours=hours)).isoformat()
            msgs = [m for m in msgs if (m.get("ts") or "") >= cut]
    return msgs
