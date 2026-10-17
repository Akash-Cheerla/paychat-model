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

import summarizer as S

# A day of a working group repeats itself. The dogfood log is the extreme - "Can you book
# cab for me" fifteen times - but every real room does it.
NEAR_DUPLICATE = 0.75

# The same bar a per-message card has to clear, so the two features agree about what
# counts. Below it, a point is an observation rather than something to remember.
POINT_THRESHOLD = 2

# A window with nothing in it gets no card at all, the same way a short message does not.
MIN_POINTS = 2

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
    # {"requests": 4, "deadlines": 2} - the at-a-glance substitute for a topic.
    shape: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"room": self.room, "messages": self.messages,
                "window_from": self.window_from, "window_to": self.window_to,
                "shape": dict(self.shape),
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
_SHAPE = [
    ("deadlines", re.compile(r"\b(?:by|before|due|until|no later than)\s+"
                             r"(?:\d|mon|tue|wed|thu|fri|sat|sun|today|tomorrow|tonight|"
                             r"eod|noon|midnight|next\s+\w+)|\b(?:eod|asap|deadline)\b",
                             re.IGNORECASE)),
    ("requests", re.compile(r"\b(?:can|could|would|will)\s+(?:you|someone|anyone|we)\b|"
                            r"\bplease\b|@\w+|\bgimme\b|\bneed\s+(?:you|someone)\b",
                            re.IGNORECASE)),
    ("problems", re.compile(r"\b(?:crash(?:es|ed|ing)?|bugs?|broken|fail(?:s|ed|ing)?|"
                            r"errors?|not\s+working|stuck|wrong|missing|issues?|"
                            r"can(?:not|n't|'t))\b", re.IGNORECASE)),
    ("decisions", re.compile(r"\b(?:assigned|approved|rejected|merged|released|deployed|"
                             r"agreed|decided|confirmed|cancelled|postponed|fixed)\b",
                             re.IGNORECASE)),
]


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
        for c in _candidates(m):
            if not c or not S._is_substantive(c):
                continue
            score = S.importance(c)
            if score < POINT_THRESHOLD or not _stands_alone(c):
                continue
            scored.append({"sender": m.get("sender"), "ts": m.get("ts"),
                           "text": c, "score": score})

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
    return RoomSummary(room=room, points=kept, messages=len(messages),
                       window_from=min(stamps) if stamps else None,
                       window_to=max(stamps) if stamps else None,
                       shape=shape(kept))


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
