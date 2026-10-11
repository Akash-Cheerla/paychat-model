"""One long chat message -> a topic and a few bullet points, on the server.

A port of Brahma's `MessageSummarizer.swift` (the EYES iOS app), so that iOS and Android
show the SAME card. Today iOS summarises on-device with Apple's Foundation Models and
Android has nothing; the cost of that is not only inconsistency but that every quality fix
needs an App Store release, and Android never gets one at all.

Ported from the rule-based half of that file, not the model half, for a reason worth
recording. The bad card Akash flagged on 2026-10-07 -

    IOS Issue
      - IOS
      - Android
      - Ride Request
      - Reply
      - IOS Issue

- cannot be produced by these rules. `points()` drops any sentence shorter than three
words, so a one-word bullet has no way in; those came from the on-device model ignoring its
own instruction ("never a single word or a bare list item"). Rules remove that failure mode
instead of patching it.

Two deliberate differences from the Swift, both fixes to bugs that card exposed:

  1. A point identical to the title is dropped. `polished()` in the Swift de-duplicates
     points against each other but never against the title, which is why "IOS Issue"
     appears as both the heading and a bullet.
  2. A message whose points are ALL fragments produces no card at all. The Swift decides
     what to summarise on length alone (140 characters, two sentences) and never asks
     whether there is anything worth remembering. A card of topic labels is worse than no
     card: the tab exists so someone can scroll back and find the thing they were told to
     do.

One known difference in capability: the Swift falls back to `NLTagger` to decide whether a
clause opens with a verb. There is no equivalent here, so only the verb list is consulted.
That makes `split_instructions` slightly more conservative - a run of instructions opening
with a verb outside the list stays one bullet rather than becoming several.
"""
import re
from typing import List, Optional
from urllib.parse import urlparse

# A message shorter than this says what it says - a summary of "see you at 5" is longer
# than the message.
MINIMUM_LENGTH = 140

# A point shorter than this is a list item, not a thing to remember.
MINIMUM_POINT_WORDS = 3

# How many bullets a card has is NOT a design choice - it is however many important things
# the message contains (Akash, 2026-10-08: "why are we sticking to fixed number of points,
# it purely depends on the messages"). A message with two things to say gets two bullets
# and one with nine gets nine. MAX_POINTS is a safety bound for a pathological message,
# not a target; MIN_POINTS keeps a card from being a single line when the message is a
# report made entirely of neutral observation.
MAX_POINTS = 14
MIN_POINTS = 3
# At or below this many candidate points, the message is treated as all content.
SHORT_MESSAGE = 5

# No point below this scores at all, so a card made only of 1s is still just talk.
CARD_THRESHOLD = 2

# A message earns a card when it has TWO points that score, or one that scores strongly.
# One signal on its own is usually an accident of vocabulary: a coffee machine is
# "broken", a bagel means "we need to talk", a flight is "moved to the 21st". Real
# messages carry several. Requiring two lifted the share of pure-chatter messages
# correctly dropped from 71% to 90% at no cost to recall, measured over 90 generated
# messages (2026-10-08). Requiring three reached 100% but suppressed a genuine two-line
# bug report, which is the wrong trade.
#
# Lowered from 5 to 4 after a live rehearsal (2026-10-11): a regression report listing
# three open bugs, and an iOS release note, each had ONE point scoring 4 and the rest
# scoring zero - so "two positives, or one >= 5" rejected both, while a message of three
# weak 1s passed. One strong signal is enough. The chatter that used to slip through
# topped out at 3 ("the coffee machine is broken", "we need to talk"), so 4 still
# excludes it.
CARD_STRONG = 4
# About two lines of a card - the design lets a point wrap once.
MAX_POINT_LENGTH = 120

_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)

# Words the model and the sender often case wrongly - "IOS", "Api".
_PROPER_CASE = {
    "ios": "iOS", "iphone": "iPhone", "ipad": "iPad", "macos": "macOS",
    "api": "API", "url": "URL", "ui": "UI", "ux": "UX", "otp": "OTP", "upi": "UPI",
}

_PUNCT = " \t\n\r.,;:!?\"'()[]{}"


def link_label(url: str) -> str:
    """"figma.com link" - a full URL wraps over several lines of a card and says nothing a
    reader can use there; the message itself is a tap away."""
    try:
        host = (urlparse(url if "://" in url else "https://" + url).hostname or "").lower()
    except ValueError:
        return "a link"
    if not host:
        return "a link"
    if host.startswith("www."):
        host = host[4:]
    return f"{host} link"


def replacing_links(text: str, label=link_label) -> str:
    return _URL.sub(lambda m: label(m.group(0)), text or "")


def fixing_case(text: str) -> str:
    out = []
    for word in (text or "").split(" "):
        core = word.strip(_PUNCT)
        fixed = _PROPER_CASE.get(core.lower())
        out.append(word.replace(core, fixed, 1) if fixed and core and core != fixed else word)
    return " ".join(out)


def tidy(text: str) -> str:
    collapsed = " ".join((text or "").split()).strip(".;:,")
    return collapsed[:1].upper() + collapsed[1:] if collapsed else ""


def _is_fragment(point: str) -> bool:
    return len(point.split()) < MINIMUM_POINT_WORDS


# The joints of a sentence. A run of topic nouns has none of these; almost any real
# statement has at least one. Used only to decide whether a card is worth showing.
_FUNCTION_WORDS = frozenset("""
a an the this that these those it its we you they them our your their my his her
in on at to for of with from by as into onto over under after before once when while
until during about across through between among per via
is are was were be been being has have had do does did
will would shall should can could may might must need needs
and or but not no nor if then than so because
all any both each every few many more most other others same some such only
remains remain stays stay keeps keep shows show showing creating causing
""".split())


# Characters that separate names in an address line, plus WhatsApp's invisible mention
# brackets. Written as a strip-set rather than a pattern: the regex this replaced had
# nested quantifiers and hung the test run outright.
_MENTION_CHARS = " @/,&+-~⁨⁩"


def _is_only_mentions(point: str) -> bool:
    """"@Yash Sharma / @rupeshsinghrajput91" - who, but not what (dogfood 2026-10-07)."""
    if "@" not in point:
        return False
    rest = "".join(" " if c in _MENTION_CHARS else c for c in point)
    # Whatever is left once the handles themselves are taken out.
    words = [w for w in rest.split() if w and not w.isdigit()]
    return len(words) <= 3 and all(w.replace(".", "").isalnum() for w in words)


def _is_dangling(point: str) -> bool:
    """Does this bullet stop mid-phrase? "Ask for a" and "bro i'm broke till" are halves."""
    words = [w.strip(_PUNCT).lower() for w in point.split()]
    words = [w for w in words if w]
    if not words:
        return True
    # Ends on a word that must be followed by something.
    if words[-1] in _FUNCTION_WORDS - {"it", "them", "us", "me", "you", "this", "that"}:
        return True
    # An odd number of double quotes means one was left open by a split.
    return _quote_count(point) % 2 == 1


def _is_substantive(point: str) -> bool:
    words = [w.strip(_PUNCT).lower() for w in point.split()]
    words = [w for w in words if w]
    if len(words) < MINIMUM_POINT_WORDS:
        return False
    if _is_dangling(point):
        return False
    # Names and handles with no verb and no object are not a point.
    if _is_only_mentions(point):
        return False
    # A bullet that is nothing but a quotation is an EXAMPLE, not an instruction. The
    # message complaining about the test logs quotes '"Can you book cab for me"' as the
    # thing people keep sending, and it arrived on the card reading like a thing to do
    # (dogfood 2026-10-07).
    stripped = point.strip()
    if len(stripped) > 1 and stripped[0] in _QUOTES and stripped[-1] in _QUOTES:
        return False
    if any(w in _FUNCTION_WORDS for w in words):
        return True
    if words[0] in _INSTRUCTION_VERBS:
        return True
    return any(c.isdigit() for c in point)


def merging_fragments(points: List[str]) -> List[str]:
    """"IOS" / "Android" / "Ride Request" as separate bullets -> one bullet, where the
    first of them stood."""
    fragments = [p for p in points if _is_fragment(p)]
    if len(fragments) < 2:
        return points
    result, placed = [], False
    for point in points:
        if not _is_fragment(point):
            result.append(point)
        elif not placed:
            result.append(", ".join(fragments))
            placed = True
    return result


class SummaryDraft:
    __slots__ = ("title", "points")

    def __init__(self, title: str, points: List[str]):
        self.title = title
        self.points = points

    def polished(self) -> "SummaryDraft":
        """Links shortened, casing fixed, duplicates dropped, fragments gathered into one,
        and - unlike the Swift - any point that merely repeats the title removed."""
        title = fixing_case(tidy(self.title))
        seen = set()
        cleaned = []
        for p in self.points:
            p = fixing_case(tidy(replacing_links(p)))
            if not p or p.lower() in seen:
                continue
            # "IOS Issue" as the heading AND the last bullet (reported 2026-10-07).
            if p.lower() == title.lower():
                continue
            seen.add(p.lower())
            cleaned.append(p)
        return SummaryDraft(title, merging_fragments(cleaned))

    def has_substance(self) -> bool:
        """Is there anything here worth scrolling back for?

        A card of topic labels - "iOS, Android, Ride Request, Reply" - tells the reader
        nothing they could act on, and showing it costs them the scroll. No card is better.

        Counting words is not enough: "iOS Android Ride Request Reply" is five of them and
        still says nothing. What separates a statement from a run of nouns is that a
        statement has joints - "White space remains ON all modules", "End call WHEN top
        banner disappears", "Build 52 assigned IN live". So a point counts when it has a
        function word, opens with an instruction verb ("Verify Ride Intents" has no joints
        and is still a thing to do), or carries a number.
        """
        return any(_is_substantive(p) for p in self.points)

    def as_dict(self) -> dict:
        return {"title": self.title, "points": list(self.points)}


# ---------------------------------------------------------------- sentences

# A line break ends a thought even without a full stop ("Hi Olivia," on its own line).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# "1. ", "*2.", "3) " - a list marker, not the end of a sentence. Splitting on its full
# stop produced the bullet "What actually helps: *1" (dogfood 2026-10-07). Turned into a
# line break instead, which is already a boundary. The trailing space is required, so
# "3.2" and "$1.50" are untouched.
#
# The prefix group is NOT optional. Matching after any whitespace ate sentence-final
# numbers instead: "We shipped Build 52. Please test" came back as "We shipped Build",
# losing the 52, and "The total came to 47." lost the amount. A list marker only ever
# follows a line start, a colon or a bullet.
_LIST_PREFIX = re.compile(r"(^|\n|:|[*•])\s*\d{1,2}\s*[.)]\s*(?=[A-Za-z])")


def sentences(text: str) -> List[str]:
    text = _LIST_PREFIX.sub(r"\1\n", text or "")
    out = []
    for paragraph in (text or "").splitlines():
        for piece in _SENTENCE_END.split(paragraph):
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out


# ---------------------------------------------------------------- points

# Openers that carry nothing to remember.
_PLEASANTRIES = (
    "hi", "hello", "hey", "dear", "thanks", "thank you", "best", "regards",
    "cheers", "looking forward", "hope", "good luck", "see you", "welcome",
    "congratulations", "if you have any questions",
    "feel free to reach", "have a great", "take care",
)
# "let me know" was here and dropped the whole sentence. That is right for "let me know
# if you have any questions" and wrong for "Let me know whether the banner behaviour
# matches what we agreed" - a real request for confirmation, thrown away (2026-10-08).
# Scoring already docks it a point, so a sentence that carries nothing else still falls
# out; one that carries something survives on its merits.

# Lead-ins stripped so the point starts with what to do. Applied until none is left, so
# "Please try to arrive..." comes out as "Arrive...".
_LEAD_INS = (
    "before you begin, please ", "before you begin, ", "just a reminder that ",
    "just a reminder about ", "just a reminder: ", "a quick reminder that ",
    "a reminder about ", "reminder: ", "please note that ", "please ", "kindly ",
    "don't forget to ", "do not forget to ", "make sure to ", "make sure you ",
    "remember to ", "try to ",
)

_REWRITES = [
    # "If you have any UX case studies, feel free to include them as well."
    #   -> "include UX case studies (if available)"
    (re.compile(r"^if you have (?:any )?(.+?), (?:please |feel free to |you can |you may )?"
                r"(\w+) (?:them|it|those)(?: along| as well| too)?$", re.IGNORECASE),
     r"\2 \1 (if available)"),
    # "If you haven't completed the forms yet, please submit them by Friday."
    #   -> "submit the forms by Friday"
    (re.compile(r"^if you haven'?t \w+ (.+?)(?: yet)?, (?:please )?(\w+) (?:them|it) (.+)$",
                re.IGNORECASE),
     r"\2 \1 \3"),
    # "Your orientation is scheduled for Monday..." -> "orientation on Monday..."
    (re.compile(r"^your (\w+) is (?:scheduled|booked|set) for (.+)$", re.IGNORECASE),
     r"\1 on \2"),
]

# Where a point stops - what follows is the reason, not the thing to do.
_REASON_MARKERS = (", so ", " so that ", " so ", " because ", " in order to ")

# The verbs chat instructions open with.
_INSTRUCTION_VERBS = {
    "add", "arrive", "ask", "attach", "attend", "book", "bring", "buy", "call",
    "cancel", "carry", "check", "collect", "come", "complete", "confirm", "contact",
    "create", "download", "drop", "email", "fill", "finish", "follow", "forward",
    "get", "give", "go", "include", "install", "join", "keep", "leave", "let",
    "log", "make", "meet", "message", "order", "pack", "pay", "pick", "prepare",
    "print", "provide", "read", "register", "renew", "reply", "reserve", "return",
    "review", "schedule", "send", "share", "sign", "start", "submit", "take",
    "text", "transfer", "update", "upload", "use", "visit", "wear", "write",
    # This team's vocabulary. Without them "Verify Ride Intents" and "Try other cases"
    # read as noun phrases, and a run of them never splits into separate bullets.
    "verify", "test", "try", "ensure", "fix", "deploy", "merge", "revert",
    "reproduce", "restart", "rerun", "end", "raise", "assign", "retest",
}


def _starts_with_verb(clause: str) -> bool:
    first = clause.split(" ", 1)[0].lower().strip(_PUNCT) if clause.split() else ""
    return first in _INSTRUCTION_VERBS


def split_instructions(sentence: str) -> List[str]:
    """"upload your portfolio, complete the questionnaire, and join Slack" -> three points
    - but only when every piece starts with a verb. "Submit your Figma file, prototype
    link, and PDF" stays whole: the pieces after the first are the objects of one
    instruction, not new ones.
    """
    if _quote_count(sentence) >= 2:
        return [sentence]           # see _split_long: quoted examples stay whole
    flat = sentence.replace(", and ", ", ").replace(", then ", ", ")
    pieces = [p.strip() for p in _split_outside_quotes(flat)]
    pieces = [p for p in pieces if p]
    if len(pieces) > 1 and all(_starts_with_verb(p) for p in pieces):
        return pieces
    return [sentence]


def _rewritten(sentence: str) -> str:
    for pattern, template in _REWRITES:
        if pattern.search(sentence):
            return pattern.sub(template, sentence)
    return sentence


_QUOTES = "\"'“”‘’"


def _quote_count(text: str) -> int:
    return sum(text.count(q) for q in "\"“”")


def _split_outside_quotes(body: str) -> List[str]:
    """Split on ", " but never on a comma inside quotation marks."""
    out, current, depth = [], "", 0
    i = 0
    while i < len(body):
        ch = body[i]
        if ch in "\"“":
            depth += 1
        elif ch in "”":
            depth = max(0, depth - 1)
        if depth == 0 and body.startswith(", ", i):
            out.append(current)
            current = ""
            i += 2
            continue
        current += ch
        i += 1
    if current:
        out.append(current)
    return out or [body]


def _split_long(body: str) -> List[str]:
    """An over-long sentence broken at its commas into chunks that each fit on the card.

    Truncating instead lost the end of the sentence, and the end of a sentence is where
    people put the thing that matters: "...Jun, that's yours and it has to be in" lost
    "before Friday at noon", "...stack trace is in" lost the ticket number, "@Kai do you
    have" lost the ask (generated set, 2026-10-08). Each chunk is then scored on its own,
    so the half with the deadline competes for a slot instead of riding on the half
    without it.
    """
    if len(body) <= MAX_POINT_LENGTH:
        return [body]
    # A message that QUOTES examples is not a list of clauses. Splitting on every comma
    # cut straight through them on real traffic: a message listing phrasings to try came
    # back as '"can u send me 500", "bro i\'m broke till' / 'Friday", "spot me for the
    # tickets"' - two bullets of rubble out of one sentence (dogfood 2026-10-07). Leave a
    # quoting sentence whole and let the length cap trim it.
    if _quote_count(body) >= 2:
        return [body]
    chunks, current = [], ""
    for piece in _split_outside_quotes(body):
        candidate = f"{current}, {piece}" if current else piece
        if current and len(candidate) > MAX_POINT_LENGTH:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    # A chunk too short to stand alone is glued back onto the one before it rather than
    # left as a bare "Jun".
    merged: List[str] = []
    for chunk in chunks:
        if merged and _is_fragment(chunk):
            merged[-1] = f"{merged[-1]}, {chunk}"
        else:
            merged.append(chunk)
    return merged or [body]


# "1.", "2)", "*", "-", "•" at the start of a point, and WhatsApp's bold asterisks. Real
# messages are written as numbered lists and the markers came through onto the card:
# "*2.Say it differently every time" (dogfood 2026-10-07). The same message posted in two
# rooms also produced different cards - 6 bullets in one, 11 in the other - purely because
# one copy carried asterisks.
_LIST_MARKER = re.compile(r"^\s*(?:[*•\-–]+\s*)?(?:\d{1,2}\s*[.)]\s*)?\s*")


def _tidy_point(clause: str) -> str:
    point = _LIST_MARKER.sub("", clause or "").strip().strip("*").strip()
    point = point.strip(" \t\n\r.!;:")
    if len(point) > MAX_POINT_LENGTH:
        cut = point[:MAX_POINT_LENGTH]
        space = cut.rfind(" ")
        point = (cut[:space] if space > 0 else cut) + "…"
    return point[:1].upper() + point[1:] if point else ""


# What makes a sentence worth keeping. "Summary is to summarize important parts of
# messages, not everything" - Akash, 2026-10-08. Taking the first five sentences in order
# kept "Overall it feels a lot smoother than 51" and dropped both the security deadline and
# the crash someone had found, which is the opposite of the tab's purpose.
_IMPORTANT = [
    # A time something is due. The single most useful thing to scroll back for.
    (4, r"\b(?:by|before|due|until|no later than)\s+"
        r"(?:\d|mon|tue|wed|thu|fri|sat|sun|today|tomorrow|tonight|eod|noon|midnight|next\s+\w+)"),
    (3, r"\b(?:eod|asap|urgent(?:ly)?|deadline|immediately)\b"),
    # Someone being ASKED to do something. The commonest important thing in a group chat
    # and the biggest hole in the first version of this list: "Priya, can you update the
    # spec doc", "so can Marco take it", "@Kai do you have those handy" all scored zero
    # and were dropped (generated set, 2026-10-08).
    (3, r"\b(?:can|could|would|will)\s+(?:you|someone|anyone|we|[A-Z]\w+)\b"
        r"|\bare\s+you\s+(?:able|ok)\b|\bwho\s+(?:can|is|wants)\b"),
    # An @mention is an owner being named.
    (3, r"@\w+"),
    # A change that was decided, phrased as a change rather than as a decision.
    (2, r"\b(?:cut\s+from|cutting|dropping|dropped|moving\s+to|moved\s+to|switching\s+to|"
        r"renaming|replacing|adding|removing|aiming\s+to|going\s+to\s+(?:ship|cut|move))\b"),
    # An obligation, not an observation. "do not" and "don't" used to be here and were the
    # second-commonest false signal in chatter - "I don't know why they'd do that" bought
    # a card for a message about a sandwich shop (2026-10-08).
    # "needs an owner" and "needs a pass" are obligations too; only "needs to" was matched.
    (3, r"\b(?:must|mandatory|required|has\s+to|have\s+to|blocking|blocker|"
        r"needs?\s+(?:to|a|an|another|someone|sign)|still\s+needs?)\b"),
    # "should" was tried here and taken out again. It reads as an obligation ("the spacing
    # should revert once the banner disappears") but also as casual advice ("you should try
    # the ramen place"), and scoring it cost 14 points of chatter detection for half a
    # point of recall. The report sentence it was added for is kept by the short-message
    # rule in _select instead (2026-10-08).
    # Something is broken - the other reason anyone scrolls back. The second line is the
    # vocabulary of a report that never says "bug": the voice-note card in the app
    # ("saves in 15-20 seconds", "auto-disconnects after 2-3 attempts") scored almost
    # nothing without it.
    (3, r"\b(?:crash(?:es|ed|ing)?|bug|broken|fail(?:s|ed|ing)?|error|not\s+working|"
        r"regression|blocked|stuck|wrong|missing|issue)\b"),
    (3, r"\b(?:too\s+slow|slow(?:er|ly)?|delay(?:s|ed)?|lag(?:s|gy|ging)?|hangs?|"
        r"freez(?:e|es|ing)|disconnects?|auto-disconnects?|drops?\s+(?:the\s+)?call|"
        r"times?\s+out|timeout|overlap(?:s|ping)?|misaligned|unable\s+to|"
        r"does\s?n'?t\s+work|won'?t\s+(?:load|open|save|send))\b"),
    # Something still outstanding. A live rehearsal on 2026-10-11 dropped a regression
    # report whose central line was "Three things still open: the split amount shows
    # blank..." - a list of three bugs, scoring zero, because none of them used a word
    # from the pattern above.
    (3, r"\b(?:still\s+(?:open|failing|broken|outstanding|not\s+working)|"
        r"remains?\s+(?:open|broken)|outstanding|shows?\s+blank|comes?\s+back\s+empty)\b"),
    # A fix is as worth recording as the fault. The same rehearsal dropped "I've pushed
    # the fix for the summary card showing duplicate bullets" for the same reason.
    (2, r"\b(?:fix(?:ed|es|ing)?|resolved|patched|workaround|rolled\s+back|reverted)\b"),
    # A decision that was taken.
    (2, r"\b(?:assigned|approved|rejected|merged|released|deployed|signed\s+off|"
        r"marked|agreed|decided|confirmed|cancelled|postponed)\b"),
    # Weak on its own. "Meeting room 2" and "the third day in a row" are numbers in
    # chatter; a number only means something next to one of the signals above.
    (1, r"\d"),
]
# Throat-clearing, opinion and hedging. Present in nearly every long chat message and
# worth nothing to the person reading the card a day later.
_UNIMPORTANT = [
    (-4, r"^(?:i\s+(?:think|guess|feel|believe|spent|was|am|have\s+been|wanted)|overall|"
         r"anyway|personally|honestly|basically|just\s+wanted|fyi)\b"),
    # A message that says outright it has nothing in it. The scoring word is INSIDE the
    # negation - "Nothing important from me today, just procrastinating before I start on
    # the bug list" scored on "bug" and "today" and got a card (2026-10-08).
    (-3, r"\bnothing\s+(?:else\s+)?(?:important|urgent|major|pressing|critical|new|"
         r"to\s+report|blocking|much)\b"
         r"|\bno\s+(?:big\s+deal|blockers?|updates?|news|agenda)\b"
         r"|\bnot\s+a\s+blocker\b"),
    # Softer: "let me know which ones are blocking" is still a real ask, so this must not
    # cancel out the content around it.
    # -2, not -1: "let" is an instruction verb and scores +2 on its own, so at -1 a bare
    # "Let me know if you want to grab lunch later" still came out positive and helped buy
    # a card. At -2 it nets zero, while "Let me know whether the banner matches what we
    # agreed" keeps the points its content earns (2026-10-08).
    (-2, r"\b(?:let\s+me\s+know|up\s+to\s+you|no\s+rush|if\s+you\s+get\s+a\s+chance)\b"),
    # Hedging is weak evidence, not a veto. At -2 it was cancelling real content in the
    # same sentence: "probably a two-line fix in the rounding helper, so can Marco take
    # it" scored negative and lost the only ask in the message (2026-10-08).
    (-1, r"\b(?:probably|maybe|i\s+think|i\s+guess|seems?\s+(?:like|to)|feels?\s+like|"
         r"kind\s+of|sort\s+of|more\s+or\s+less)\b"),
    # "I just need to say this somewhere" is not an obligation, but it matches the
    # "needs? to" pattern above and bought a card for a message about a sandwich shop.
    # Cancels that score rather than merely reducing it.
    (-4, r"\b(?:need|needed|want|wanted|have|had)\s+to\s+"
         r"(?:say|vent|mention|add|share|tell|ask|rant|get\s+this\s+off)\b"),
]
_IMPORTANT = [(w, re.compile(p, re.IGNORECASE)) for w, p in _IMPORTANT]
_UNIMPORTANT = [(w, re.compile(p, re.IGNORECASE)) for w, p in _UNIMPORTANT]


# "today" is a deadline in "send it today" and small talk in "did you see the weather
# today?". On its own it was the single commonest reason a chatter message got a card -
# "Catch you all at standup tomorrow", "that standup ran long today", "No agenda today,
# just vibes" (2026-10-08). It only counts next to something being asked for. A real
# deadline phrased "by Friday" is already caught by the first pattern above.
_DAY_WORD = re.compile(r"\b(?:today|tomorrow|tonight|this\s+(?:morning|afternoon|evening)|"
                       # A weekday or a month is the same kind of word and makes the same
                       # mistake unaided: "I went hiking on Saturday" is not a deadline.
                       r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                       r"january|february|march|april|june|july|august|september|"
                       r"october|november|december)\b", re.IGNORECASE)
_OBLIGATION = re.compile(r"\b(?:must|mandatory|required|has\s+to|have\s+to|needs?\s+to|"
                         r"blocking|blocker|please|make\s+sure|send|sending|ship|shipping|"
                         r"deliver|finish|complete|submit|upload|review|merge|deploy|"
                         r"cut|release|aiming|get\s+me|due|deadline|by)\b", re.IGNORECASE)

# "can you update the spec doc" is a request; "can you believe it" is not.
_RHETORICAL = re.compile(r"\b(?:can|could)\s+you\s+(?:believe|imagine|even|tell)\b",
                         re.IGNORECASE)


def importance(point: str) -> int:
    """How much this point earns its place on the card. Can go negative."""
    score = sum(w for w, p in _IMPORTANT if p.search(point))
    score += sum(w for w, p in _UNIMPORTANT if p.search(point))
    if _starts_with_verb(point):
        score += 2          # an instruction aimed at the reader
    if _DAY_WORD.search(point) and (_OBLIGATION.search(point) or _starts_with_verb(point)):
        score += 3
    if _RHETORICAL.search(point):
        score -= 3          # cancels the request score above
    return score


def _select(points: List[str]) -> List[str]:
    """The points worth keeping, in the order they were written.

    Chatter is dropped outright; what survives competes for the remaining slots. Order is
    restored afterwards because a card is read top to bottom, not best-first.
    """
    scored = [(i, p, importance(p)) for i, p in enumerate(points)]
    # When a message contains chatter, the neutral middle is usually chatter too and only
    # the points that score belong on the card. When it contains NONE, the message is all
    # content and a neutral sentence is still something the sender wanted read: the four
    # lines of the Overlap Issues report are a deadline-free bug report, and keeping three
    # of them dropped whichever one happened to score lowest (2026-10-08).
    # A SHORT message is all content: the four lines of the Overlap Issues report are a
    # bug report with no filler, and keeping only the three that scored dropped whichever
    # one happened to score lowest. A long one has filler in the middle, so there only the
    # points that score belong (2026-10-08). Gating on "no negatives" instead was tried and
    # took leakage from 17% to 33%.
    # A bullet that stops mid-phrase, or is only names, is not a point whatever it scores.
    scored = [s for s in scored if _is_substantive(s[1])] or scored
    keep = [s for s in scored if (s[2] >= 0 if len(scored) <= SHORT_MESSAGE else s[2] > 0)]
    if len(keep) < MIN_POINTS:
        # Nothing scored, or almost nothing. A message can be a real report made entirely
        # of neutral observation - "White space remains on all modules" carries no deadline
        # and no instruction and is still what the sender wanted read - so the best of the
        # rest fills in. has_substance() and CARD_THRESHOLD decide whether the card is
        # shown at all.
        rest = sorted((s for s in scored if s[2] <= 0), key=lambda s: (-s[2], s[0]))
        keep += rest[:MIN_POINTS - len(keep)]
    keep.sort(key=lambda s: (-s[2], s[0]))
    return [p for i, p, _ in sorted(keep[:MAX_POINTS], key=lambda s: s[0])]


def points_for(text: str) -> List[str]:
    out: List[str] = []
    for sentence in sentences(text):
        lowered = sentence.lower()
        if any(lowered.startswith(p) for p in _PLEASANTRIES):
            # Only the greeting goes, not the sentence. "Hi all, just finished regression
            # testing on build 456" was dropped whole, and the build number with it
            # (generated set, 2026-10-08).
            head, comma, rest = sentence.partition(",")
            rest = rest.strip()
            # The remainder has to be checked too: "Hi team, hope you are all doing well"
            # strips to "hope you are all doing well", which is itself a pleasantry.
            if (comma and len(head.split()) <= 4 and len(rest.split()) >= 3
                    and not any(rest.lower().startswith(p) for p in _PLEASANTRIES)):
                sentence = rest
                lowered = sentence.lower()
            else:
                continue
        if len(sentence.split()) < 3:
            continue

        body = sentence.strip(".!")
        changed = True
        while changed:
            changed = False
            for lead in _LEAD_INS:
                if body.lower().startswith(lead):
                    body = body[len(lead):]
                    changed = True
                    break
        body = _rewritten(body)
        # "Your appointment with..." -> "Appointment with...". After the rewrites, one of
        # which reads the "your".
        if body.lower().startswith("your "):
            body = body[5:]
        # What follows "so" is usually the reason and worth cutting - but not always. "I
        # don't want to be the one guessing, SO CAN MARCO TAKE IT" and "tell me today SO I
        # CAN CUT SCOPE" both had the real content after the marker, and cutting threw it
        # away (generated set, 2026-10-08). Keep the tail as its own candidate when it
        # carries something; let it compete, and drop it when it does not.
        tails = []
        for marker in _REASON_MARKERS:
            idx = body.lower().find(marker)
            if idx > 0 and len(body[:idx].split()) >= 3:
                tail = body[idx + len(marker):].strip()
                if tail and importance(tail) >= 2:
                    tails.append(tail)
                body = body[:idx]
        for clause in list(split_instructions(body)) + tails:
            for chunk in _split_long(clause):
                point = _tidy_point(chunk)
                if point:
                    out.append(point)
            # Collect every candidate rather than stopping at MAX_POINTS: the five that
            # belong on the card are chosen by _select(), and stopping early would let
            # position decide instead. Bounded so a pathological message cannot run away.
            if len(out) >= 40:
                break
        if len(out) >= 40:
            break
    return _select(out)


# ---------------------------------------------------------------- title

_GENERIC_TOPIC = "Reminder"
# The generic list below was written for consumer messages (appointments, deliveries,
# travel). Run over this team's actual chat every card came out "Key Points", because
# nothing in it covers builds, releases, bugs or design. The good titles in the app today
# ("Build 52", "Overlap Issues") were WRITTEN by the on-device model, which rules cannot
# do - so the vocabulary has to carry them instead.
_WORK_TOPICS = [
    ("Build",        ("build ", "apk", "ipa", "testflight", "staging build")),
    ("Release",      ("release", "rollout", "shipped to", "go live", "live build")),
    ("Security",     ("security", "vulnerability", "cve", "pen test", "penetration")),
    ("Design",       ("figma", "design", "mockup", "prototype", "wireframe")),
    ("Testing",      ("qa", "test the", "regression", "verify", "reproduce")),
    ("Bug",          ("bug", "crash", "issue", "broken", "not working", "fails")),
    ("Call",         ("call drop", "auto-disconnect", "voice note", "voice call")),
    ("Layout",       ("overlap", "spacing", "white space", "banner", "alignment")),
    ("Login",        ("login", "sign in", "otp", "authentication")),
    ("Backup",       ("backup", "restore")),
]
_TOPICS = [
    ("Onboarding",  ("onboarding", "orientation", "welcome to", "first day")),
    ("Appointment", ("appointment", "clinic", "dr.", "doctor", "consultation")),
    ("Assignment",  ("assignment", "homework", "submission", "deadline")),
    ("Interview",   ("interview",)),
    ("Meeting",     ("meeting", "agenda", "zoom", "standup", "sync")),
    ("Payment",     ("payment", "invoice", "refund", "pay ", "$")),
    ("Travel",      ("flight", "hotel", "trip", "itinerary", "check-in")),
    ("Delivery",    ("delivery", "shipped", "package", "courier", "order")),
    ("Event",       ("event", "party", "wedding", "birthday", "celebration")),
    ("Reminder",    ("reminder", "don't forget", "remember to")),
]


# "Build 52", "build #52", "Build 52.1" - the number is what distinguishes one of these
# messages from the next, so it belongs in the title.
_NUMBERED = re.compile(r"\b(build|release|version|sprint|ticket|rc)\s*#?\s*(\d+(?:\.\d+)?)\b",
                       re.IGNORECASE)


def title_for(text: str) -> str:
    """First topic whose keywords appear, earliest mention winning - what a message opens
    with is usually what it is about. "Reminder" only as a fallback: "just a reminder about
    your appointment" is about the appointment.

    Work vocabulary is consulted first and keeps its number when it has one, so a room full
    of build chatter does not produce twenty cards all called "Key Points".
    """
    lowered = (text or "").lower()
    numbered = _NUMBERED.search(text or "")
    if numbered:
        return f"{numbered.group(1).capitalize()} {numbered.group(2)}"

    def best(table):
        hits = []
        for name, keywords in table:
            found = [lowered.find(k) for k in keywords if lowered.find(k) >= 0]
            if found:
                hits.append((name, min(found)))
        return hits

    work = best(_WORK_TOPICS)
    if work:
        return min(work, key=lambda h: h[1])[0]
    hits = best(_TOPICS)
    specific = [h for h in hits if h[0] != _GENERIC_TOPIC]
    pool = specific or hits
    if pool:
        return min(pool, key=lambda h: h[1])[0]
    return _topic_from_words(text) or "Key Points"


# Words that are common in any chat and so say nothing about what a message is ABOUT.
_TITLE_STOP = frozenset("""
think know want need make take come give look thing things stuff lot bit way ways
time times day days today tomorrow guys team everyone people person someone anyone
send sending sent get got going go went done doing have has had will would should
please thanks thank sure okay yeah yes actually really just also still even maybe
good great fine nice better best able let lets much many more most other same
one two three four five six seven eight nine ten first second third
which what when where why how who whom whose there here again back then than
app user users side end new old next last another thing done
""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def _topic_from_words(text: str) -> Optional[str]:
    """A title built from the words the message actually repeats.

    "Key Points" was a third of all titles on real traffic, because the topic lists cover
    builds and bugs and not ordinary discussion (dogfood 2026-10-07). A word the sender
    used three times is what the message is about, whatever our vocabulary thinks.
    """
    counts = {}
    for w in _WORD.findall(text or ""):
        low = w.lower()
        if low in _FUNCTION_WORDS or low in _TITLE_STOP or low in _PLEASANTRIES:
            continue
        counts[low] = counts.get(low, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], text.lower().find(kv[0])))
    # Four letters minimum and said at least twice. Shorter or rarer words produced
    # titles like "Again One" and "Model Two" - the shape of a title without the sense.
    top = [w for w, n in ranked if n >= 2 and len(w) >= 4][:2]
    if not top:
        return None
    return " ".join(w.capitalize() for w in top)


# ---------------------------------------------------------------- entry points

# "Restoring a chat backup after login: https://figma.com/..." - a label, then a link. A
# list of these is a reference index, which is exactly what the Design Reference card in
# the app is, and it is one of the cards Akash singled out as right (2026-10-07).
_LABELLED_LINK = re.compile(r"([^\n:]{3,80}):\s*(?:https?://|www\.)\S+", re.IGNORECASE)


def labelled_links(text: str) -> List[str]:
    out = []
    for m in _LABELLED_LINK.finditer(text or ""):
        label = tidy(m.group(1))
        url = _URL.search(m.group(0))
        if label and url:
            out.append(f"{label}: {url.group(0)}")
    return out


# Where a reference list points decides what to call it.
_REFERENCE_HOSTS = [
    ("Design Reference", ("figma.com", "sketch.com", "zeplin.io", "invisionapp.com")),
    ("Documents", ("docs.google.com", "notion.so", "confluence", "sharepoint.com")),
    ("Tickets", ("atlassian.net", "jira", "linear.app", "github.com", "gitlab.com")),
    ("Recordings", ("zoom.us", "loom.com", "drive.google.com")),
]


def _reference_title(text: str) -> str:
    hosts = " ".join(_URL.findall(text or "")).lower()
    for name, needles in _REFERENCE_HOSTS:
        if any(n in hosts for n in needles):
            return name
    return "Links"


def is_worth_summarizing(text: str) -> bool:
    """Long enough, and more than one sentence, measured WITHOUT its links.

    A message that is mostly URLs is long only on paper, and its summary would just be the
    message again - UNLESS the links are labelled, in which case the labels are the point.
    """
    if len(labelled_links(text)) >= 2:
        return True
    prose = replacing_links(text or "", lambda _: "").strip()
    if len(prose) < MINIMUM_LENGTH:
        return False
    return len(sentences(prose)) >= 2


def summarize(text: str) -> Optional[SummaryDraft]:
    """The card for this message, or None when there is nothing worth showing."""
    if not is_worth_summarizing(text):
        return None
    # A reference message IS its links: shortening them to "figma.com link" would leave a
    # card nobody can act on. Every other message gets the links shortened before anything
    # reads it, so a point is not cut off halfway through a URL and "zoom.us/..." does not
    # make it a Meeting.
    links = labelled_links(text)
    if len(links) >= 2:
        # Titled by where the links POINT, not by the first keyword in the labels. Two
        # Figma links labelled "Restoring a chat backup..." and "Deleting a location..."
        # came out as "Backup", because earliest-mention wins and "backup" is in the first
        # label - a message that is plainly a design reference.
        return SummaryDraft(_reference_title(text), links[:MAX_POINTS])

    shortened = replacing_links(text)
    draft = SummaryDraft(title_for(shortened), points_for(shortened)).polished()
    if not draft.points or not draft.has_substance():
        return None
    # Shaped like sentences is not the same as worth reading. 21 generated messages of
    # pure chatter - a sandwich shop changing its bread, a lost jacket, someone's cat -
    # every one produced a card, because every one is made of grammatical sentences that
    # simply say nothing (2026-10-08). A card needs at least one point that is actually
    # about something: a deadline, an instruction, a decision, something broken.
    scores = [importance(p) for p in draft.points]
    if max(scores, default=0) < CARD_THRESHOLD:
        return None
    if not (sum(1 for s in scores if s > 0) >= 2
            or max(scores) >= CARD_STRONG):
        return None
    return draft
