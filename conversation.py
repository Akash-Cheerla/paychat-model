"""
Conversation State Machine for PayChat intent detection.

Architecture:
  1. RoomStateStore — tracks pending requests per room (who asked what, with slots)
  2. ResponseClassifier — classifies how someone responded (ack, reject, question, etc.)
  3. DecisionEngine — pending request + response type → fire / remind / cancel / no-op
  4. ConversationStateMachine — orchestrates all three, called from full_pipeline()

Design:
  - Model runs standalone (no context prefix) and detects WHAT a message says.
  - State machine decides WHEN to fire based on conversation flow.
  - For money+ride: requests are stored as pending, intents fire on the RESPONSE.
  - Other intents pass through unchanged (future expansion adds rules per intent).
  - When sender is not provided, falls back to immediate firing (backward compat).
"""

import re
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("paychat.conversation")

# ── Constants ──

PENDING_TTL = 300          # 5 minutes — pending requests expire after this
PENDING_MSG_LIMIT = 10     # expire pending after this many intervening messages in the room
ARCHIVE_TTL = 172800       # 48 hours — archived (expired) requests kept for reply_to matching
MANAGED_INTENTS = {"money", "ride"}  # intents governed by state machine


# ── Response Types ──

class ResponseType:
    POSITIVE_ACK = "positive_ack"       # sure, bet, sending now, on it
    FUTURE_PROMISE = "future_promise"   # next week, later, friday
    REJECTION = "rejection"             # nah, no way, that was ur treat
    ALREADY_DONE = "already_done"       # done, already sent, eta 5 min
    QUESTION = "question"               # how much, where to, which one
    DEFERRAL = "deferral"               # lets see, maybe later, not sure yet
    SELF_INITIATED = "self_initiated"   # ill grab a cab, im sending now
    NEW_REQUEST = "new_request"         # a new request (not a response)
    NEUTRAL = "neutral"                 # unrelated message


# ── Actions ──

class Action:
    FIRE = "fire"               # fire the intent now
    REMINDER = "reminder"       # set a reminder, not immediate payment/ride
    CANCEL = "cancel"           # cancel the pending request
    NO_FIRE = "no_fire"         # don't fire (already done, deferred, etc.)
    KEEP_PENDING = "keep_pending"  # question asked, keep waiting for real response
    STORE_PENDING = "store_pending"  # new request, store and wait for response


# ── Pending Request ──

@dataclass
class PendingRequest:
    intent: str
    sender: str
    text: str
    slots: dict = field(default_factory=dict)
    money_info: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    message_id: Optional[str] = None
    responded_by: set = field(default_factory=set)  # senders who already responded
    cls_embedding: object = None  # cached CLS tensor for ML response classification

    messages_since: int = 0

    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > PENDING_TTL or self.messages_since >= PENDING_MSG_LIMIT

    def has_responded(self, sender: str) -> bool:
        return sender in self.responded_by

    def mark_responded(self, sender: str):
        self.responded_by.add(sender)


# ── Room State Store ──

def _is_dm(room_id: str) -> bool:
    """DM rooms use ambient matching; group rooms require reply_to."""
    return room_id.startswith("dm_") if room_id else False


class RoomStateStore:
    """Tracks pending requests per room. Thread-safe for single-process async."""

    def __init__(self):
        self.rooms: dict[str, list[PendingRequest]] = defaultdict(list)
        self.archived: dict[str, list[PendingRequest]] = defaultdict(list)

    def add_pending(self, room_id: str, req: PendingRequest):
        self._clean_expired(room_id)
        # Replace existing pending of same intent from same sender
        self.rooms[room_id] = [
            r for r in self.rooms[room_id]
            if not (r.intent == req.intent and r.sender == req.sender)
        ]
        self.rooms[room_id].append(req)
        logger.info(f"[{room_id}] Stored pending {req.intent} from {req.sender}: {req.text!r}")

    def get_pending(self, room_id: str, intent: str = None) -> list[PendingRequest]:
        self._clean_expired(room_id)
        reqs = self.rooms.get(room_id, [])
        if intent:
            return [r for r in reqs if r.intent == intent]
        return list(reqs)

    def get_pending_from_others(self, room_id: str, sender: str) -> list[PendingRequest]:
        """Get pending requests in this room from senders OTHER than the given sender,
        excluding requests this sender already responded to."""
        self._clean_expired(room_id)
        return [
            r for r in self.rooms.get(room_id, [])
            if r.sender != sender and not r.has_responded(sender)
        ]

    def remove_pending(self, room_id: str, intent: str, requester: str = None):
        if room_id in self.rooms:
            if requester:
                self.rooms[room_id] = [
                    r for r in self.rooms[room_id]
                    if not (r.intent == intent and r.sender == requester)
                ]
            else:
                self.rooms[room_id] = [
                    r for r in self.rooms[room_id] if r.intent != intent
                ]

    def find_by_message_id(self, room_id: str, message_id: str) -> Optional[PendingRequest]:
        """Find a pending or archived request by its original message_id."""
        for r in self.rooms.get(room_id, []):
            if r.message_id == message_id:
                return r
        for r in self.archived.get(room_id, []):
            if r.message_id == message_id:
                return r
        return None

    def clear_room(self, room_id: str):
        self.rooms.pop(room_id, None)
        self.archived.pop(room_id, None)

    def _clean_expired(self, room_id: str):
        if room_id in self.rooms:
            alive, expired = [], []
            for r in self.rooms[room_id]:
                (expired if r.is_expired() else alive).append(r)
            if expired:
                self.rooms[room_id] = alive
                self.archived[room_id].extend(expired)
                logger.info(f"[{room_id}] Archived {len(expired)} expired pending requests")
        if room_id in self.archived:
            now = time.time()
            before = len(self.archived[room_id])
            self.archived[room_id] = [r for r in self.archived[room_id]
                                       if (now - r.timestamp) <= ARCHIVE_TTL]
            pruned = before - len(self.archived[room_id])
            if pruned:
                logger.info(f"[{room_id}] Pruned {pruned} old archived requests")


# ── Response Classifier ──

# Compiled once at import time
_POSITIVE_ACK_PATTERNS = [
    # Single-word acks (anchored to full message)
    re.compile(r"^\s*(?:sure|bet|ok|okay|k|yep|yea|yeah|yes|yup|ya|ight|aight|alr|alright|word|say\s*less|gotchu|gotcha|got\s*it|got\s*u|on\s*it|will\s*do|done\s*deal|for\s*sure|def|definitely|absolutely|of\s*course|copy|roger|say\s*no\s*more|done|lol\s*ok|kk)\s*[.!]*\s*$", re.IGNORECASE),
    # Two-word ack combos: "aight bet", "yeah sure", "ok bet", "yeah ok", "sure thing"
    re.compile(r"^\s*(?:yeah|yea|yes|yep|yup|ok|okay|ight|aight|alr|alright)\s+(?:bet|sure|ok|okay|do\s*it|go|send\s*it|book\s*it|done|cool|fosho|word|on\s*it|thing)\s*[.!]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:sure|ok|yeah)\s+(?:thing|np|no\s*prob)\s*[.!]*\s*$", re.IGNORECASE),
    # Action phrases: "do it", "go ahead", "send it", "book it", "go for it", "on it"
    re.compile(r"^\s*(?:do\s*it|go\s*(?:ahead|for\s*it)|send\s*it|book\s*it|on\s*it|lets?\s*(?:go|do\s*it))\s*[.!]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:ok|yeah|sure|bet|ight|yep)\s+(?:on\s*it|done|got\s*it)\s*[.!]*\s*$", re.IGNORECASE),
    # Active confirmations with verbs
    re.compile(r"\b(?:sending|sent|booking|booked|ordering|ordered|paying|paid)\b.*\b(?:now|rn|it|that)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:sending|sent|booked|ordered|done)\s*[.!]*\s*$", re.IGNORECASE),
    re.compile(r"\b(?:ok|yeah|sure|bet|ight)\s+(?:sending|booking|ordering|doing)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'?ll|will)\s+(?:send|book|order|do|get)\s+(?:it|that|one)", re.IGNORECASE),
    re.compile(r"\blet\s*(?:me|'s)\s+(?:send|book|order|do|get)\b", re.IGNORECASE),
]

_FUTURE_PROMISE_PATTERNS = [
    re.compile(r"\b(?:next\s+(?:week|month|friday|monday|tuesday|wednesday|thursday|saturday|sunday)|tomorrow|tmrw|tmr)\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'?ll|will)|gonna|going\s+to)\s+(?:send|pay|get\s+(?:you|u)|transfer|venmo).*\b(?:later|soon|tonight|this\s+(?:week|friday|weekend))\b", re.IGNORECASE),
    re.compile(r"\b(?:when\s+i\s+get\s+paid|after\s+payday|on\s+friday|end\s+of\s+(?:the\s+)?(?:week|month))\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'?ll|will)|gonna)\s+(?:send|pay|get\s+(?:you|u)|venmo)\b.*\b(?:back|later)\b", re.IGNORECASE),
    re.compile(r"\bgive\s+me\s+(?:a\s+(?:few|couple)\s+)?(?:days?|time)\b", re.IGNORECASE),
]

_REJECTION_PATTERNS = [
    re.compile(r"^\s*(?:nah|nope|no|hell\s+no|naw|pass)\s*[.!]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*nah\s+(?:i(?:'?m|m)\s+good|i(?:'?m|m)\s+ok|not\s+(?:now|rn|today)|bro|man|fam|dude|bruh|thanks|thank\s+you)\s*[.!]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:nah|nope|no)\s+(?:i(?:'?ll|will|m\s+gonna)\s+(?:walk|drive|take\s+the\s+bus))\b", re.IGNORECASE),
    re.compile(r"^\s*(?:nah|nope|no)\s+(?:just|u\s+can|u\s+should|we\s+can|we\s+should)?\s*(?:walk|drive|take\s+the\s+bus|bike|run)\b", re.IGNORECASE),
    re.compile(r"\b(?:that\s+was\s+(?:ur|your)\s+treat|you\s+(?:said|offered)|i\s+(?:don'?t|dont)\s+owe|not\s+(?:paying|sending)|no\s+way|forget\s+it)\b", re.IGNORECASE),
    re.compile(r"\b(?:yeah\s+right|as\s+if|in\s+(?:ur|your)\s+dreams)\b.*\b(?:lmao|lol|haha|😂|🤣)\b", re.IGNORECASE),
    re.compile(r"\b(?:lmao|lol|haha|😂|🤣)\b.*\b(?:yeah\s+right|as\s+if|in\s+(?:ur|your)\s+dreams)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:yeah\s+right|as\s+if)\s*(?:lmao|lol|haha)?\s*$", re.IGNORECASE),
    re.compile(r"\bi(?:'?m|m)\s+broke\b", re.IGNORECASE),
    re.compile(r"^\s*nah\s+i(?:'?ll|will)\s+\w+", re.IGNORECASE),
]

_ALREADY_DONE_PATTERNS = [
    re.compile(r"\b(?:already|just)\s+(?:sent|paid|booked|ordered|done|did|transferred|venmo(?:ed|'d)?)\b", re.IGNORECASE),
    re.compile(r"\bdone\b.*\beta\b", re.IGNORECASE),
    re.compile(r"^\s*done\s*[.!]*\s*$", re.IGNORECASE),
    re.compile(r"\beta\s+\d+\s*(?:min|minute|m)\b", re.IGNORECASE),
    re.compile(r"\b(?:driver|cab|uber|lyft)\s+is\s+(?:here|coming|on\s+(?:the\s+)?way)\b", re.IGNORECASE),
    re.compile(r"\b(?:booked|ordered)\s+(?:it|one|the)\b.*\b(?:already|earlier|before)\b", re.IGNORECASE),
    re.compile(r"\b(?:ok\s+)?(?:booked|ordered|sent|paid)\b.*\beta\b", re.IGNORECASE),
    re.compile(r"\b(?:ok\s+)?(?:booked|ordered|sent|paid|done)\b.*\b\d+\s*(?:min|minute|m)\b", re.IGNORECASE),
]

_QUESTION_PATTERNS = [
    re.compile(r"\b(?:how\s+much|what\s+amount|what'?s\s+(?:the\s+)?(?:total|amount|damage))\b", re.IGNORECASE),
    re.compile(r"\b(?:where\s+(?:to|from|at)|which\s+(?:one|place|airport|restaurant|address))\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+time|when|what'?s\s+the\s+(?:address|time|plan))\b", re.IGNORECASE),
    re.compile(r"^\s*(?:how\s+much|where|which|when|what)\b.*\?\s*$", re.IGNORECASE),
]

_DEFERRAL_PATTERNS = [
    re.compile(r"\b(?:let(?:'?s)?\s+see|we(?:'?ll)?\s+see|maybe|not\s+sure\s+yet|idk|i\s+don'?t\s+know)\b", re.IGNORECASE),
    re.compile(r"\b(?:closer\s+to\s+the\s+time|depends|we(?:'?ll)?\s+figure\s+(?:it\s+)?out)\b", re.IGNORECASE),
    re.compile(r"\b(?:hold\s+(?:on|up)|wait|not\s+(?:yet|now|rn))\b", re.IGNORECASE),
]

_SELF_INITIATED_MONEY_PATTERNS = [
    re.compile(r"\bi(?:'?m|m)\s+(?:sending|paying|transferring|venmo-?ing)\b", re.IGNORECASE),
    re.compile(r"\b(?:just\s+)?(?:sent|paid|transferred|venmo(?:ed|'d|d)?)\s+(?:you|u|him|her|them)\b", re.IGNORECASE),
    re.compile(r"\blet\s+me\s+(?:send|pay|venmo|transfer)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'?m|m)\s+(?:gonna|going\s+to)\s+(?:send|pay|venmo|transfer)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'?ll|will)\s+(?:send|pay|venmo|transfer)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:just\s+)?(?:sending|paying)\s+(?:it|that|the)\b", re.IGNORECASE),
]

_SELF_INITIATED_RIDE_PATTERNS = [
    re.compile(r"\bi(?:'?ll|will|m\s+(?:gonna|going\s+to))\s+(?:grab|get|take|book|call|order)\s+(?:a\s+)?(?:one|cab|taxi|uber|lyft|ride)\b", re.IGNORECASE),
    re.compile(r"\b(?:grabbing|getting|taking|booking|calling)\s+(?:a\s+)?(?:cab|taxi|uber|lyft|ride)\b", re.IGNORECASE),
    re.compile(r"\b(?:let\s+me|lemme)\s+(?:grab|get|book|call)\s+(?:a\s+)?(?:one|cab|taxi|uber|lyft|ride)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'?ll|will|m\s+(?:gonna|going\s+to))\s+(?:just\s+)?(?:uber|lyft|cab)\b", re.IGNORECASE),
    re.compile(r"\b(?:lets?|lemme)\s+(?:just\s+)?(?:uber|lyft|cab)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'?m|m)\s+(?:gonna|going\s+to)\s+(?:uber|lyft|cab)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'?m|m)\s+(?:just\s+)?(?:gonna|going\s+to)\s+(?:uber|lyft|cab)\b", re.IGNORECASE),
]


# Bare greetings/filler that should never count as a response to a pending request
_GREETING_ONLY = re.compile(
    r"^(?:h[ie]y*|hi+|hello|yo+|sup|what'?s?\s*up|hola|hm+|oh|lol|lmao|haha+|ok(?:ay)?|k|yea+h?|nah|wow|damn|bruh?|dude|bro|ayo|wsg|gm|gn|morning|night|bye|later|peace|testing|test)\s*[.!?]*$",
    re.IGNORECASE,
)

_ML_RESPONSE_CLASSES = ['ack', 'reject', 'future_promise', 'question', 'already_done', 'neutral']
_ML_TO_RESPONSE_TYPE = {
    'ack': ResponseType.POSITIVE_ACK,
    'reject': ResponseType.REJECTION,
    'future_promise': ResponseType.FUTURE_PROMISE,
    'question': ResponseType.QUESTION,
    'already_done': ResponseType.ALREADY_DONE,
    'neutral': ResponseType.NEUTRAL,
}


def _ml_classify_response(model, pending_cls, current_cls) -> str:
    """Classify response using the ML response_head."""
    with torch.no_grad():
        logits = model.classify_response(
            pending_cls.unsqueeze(0),
            current_cls.unsqueeze(0),
        )
    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = probs.argmax().item()
    pred_class = _ML_RESPONSE_CLASSES[pred_idx]
    confidence = probs[pred_idx].item()
    logger.info(f"ML response classifier: {pred_class} (conf={confidence:.3f})")
    return _ML_TO_RESPONSE_TYPE[pred_class]


def _regex_classify_response(text: str) -> str:
    """Classify response using regex patterns (fallback)."""
    t = text.strip()

    if any(p.search(t) for p in _REJECTION_PATTERNS):
        return ResponseType.REJECTION

    if any(p.search(t) for p in _ALREADY_DONE_PATTERNS):
        return ResponseType.ALREADY_DONE

    if any(p.search(t) for p in _FUTURE_PROMISE_PATTERNS):
        return ResponseType.FUTURE_PROMISE

    if any(p.search(t) for p in _DEFERRAL_PATTERNS):
        return ResponseType.DEFERRAL

    if any(p.search(t) for p in _QUESTION_PATTERNS):
        return ResponseType.QUESTION

    if any(p.search(t) for p in _POSITIVE_ACK_PATTERNS):
        return ResponseType.POSITIVE_ACK

    return ResponseType.NEUTRAL


class ResponseClassifier:
    """Classifies a message as a response type in the context of a pending request."""

    @staticmethod
    def classify(text: str, has_pending_from_others: bool, model_intents: list,
                 pending_cls=None, current_cls=None, model=None) -> str:
        """
        Classify what kind of response this message is.
        Uses ML response_head when available, regex fallback otherwise.
        """
        t = text.strip()

        # Self-initiated actions — fire immediately regardless of pending (regex stays)
        if any(p.search(t) for p in _SELF_INITIATED_MONEY_PATTERNS):
            return ResponseType.SELF_INITIATED
        if any(p.search(t) for p in _SELF_INITIATED_RIDE_PATTERNS):
            return ResponseType.SELF_INITIATED

        # If no pending request from others, this can't be a response
        if not has_pending_from_others:
            if any(i in MANAGED_INTENTS for i in model_intents):
                return ResponseType.NEW_REQUEST
            return ResponseType.NEUTRAL

        # Guard: bare greetings / filler are never a real response
        tl = t.lower()
        if _GREETING_ONLY.match(tl):
            logger.info(f"Greeting guard: '{t}' forced neutral")
            return ResponseType.NEUTRAL

        # ML classification when both embeddings are available
        if pending_cls is not None and current_cls is not None and model is not None and torch is not None:
            ml_result = _ml_classify_response(model, pending_cls, current_cls)
            if ml_result == ResponseType.NEUTRAL and any(i in MANAGED_INTENTS for i in model_intents):
                return ResponseType.NEW_REQUEST
            return ml_result

        # Regex fallback
        regex_result = _regex_classify_response(text)
        if regex_result == ResponseType.NEUTRAL and any(i in MANAGED_INTENTS for i in model_intents):
            return ResponseType.NEW_REQUEST
        return regex_result


# ── Decision Engine ──

class DecisionEngine:
    """Given a pending request and a response type, decide what to do."""

    @staticmethod
    def decide(response_type: str, pending: PendingRequest = None) -> tuple[str, str]:
        """
        Returns (action, reason).
        """
        if response_type == ResponseType.SELF_INITIATED:
            return Action.FIRE, "self-initiated action"

        if response_type == ResponseType.NEW_REQUEST:
            return Action.STORE_PENDING, "new request — waiting for response"

        if not pending:
            return Action.NO_FIRE, "no pending request"

        if response_type == ResponseType.POSITIVE_ACK:
            return Action.FIRE, f"confirmed pending {pending.intent}"

        if response_type == ResponseType.FUTURE_PROMISE:
            return Action.REMINDER, f"future promise for pending {pending.intent}"

        if response_type == ResponseType.REJECTION:
            return Action.CANCEL, f"rejected pending {pending.intent}"

        if response_type == ResponseType.ALREADY_DONE:
            return Action.FIRE, f"completed — confirmed pending {pending.intent}"

        if response_type == ResponseType.QUESTION:
            return Action.KEEP_PENDING, f"clarifying question — still pending"

        if response_type == ResponseType.DEFERRAL:
            return Action.CANCEL, f"deferred — not happening now"

        return Action.KEEP_PENDING, "unclassified response — keeping pending"


# ── Conversation State Machine ──

class ConversationStateMachine:
    """
    Orchestrates room state, response classification, and decision-making.
    Called from full_pipeline() after model inference.

    For managed intents (money, ride):
      - Requests are stored as pending, intents are suppressed from the response.
      - When someone responds, the response is classified and the decision engine
        determines whether to fire, remind, cancel, or keep pending.
      - Self-initiated actions fire immediately.

    For unmanaged intents (food, alarm, etc.):
      - Pass through unchanged. Rules can be added per intent later.

    When sender is not provided:
      - Falls back to immediate firing (backward compatibility).
    """

    def __init__(self):
        self.store = RoomStateStore()
        self.classifier = ResponseClassifier()
        self.engine = DecisionEngine()

    def process(self, room_id: str, text: str, sender: str,
                model_result: dict, slots: dict = None,
                message_id: str = None, reply_to: str = None,
                model=None, current_cls=None) -> dict:
        """
        Process a message through the state machine.

        Args:
            room_id: The chat room ID
            text: The message text
            sender: Who sent this message
            model_result: Output from run_inference() + suppression pipeline
            slots: Extracted slots (amount, destination, etc.)
            message_id: Optional message ID

        Returns:
            dict with keys:
                action: what happened (fire, reminder, cancel, no_fire, keep_pending, store_pending)
                intents: list of intents that should fire (may differ from model_result)
                response_type: how the message was classified
                reason: human-readable explanation
                pending_intent: the pending intent this responded to (if any)
                conversation_state: summary for the API response
        """
        model_intents = model_result.get("intents", [])
        managed_fired = [i for i in model_intents if i in MANAGED_INTENTS]
        unmanaged_fired = [i for i in model_intents if i not in MANAGED_INTENTS]

        # No sender = can't track who's talking → fall back to immediate fire
        if not sender:
            return {
                "action": Action.FIRE if model_intents else Action.NO_FIRE,
                "intents": model_intents,
                "response_type": ResponseType.NEUTRAL,
                "reason": "no sender provided — immediate mode",
                "pending_intent": None,
                "conversation_state": None,
            }

        # No room = can't track conversation → fall back to immediate fire
        if not room_id:
            return {
                "action": Action.FIRE if model_intents else Action.NO_FIRE,
                "intents": model_intents,
                "response_type": ResponseType.NEUTRAL,
                "reason": "no room_id — immediate mode",
                "pending_intent": None,
                "conversation_state": None,
            }

        # Increment message counter on all pending requests in this room
        for p in self.store.get_pending(room_id):
            if p.sender != sender:
                p.messages_since += 1

        # Get pending requests from OTHER senders in this room
        pending_from_others = self.store.get_pending_from_others(room_id, sender)

        # ── DM vs Group matching strategy ──
        # DMs: ambient matching (only 2 people, unambiguous)
        # Groups: require reply_to to match a specific pending request
        is_dm = _is_dm(room_id)
        reply_target = None

        if reply_to:
            reply_target = self.store.find_by_message_id(room_id, reply_to)
            if reply_target and reply_target.sender == sender:
                reply_target = None  # can't respond to your own request

        if is_dm:
            # DMs: use all pending from the other person (ambient matching)
            effective_pending = pending_from_others
        else:
            # Groups: only match if reply_to points to a pending/archived request
            if reply_target:
                effective_pending = [reply_target]
            else:
                effective_pending = []

        has_pending = len(effective_pending) > 0

        # Get CLS embedding from the most relevant pending request for ML classification
        pending_cls = None
        if has_pending:
            for p in reversed(effective_pending):
                if p.cls_embedding is not None:
                    pending_cls = p.cls_embedding
                    break

        # Classify this message
        response_type = self.classifier.classify(
            text, has_pending, managed_fired,
            pending_cls=pending_cls, current_cls=current_cls, model=model,
        )

        # Find the most relevant pending request (newest matching intent, or newest overall)
        relevant_pending = None
        if has_pending:
            if reply_target:
                relevant_pending = reply_target
            else:
                # Prefer pending that matches the topic the model detected
                for p in reversed(effective_pending):
                    if p.intent in managed_fired:
                        relevant_pending = p
                        break
                # Otherwise use the most recent pending
                if not relevant_pending:
                    relevant_pending = effective_pending[-1]

        # Get decision
        action, reason = self.engine.decide(response_type, relevant_pending)

        # Execute the action
        final_intents = list(unmanaged_fired)  # unmanaged always pass through

        if action == Action.FIRE:
            if response_type == ResponseType.SELF_INITIATED:
                final_intents.extend(managed_fired)
            elif response_type == ResponseType.POSITIVE_ACK and len(pending_from_others) > 1:
                for p in pending_from_others:
                    if not p.has_responded(sender):
                        final_intents.append(p.intent)
                        p.mark_responded(sender)
            elif relevant_pending:
                final_intents.append(relevant_pending.intent)
                # Mark this sender as responded (don't remove — others may still respond in group)
                relevant_pending.mark_responded(sender)

            conv_state = {
                "status": "fired",
                "response_type": response_type,
                "reason": reason,
            }
            if relevant_pending:
                conv_state["triggered_by"] = {
                    "sender": relevant_pending.sender,
                    "text": relevant_pending.text,
                    "message_id": relevant_pending.message_id,
                    "slots": relevant_pending.slots or None,
                }

        elif action == Action.REMINDER:
            if relevant_pending:
                final_intents.append("reminder")
                relevant_pending.mark_responded(sender)

            conv_state = {
                "status": "reminder",
                "original_intent": relevant_pending.intent if relevant_pending else None,
                "response_type": response_type,
                "reason": reason,
            }
            if relevant_pending:
                conv_state["triggered_by"] = {
                    "sender": relevant_pending.sender,
                    "text": relevant_pending.text,
                    "message_id": relevant_pending.message_id,
                    "slots": relevant_pending.slots or None,
                }

        elif action == Action.CANCEL:
            if relevant_pending:
                relevant_pending.mark_responded(sender)

            conv_state = {
                "status": "cancelled",
                "response_type": response_type,
                "reason": reason,
            }
            if relevant_pending:
                conv_state["triggered_by"] = {
                    "sender": relevant_pending.sender,
                    "text": relevant_pending.text,
                    "message_id": relevant_pending.message_id,
                }

        elif action == Action.STORE_PENDING:
            # New request — store as pending, suppress managed intents from response
            for intent in managed_fired:
                money_info = model_result.get("money") if intent == "money" else {}
                self.store.add_pending(room_id, PendingRequest(
                    intent=intent,
                    sender=sender,
                    text=text,
                    slots=slots or {},
                    money_info=money_info or {},
                    message_id=message_id,
                    cls_embedding=current_cls,
                ))

            conv_state = {
                "status": "pending",
                "pending_intents": managed_fired,
                "response_type": response_type,
                "reason": reason,
            }

        elif action == Action.KEEP_PENDING:
            conv_state = {
                "status": "pending",
                "response_type": response_type,
                "reason": reason,
            }

        else:  # NO_FIRE
            conv_state = {
                "status": "no_fire",
                "response_type": response_type,
                "reason": reason,
            }

        logger.info(f"[{room_id}] {sender}: {text!r} → {response_type} → {action} ({reason}) → intents={final_intents}")

        return {
            "action": action,
            "intents": final_intents,
            "response_type": response_type,
            "reason": reason,
            "pending_intent": relevant_pending.intent if relevant_pending else None,
            "conversation_state": conv_state,
        }


# ── Singleton ──

conversation_sm = ConversationStateMachine()
