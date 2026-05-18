"""
Conversation context accumulator for multi-turn intent detection.

The model classifies single messages. But in real chat, meaning depends on
what came before:

    User A: "should we order pizza?"          -> food_order
    User B: "yeah do it"                      -> ??? (bare model says nothing)
    (with context)                            -> food_order (carried forward)

    User A: "uber to JFK"                     -> ride, maps
    User B: "actually make it lyft"           -> ride (correction, not new)

The context accumulator sits BETWEEN the trigger filter and the model.
It detects follow-ups, confirmations, and corrections -- and either:
  (a) carries forward the prior intent (no model call needed), or
  (b) prepends context to the model input so the model has more signal.

Architecture:
    message -> trigger_filter -> context_accumulator -> model -> slots -> action
                                      |
                              conversation history
                              (last N messages + their intents)

Two operating modes:
  - Pipeline mode (NOW): Single-turn model + heuristic follow-up detection.
    Confirmations are carried forward without a model call. Everything else
    goes to the model as usual (single message, no context prepended).

  - Native mode (v5): Model retrained on multi-turn data. The accumulator
    prepends prior turns with </s> separators and the model handles it.
    Set native_mode=True after retraining.

Usage:
    from context_accumulator import ConversationContext

    ctx = ConversationContext()

    # Classify with context awareness
    analysis = ctx.analyze("room1", "should we order pizza?")
    # -> AnalysisResult(type="independent", model_input="should we order pizza?", ...)

    result = run_model([analysis.model_input])[0]
    ctx.record("room1", "should we order pizza?", fired=result["fired"], slots=result["slots"])

    # Follow-up
    analysis = ctx.analyze("room1", "yeah do it")
    # -> AnalysisResult(type="confirmation", carry_forward={"fired": ["food_order"], ...}, ...)
    # No model call needed! Use carry_forward directly.

    ctx.record("room1", "yeah do it", fired=["food_order"], slots={...})
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Turn: one message in a conversation
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    text: str
    sender: Optional[str] = None
    fired: list[str] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    source: str = "model"  # "model" | "carried_forward" | "correction"
    handled: bool = False   # True once intent was confirmed/acknowledged


# ---------------------------------------------------------------------------
# AnalysisResult: what the accumulator returns for each new message
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    """What the accumulator decides about a new message."""
    type: str
    # "independent"   — new actionable or casual; run model normally
    # "confirmation"  — bare affirmative; carry forward prior intents
    # "correction"    — modifies a slot from the prior intent
    # "addition"      — adds new intent on top of prior context
    # "reference"     — references prior context ("do that again", "same thing")

    model_input: str
    # What to feed the model. For independent messages, this is just the text.
    # For native_mode, this may include prior turns with </s> separators.

    carry_forward: Optional[dict] = None
    # If type="confirmation", this is the prior result to reuse.
    # Keys: fired (list[str]), slots (dict), original_text (str)

    prior_intents: list[str] = field(default_factory=list)
    # What fired in the last actionable turn (for debugging/logging)

    context_turns: int = 0
    # How many prior turns were considered

    correction_detail: Optional[dict] = None
    # If type="correction", what changed. e.g. {"slot": "amount", "old": "20", "new": "30"}


# ---------------------------------------------------------------------------
# Patterns for detecting follow-ups
# ---------------------------------------------------------------------------

# Bare confirmations — messages that mean "yes, do the last thing"
_CONFIRM_EXACT = {
    # English
    "yes", "yeah", "yep", "yup", "ya", "yea", "ye",
    "sure", "ok", "okay", "k", "kk", "kay",
    "bet", "bet bet", "aight", "ight", "aii",
    "absolutely", "definitely", "totally", "for sure",
    "sounds good", "sounds great", "sounds perfect",
    "lets go", "let's go", "lessgo", "lesgo",
    "do it", "send it", "book it", "go ahead",
    "lets do it", "let's do it",
    "go for it", "yolo",
    "please", "pls", "plz",
    "cool", "dope", "fire",
    "done", "deal",
    # Status confirmations
    "sent", "on it",
    # Hindi/Hinglish
    "haan", "ha", "theek hai", "thik hai", "chal", "chalo",
    "kar de", "kar do", "kardo", "karde",
    "bhej de", "bhej do", "bhejde", "bhejdo",
    "bol de", "bolde", "bol do",
}

# Patterns — slightly more complex confirmations
_CONFIRM_PATTERNS = re.compile(
    r"^(?:"
    r"yeah?\s+(?:do it|go ahead|send it|book it|let'?s go|please|pls|sure)"
    r"|yes\s+(?:please|pls|plz|do it|go ahead)"
    r"|sure\s+(?:thing|why not|go ahead|do it)"
    r"|ok(?:ay)?\s+(?:do it|go ahead|send it|book it|let'?s go|bet)"
    r"|(?:yeah?|yep|yup|sure|ok)\s*[!.]*"
    r"|lgtm"
    r"|(?:do|send|book|order|pay|call)\s+(?:it|that)"
    r"|(?:go|let'?s)\s+(?:go|do it|for it)"
    r")$",
    re.IGNORECASE,
)

# Correction starters — "actually X", "wait X", "no make it X"
_CORRECTION_STARTERS = re.compile(
    r"^(?:"
    r"actually\b"
    r"|wait\b"
    r"|no[,.]?\s+"
    r"|nah[,.]?\s+"
    r"|change\s+(?:it\s+)?to\b"
    r"|make\s+it\b"
    r"|switch\s+(?:it\s+)?to\b"
    r"|not\s+\w+[,.]?\s+(?:but|make|change)"
    r")",
    re.IGNORECASE,
)

# Addition starters — "and also X", "oh and X", "plus X"
_ADDITION_STARTERS = re.compile(
    r"^(?:"
    r"(?:and\s+)?also\b"
    r"|oh\s+and\b"
    r"|and\s+(?:can you|could you|pls|please)?\s*"
    r"|plus\b"
    r"|one\s+more\s+thing\b"
    r"|btw\s+"
    r")",
    re.IGNORECASE,
)

# Filler / acknowledgment messages — should be skipped in context string.
# These add noise without semantic value for intent detection.
_FILLER_EXACT = {
    # Acknowledgments
    "ok", "okay", "k", "kk", "kay", "got it", "gotcha", "noted",
    "thanks", "thank you", "ty", "thx", "np", "no problem",
    "cool", "nice", "great", "awesome", "perfect", "dope", "fire", "lit",
    # Reactions
    "lol", "lmao", "lmfao", "haha", "hahaha", "ha", "rofl",
    "wow", "whoa", "damn", "dang", "yikes", "oof", "bruh",
    "omg", "smh", "ikr", "fr", "facts", "word", "true",
    # Minimal responses
    "same", "mood", "bet", "ight", "aight", "nah", "nope",
    "yeah", "yep", "yup", "ya", "yea", "ye", "yes",
    "sure", "for sure", "fs", "absolutely", "definitely",
    "sounds good", "sounds great", "sounds perfect",
    # Greetings / closings
    "hey", "hi", "yo", "sup", "whats up", "nm", "nothing much",
    "bye", "later", "ttyl", "gn", "good night",
    # Generic filler
    "idk", "hmm", "hm", "mhm", "uh", "um",
    "no cap", "deadass", "slay", "period", "say less",
    "sent", "done", "deal", "on it",
    # Reactions to updates
    "got it thanks", "got it", "gotcha thanks",
    # Common compound greetings / filler
    "hey whats up", "hey what's up", "yo whats up", "what's good",
    "nm you", "nothing much you", "not much",
    "lmaooo", "lmaoo", "hahahaha", "hahah",
}

# Rejection / deferral words — these mark prior intents as handled
# (the user said no or pushed it off, so the intent is no longer active)
_REJECTION_EXACT = {
    "nah", "nope", "no", "not now", "not yet",
    "maybe later", "later", "pass", "im good", "i'm good",
    "nah im good", "nah i'm good", "no im good", "no i'm good",
    "nah its fine", "nah it's fine", "no its fine", "no it's fine",
    "never mind", "nevermind", "nvm", "forget it",
    "cancel", "cancel that", "scratch that", "nah forget it",
    "changed my mind", "not anymore", "dont worry about it",
    "don't worry about it",
}

# Cancellation patterns — more complex cancellations
_CANCEL_PATTERNS = re.compile(
    r"^(?:"
    r"(?:wait\s+)?n(?:ah|vm|evermind)\b"
    r"|(?:actually\s+)?(?:forget|cancel|scratch|skip)\s+(?:it|that)\b"
    r"|(?:nah|no)\s+(?:let'?s\s+)?(?:not|just|stay|skip)\b"
    r"|(?:nah|no)\s+(?:i'?m\s+)?(?:good|fine|ok|okay|all\s*good)\b"
    r"|(?:wait\s+)?(?:let'?s\s+)?(?:just\s+)?stay\s+home\b"
    r"|changed?\s+(?:my\s+)?mind"
    r"|not\s+(?:anymore|any\s*more|now|yet|today|tonight)\b"
    r"|don'?t\s+(?:worry|bother)\b"
    r")",
    re.IGNORECASE,
)

# Pure filler: acknowledgments that should NEVER carry forward intents.
# Computed as filler words minus confirmation words (which CAN carry forward).
# "nice", "lol", "thanks" = pure filler (never actionable).
# "yeah", "ok", "bet" = confirmations (CAN carry forward intents).
_PURE_FILLER = _FILLER_EXACT - _CONFIRM_EXACT

# Reference patterns — "same thing", "that again", "the usual"
_REFERENCE_PATTERNS = re.compile(
    r"(?:"
    r"\bsame\s+(?:thing|one|order|place)\b"
    r"|\bthe\s+usual\b"
    r"|\bthat\s+again\b"
    r"|\blike\s+(?:last\s+time|before|yesterday)\b"
    r"|\brepeat\s+(?:that|it|the|my)\b"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# ConversationContext
# ---------------------------------------------------------------------------
class ConversationContext:
    """Manages multi-turn context for all active conversations.

    Args:
        window_size: how many prior turns to keep per conversation.
        stale_seconds: evict conversations with no activity for this long.
        native_mode: if True, prepend prior turns with </s> for a model
                     trained on multi-turn data (v5+). If False (default),
                     use heuristic follow-up detection with carry-forward.
    """

    def __init__(
        self,
        window_size: int = 5,
        stale_seconds: int = 1800,  # 30 minutes
        native_mode: bool = False,
    ):
        self.window_size = window_size
        self.stale_seconds = stale_seconds
        self.native_mode = native_mode
        self._convos: dict[str, list[Turn]] = {}

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def analyze(self, chat_id: str, text: str, sender: Optional[str] = None) -> AnalysisResult:
        """Analyze a new message in context. Does NOT record it — call
        record() after classification to store the result.

        Returns an AnalysisResult telling the caller:
          - whether to run the model or carry forward
          - what model_input to use
          - prior intent context for debugging
        """
        text = text.strip()
        history = self._convos.get(chat_id, [])
        last_actionable = self._last_actionable(history)

        # No prior context — always independent
        if not history or not last_actionable:
            return AnalysisResult(
                type="independent",
                model_input=self._format_input(text, history),
                prior_intents=[],
                context_turns=len(history),
            )

        # Check if too much time passed (>5 min since last message = new context)
        if history and (time.time() - history[-1].ts > 300):
            return AnalysisResult(
                type="independent",
                model_input=self._format_input(text, []),  # fresh context
                prior_intents=[],
                context_turns=0,
            )

        prior_intents = last_actionable.fired
        prior_slots = last_actionable.slots

        # 1. Confirmation check — bare affirmative after actionable intent
        if self._is_confirmation(text):
            return AnalysisResult(
                type="confirmation",
                model_input=text,
                carry_forward={
                    "fired": list(prior_intents),
                    "slots": dict(prior_slots),
                    "original_text": last_actionable.text,
                },
                prior_intents=prior_intents,
                context_turns=len(history),
            )

        # 2. Correction check — modifies the prior intent
        correction = self._detect_correction(text, last_actionable)
        if correction:
            return AnalysisResult(
                type="correction",
                model_input=text,
                carry_forward={
                    "fired": list(prior_intents),
                    "slots": correction["updated_slots"],
                    "original_text": last_actionable.text,
                },
                prior_intents=prior_intents,
                context_turns=len(history),
                correction_detail=correction["detail"],
            )

        # 3. Addition check — new intent on top of existing context
        if _ADDITION_STARTERS.match(text):
            return AnalysisResult(
                type="addition",
                model_input=self._format_input(text, history),
                prior_intents=prior_intents,
                context_turns=len(history),
            )

        # 4. Reference check — "same thing", "the usual"
        if _REFERENCE_PATTERNS.search(text):
            return AnalysisResult(
                type="reference",
                model_input=text,
                carry_forward={
                    "fired": list(prior_intents),
                    "slots": dict(prior_slots),
                    "original_text": last_actionable.text,
                },
                prior_intents=prior_intents,
                context_turns=len(history),
            )

        # 5. Default — independent message
        return AnalysisResult(
            type="independent",
            model_input=self._format_input(text, history),
            prior_intents=prior_intents,
            context_turns=len(history),
        )

    def record(
        self,
        chat_id: str,
        text: str,
        sender: Optional[str] = None,
        fired: Optional[list[str]] = None,
        slots: Optional[dict] = None,
        source: str = "model",
    ) -> None:
        """Record a classified message into the conversation history.
        Call this AFTER classification so future messages have context.

        Also marks prior actionable turns as "handled" when the current
        message is a filler/confirmation — so they stop bleeding into
        future context.
        """
        if chat_id not in self._convos:
            self._convos[chat_id] = []

        history = self._convos[chat_id]

        # Mark prior actionable turns as handled if this message is
        # a confirmation/acknowledgment (the intent was acted on)
        text_clean = text.strip().lower().rstrip("!.?")
        is_filler = text_clean in _FILLER_EXACT or len(text_clean) <= 2
        is_confirm = self._is_confirmation(text)

        if is_filler or is_confirm:
            # Walk backward and mark the most recent unhandled actionable turn
            for turn in reversed(history):
                if turn.fired and not turn.handled:
                    turn.handled = True
                    break

        # Rejection / cancellation: mark ALL prior unhandled intents as handled
        # (the user said no / changed mind — all prior actionable intents are dead)
        is_rejection = text_clean in _REJECTION_EXACT
        is_cancel = bool(_CANCEL_PATTERNS.match(text_clean))
        if is_rejection or is_cancel:
            for turn in reversed(history):
                if turn.fired and not turn.handled:
                    turn.handled = True

        # Same-intent chaining: if this message fires intent X,
        # mark prior turns that also fired X as handled.
        # This prevents intent chains (money→money→money) from bloating context
        # and self-corrects context bleed (if A bleeds X into B, marking A as
        # handled removes the source of bleed for future messages).
        if fired:
            fired_set = set(fired)
            for turn in reversed(history):
                if turn.fired and not turn.handled:
                    if set(turn.fired) & fired_set:
                        turn.handled = True

        history.append(Turn(
            text=text,
            sender=sender,
            fired=fired or [],
            slots=slots or {},
            source=source,
        ))

        # Trim to window size
        if len(history) > self.window_size:
            self._convos[chat_id] = history[-self.window_size:]

    def get_history(self, chat_id: str) -> list[Turn]:
        """Return the conversation history for a chat."""
        return list(self._convos.get(chat_id, []))

    def clear(self, chat_id: str) -> None:
        """Clear context for a conversation."""
        self._convos.pop(chat_id, None)

    def evict_stale(self) -> int:
        """Remove conversations with no activity for stale_seconds. Returns count evicted."""
        now = time.time()
        stale = [
            cid for cid, turns in self._convos.items()
            if turns and (now - turns[-1].ts) > self.stale_seconds
        ]
        for cid in stale:
            del self._convos[cid]
        return len(stale)

    @property
    def active_conversations(self) -> int:
        return len(self._convos)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _last_actionable(self, history: list[Turn]) -> Optional[Turn]:
        """Find the most recent turn that actually fired an intent.
        Skips casual messages (fired=[]) in between."""
        for turn in reversed(history):
            if turn.fired:
                # Don't carry forward if it's too old (>5 min)
                if time.time() - turn.ts > 300:
                    return None
                return turn
        return None

    def _is_confirmation(self, text: str) -> bool:
        """Is this a bare confirmation/affirmative?"""
        clean = text.strip().lower().rstrip("!.?")
        # Exact match
        if clean in _CONFIRM_EXACT:
            return True
        # Pattern match
        if _CONFIRM_PATTERNS.match(text.strip()):
            return True
        # Too long to be a confirmation (>6 words is likely independent)
        if len(clean.split()) > 6:
            return False
        return False

    def _detect_correction(self, text: str, prior: Turn) -> Optional[dict]:
        """Detect if the message corrects a slot from the prior intent.
        Returns {"detail": {...}, "updated_slots": {...}} or None."""
        if not _CORRECTION_STARTERS.match(text):
            return None
        if not prior.fired:
            return None

        updated_slots = dict(prior.slots)
        detail = {}

        # Amount correction: "actually make it 30", "wait $50"
        amount_match = re.search(r'[\$]?\d+(?:\.\d{1,2})?', text)
        if amount_match and any(i in ("money", "bills") for i in prior.fired):
            old_amount = updated_slots.get("amount") or updated_slots.get("money", {}).get("amount")
            new_amount = amount_match.group(0)
            if old_amount != new_amount:
                # Update amount in the right place
                if "money" in updated_slots and isinstance(updated_slots["money"], dict):
                    updated_slots["money"]["amount"] = new_amount
                else:
                    updated_slots["amount"] = new_amount
                detail = {"slot": "amount", "old": old_amount, "new": new_amount}
                return {"detail": detail, "updated_slots": updated_slots}

        # App/service correction: "actually lyft", "switch to uber"
        app_patterns = {
            "ride": ["uber", "lyft", "ola", "rapido"],
            "food_order": ["doordash", "ubereats", "grubhub", "swiggy", "zomato", "postmates"],
            "music": ["spotify", "apple music", "youtube music"],
            "money": ["venmo", "cashapp", "zelle", "paypal", "paytm", "gpay"],
        }
        text_lower = text.lower()
        for intent, apps in app_patterns.items():
            if intent in prior.fired:
                for app in apps:
                    if app in text_lower:
                        old_app = None
                        if intent in updated_slots and isinstance(updated_slots[intent], dict):
                            old_app = updated_slots[intent].get("service") or updated_slots[intent].get("app")
                            updated_slots[intent]["service"] = app
                        else:
                            old_app = updated_slots.get("service") or updated_slots.get("app")
                            updated_slots["service"] = app
                        detail = {"slot": "service", "old": old_app, "new": app}
                        return {"detail": detail, "updated_slots": updated_slots}

        # Time correction: "no 5pm", "make it tomorrow"
        time_words = ["am", "pm", "oclock", "o'clock", "tomorrow", "tonight", "morning", "evening"]
        if any(w in text_lower for w in time_words):
            if any(i in ("alarm", "calendar", "ride") for i in prior.fired):
                # We detect the correction but don't parse the time here —
                # the slot filler will handle it when re-run on the correction text
                detail = {"slot": "time", "old": "prior", "new": "re-extract needed"}
                return {"detail": detail, "updated_slots": updated_slots}

        # Name/recipient correction: "actually send it to sarah"
        name_match = re.search(r'(?:to|for)\s+([A-Z][a-z]+)', text)
        if name_match and any(i in ("money", "contact") for i in prior.fired):
            new_name = name_match.group(1)
            old_name = updated_slots.get("recipient") or updated_slots.get("name")
            if old_name != new_name:
                if "money" in prior.fired:
                    updated_slots["recipient"] = new_name
                else:
                    updated_slots["name"] = new_name
                detail = {"slot": "recipient", "old": old_name, "new": new_name}
                return {"detail": detail, "updated_slots": updated_slots}

        # Generic correction detected (starts with "actually" etc.) but we
        # can't figure out what changed — fall through to model
        return None

    def _format_input(self, text: str, history: list[Turn]) -> str:
        """Format model input text. In pipeline mode, just returns the text.
        In native mode (v5), returns current text (context is separate)."""
        return text

    def get_context_string(self, chat_id: str, current_text: str = None) -> Optional[str]:
        """Get a SMART context string for the tokenizer's pair encoding.

        Instead of blindly piping last N messages, this filters out:
          - Filler / acknowledgment messages ("ok", "nice", "lol")
          - Messages whose intents were already handled (confirmed)
          - Very old context (only keeps last 2 meaningful messages)

        If current_text is provided, returns None for:
          - Rejections / cancellations ("nah", "maybe later", "nvm")
          - Pure filler ("nice", "lol", "thanks") that isn't a confirmation

        Returns " | "-joined prior messages (matching training data format),
        or None if no useful context.
        """
        # If current message is a rejection/cancellation, don't pollute with
        # prior context (prevents "maybe later" after "order pizza" → false fire)
        if current_text:
            ct = current_text.strip().lower().rstrip("!.?")
            if ct in _REJECTION_EXACT or _CANCEL_PATTERNS.match(ct):
                return None
            # Pure filler (reactions, acknowledgments) shouldn't get context
            # (prevents "nice" after "order pizza" → false food_order)
            # But confirmations ("yeah", "ok", "bet") SHOULD get context
            if ct in _PURE_FILLER:
                return None

        history = self._convos.get(chat_id, [])
        if not history:
            return None

        # Walk backward, collect up to 2 meaningful messages
        meaningful = []
        for turn in reversed(history):
            # Skip stale messages (>5 min old)
            if time.time() - turn.ts > 300:
                break

            text_clean = turn.text.strip().lower().rstrip("!.?")

            # Skip pure filler / acknowledgment messages
            if text_clean in _FILLER_EXACT:
                continue

            # Skip very short non-actionable messages (1-2 chars like "?", "k")
            if len(text_clean) <= 2:
                continue

            # Skip messages whose intents were already handled/confirmed
            # (they bleed into future messages otherwise)
            if turn.handled and turn.fired:
                continue

            meaningful.append(turn.text)
            if len(meaningful) >= 2:
                break

        if not meaningful:
            return None

        meaningful.reverse()
        return " | ".join(meaningful)


# ---------------------------------------------------------------------------
# Convenience: module-level singleton for simple usage
# ---------------------------------------------------------------------------
_default_ctx: Optional[ConversationContext] = None


def get_context(
    window_size: int = 5,
    stale_seconds: int = 1800,
    native_mode: bool = False,
) -> ConversationContext:
    """Get or create the module-level ConversationContext singleton."""
    global _default_ctx
    if _default_ctx is None:
        _default_ctx = ConversationContext(
            window_size=window_size,
            stale_seconds=stale_seconds,
            native_mode=native_mode,
        )
    return _default_ctx


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ctx = ConversationContext()
    passed = 0
    total = 0

    def _test(chat_id, text, expected_type, desc, fired=None, slots=None, record_fired=None):
        global passed, total
        total += 1
        analysis = ctx.analyze(chat_id, text)
        ok = analysis.type == expected_type
        status = "ok" if ok else "FAIL"
        if not ok:
            print(f"  {status}: {desc}")
            print(f"    text: {text!r}")
            print(f"    expected: {expected_type}, got: {analysis.type}")
        else:
            passed += 1
        # Record the turn for future context
        ctx.record(
            chat_id, text,
            fired=record_fired if record_fired is not None else (fired or []),
            slots=slots or {},
            source="model" if analysis.type == "independent" else analysis.type,
        )

    # ── Conversation 1: pizza order + confirmation ──
    _test("c1", "should we order pizza?", "independent", "first msg is always independent",
          record_fired=["food_order"])
    _test("c1", "yeah do it", "confirmation", "bare confirmation after food_order")
    _test("c1", "how was your day", "independent", "casual after confirmation",
          record_fired=[])

    # ── Conversation 2: money + correction ──
    _test("c2", "venmo sarah 20", "independent", "first msg is independent",
          record_fired=["money"], slots={"amount": "20", "recipient": "sarah"})
    _test("c2", "actually make it 30", "correction", "amount correction")
    _test("c2", "sure thing", "confirmation", "confirmation after correction")

    # ── Conversation 3: ride + app correction ──
    _test("c3", "uber to JFK", "independent", "ride request",
          record_fired=["ride", "maps"], slots={"service": "uber", "destination": "JFK"})
    _test("c3", "actually make it lyft", "correction", "app correction")

    # ── Conversation 4: addition ──
    _test("c4", "order pizza from dominos", "independent", "food order",
          record_fired=["food_order"])
    _test("c4", "and also remind me to take meds", "addition", "addition of new intent")

    # ── Conversation 5: reference ──
    _test("c5", "the usual uber to work", "independent", "ride request",
          record_fired=["ride", "maps"])
    _test("c5", "same thing tomorrow", "reference", "reference to prior")

    # ── Conversation 6: independent after non-actionable ──
    _test("c6", "hey whats up", "independent", "casual",
          record_fired=[])
    _test("c6", "yeah", "independent", "yeah after non-actionable = not confirmation",
          record_fired=[])

    # ── Conversation 7: multi-word confirmations ──
    _test("c7", "can you venmo me 30", "independent", "money request",
          record_fired=["money"])
    _test("c7", "yes please", "confirmation", "yes please = confirmation")

    _test("c8", "book uber to airport", "independent", "ride request",
          record_fired=["ride"])
    _test("c8", "go ahead", "confirmation", "go ahead = confirmation")

    _test("c9", "order sushi tonight", "independent", "food order",
          record_fired=["food_order"])
    _test("c9", "let's do it", "confirmation", "let's do it = confirmation")

    # ── Conversation 10: not a confirmation — too long / has content ──
    _test("c10", "venmo me 20", "independent", "money",
          record_fired=["money"])
    _test("c10", "I already paid you back yesterday dude", "independent",
          "long reply is independent, not confirmation", record_fired=[])

    # ── Conversation 11: hinglish confirmations ──
    _test("c11", "uber book kar airport tak", "independent", "ride",
          record_fired=["ride"])
    _test("c11", "haan kar de", "confirmation", "hinglish yes")

    print(f"\ncontext accumulator: {passed}/{total} passed")
