# FYOE v4 — slot-filling architecture spec

**Status:** draft for team review
**Owner:** Akash
**Last updated:** 2026-05-02

---

## 1. Why this exists

The v3 model is a multi-label intent classifier. It tells you *what* the user wants (`money`, `alarm`, `contact`, ...). It does not tell you *who*, *when*, *where*, *what about*, or *how much*.

That information lives in **slots** (also called entities, parameters, or arguments). Every action card in the FYOE UI needs them filled or it can't act:

| Card | Without slots, the card says... |
|---|---|
| Send payment | "Send $? to who?" |
| Add reminder | "Remind you about what at when?" |
| Call contact | "Call who?" |
| Add calendar event | "Title? When?" |
| Open in Maps | "Going where?" |

**A classifier without a slot filler is a half-built brain.** Ship that and users get a card with "—" everywhere, then bounce.

This document specifies how v4 fills slots, when it asks the user to clarify, and how the backend resolves ambiguity using context.

---

## 2. Slot schema (per intent)

Required slots are marked **`*`**. If a required slot can't be filled with confidence ≥ 0.8, the card **must** show a clarification prompt instead of guessing.

| Intent | Required slots | Optional slots |
|---|---|---|
| `contact` | `name *`, `action *` | — |
| `alarm` | `time *`, `note *` | `recurrence`, `lead_time` |
| `money` | `amount *`, `recipient *` | `app`, `note`, `split_count` |
| `calendar` | `title *`, `time *` | `location`, `attendees`, `duration` |
| `maps` | `destination *` | `origin`, `mode` |
| `ride` | `destination *` | `time`, `app`, `origin` |
| `food_order` | `item *` OR `app *` | `cuisine`, `time`, `location` |
| `travel` | `destination *` | `time`, `mode`, `return_date` |
| `music` | `song *` OR `artist *` | `app`, `playlist` |
| `video` | `title *` | `platform`, `time` |
| `reservation` | `place *` | `time`, `party_size` |
| `tickets` | `event *` | `date`, `count` |
| `health` | `type *` | `doctor`, `time` |
| `bills` | `type *` OR `amount *` | `due_date`, `payee` |
| `shopping` | `item *` | `qty`, `app`, `budget` |
| `weather` | — (defaults to user location) | `location`, `time` |
| `task` | `description *` | `due` |
| `note` | `content *` | `tag` |

**Notes:**
- `action` for `contact` is one of: `call`, `text`, `facetime`, `whatsapp`, `email`. Defaults to `call` if ambiguous.
- `app` is always optional — if missing, the user picks from chips on the card.
- `recipient` and `destination` can fall back to **context** before triggering a clarification prompt (see §5).

---

## 3. Architecture

```
                ┌──────────────────────────────┐
   user msg ─►  │  Step 1: Classifier (v3)     │  ─►  fired intents
                │  RoBERTa multi-label, 99% IID│
                └──────────────────────────────┘
                              │
                              ▼  for each fired intent
                ┌──────────────────────────────┐
                │  Step 2: Slot filler         │
                │  (parallel within request)   │
                │                              │
                │  Layer A: Regex / parsers   ─┼─► time, date, money, phone, email, url
                │  Layer B: Small LLM call    ─┼─► name, item, place, song, title, etc.
                │  Layer C: Context resolver  ─┼─► contact_id, default app, last destination
                └──────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │  Step 3: Card decision       │
                │                              │
                │  Required slots all filled? ─┼─► render Action Card
                │  Required slot missing?     ─┼─► render Clarification Card
                │  Confidence in slot < 0.8?  ─┼─► render with editable field
                └──────────────────────────────┘
```

**Latency budget (P95):**
- Step 1: 80 ms (model on CPU)
- Step 2 Layer A: 5 ms (deterministic)
- Step 2 Layer B: 150 ms (LLM call, parallel with A & C)
- Step 2 Layer C: 20 ms (DB lookup)
- Step 3: 5 ms
- **Total target: < 200 ms P95** end-to-end

---

## 4. Layer A — deterministic extractors

Cheap, fast, exact. Run **always**, regardless of intent. Built from these libraries:

| Slot type | Library / approach | Example input → output |
|---|---|---|
| Time, date, datetime | [`dateparser`](https://github.com/scrapinghub/dateparser) (handles "tomorrow at 6pm", "next friday", "in 20 min", "kal 6 baje") | "remind me at 6pm tomorrow" → `2026-05-03T18:00:00` |
| Money amount + currency | regex + ISO 4217 lookup | "$1,200.50", "20 bucks", "₹500", "€10" → `{amount: 1200.50, currency: "USD"}` |
| Phone number | `phonenumbers` (Google libphonenumber port) | "+1 415 555 0184" → E.164 normalized |
| Email | regex (RFC-light) | `alex@gmail.com` → `alex@gmail.com` |
| URL | regex | `https://...` → URL |
| App name | dictionary match (case-insensitive, fuzzy) | "venmo", "doordash", "spotify", "uber" → canonical app id |
| Numbers (counts) | regex on cardinal/ordinal forms | "4 people", "two", "for 6" → integers |
| Common locations | dictionary (airports, neighborhoods) | "JFK", "SFO", "downtown" → resolved place |

**Why deterministic first:** Regex is free, ~5 ms, and never hallucinates. We extract everything we can without the LLM, then only ask the LLM for the slots regex couldn't fill.

---

## 5. Layer B — LLM extractor

For free-form named entities the regex layer can't handle: contact names, song titles, restaurant names, item descriptions, vague references.

**Implementation v4:** OpenAI `gpt-4o-mini` API (or Anthropic `claude-haiku`). Single structured-output call per request.

**Prompt template:**

```
You extract structured fields from a chat message. Return JSON only.

Message: "{text}"
Detected intents: [{intent_list}]

For each intent, fill these slots if (and ONLY if) they appear in the message:
{slot_schema_for_each_intent}

Already extracted by regex (do NOT overwrite):
{layer_a_results}

Rules:
- If a slot does not clearly appear, set it to null. NEVER guess.
- Names should be exactly as written ("dad", "mom", "Sarah").
- Item descriptions should be the noun phrase ("sushi", "concert tickets").
- Return JSON: {{"contact": {{"name": "..."}}, "alarm": {{"note": "..."}}}}
```

**Why an LLM and not a NER model:**
- Off-the-shelf NER (spaCy, Flair) doesn't know "dad" is a contact name or that "Anti-Hero" is a song title — those are domain entities.
- A custom NER model would require labeling 5000+ tokens per intent and retraining for every new intent. The LLM generalizes.
- Cost: ~$0.0002 / call (gpt-4o-mini, ~200 input + 100 output tokens). At 10K msgs/day = $2/day = $60/mo. Negligible for v1.
- Latency: 100–200 ms typical. Acceptable.

**Self-host migration path (v5 / scale):** when API cost crosses ~$2K/mo (≈ 350K msgs/day), self-host Phi-3-mini-4k or Llama 3.2 1B on the same GPU as the classifier. Same prompt format works on these models.

---

## 6. Layer C — context resolver

Backend Python service that takes Layer A + B output and **resolves named references against user data**. This is the most underrated layer — it's what makes "call dad" actually call your dad.

### Resolution order (per slot type)

**`recipient` / `name` (contact resolution):**
1. Exact handle match in `contacts.handles` (`@alexkim`, `+14155550184`)
2. Fuzzy name match in `contacts` (Levenshtein ≤ 2 for short names; tokenized for long): "dad" → contact tagged `relationship: dad`
3. Thread participants: in a 1:1 chat, a missing recipient = the other person
4. Recent action history: `last_called_named["dad"]` → +1 415 555 0184
5. **No match?** → return `needs_clarification` with a picker of (a) frequent contacts, (b) thread participants, (c) "Add new contact" CTA

**`destination` (places):**
1. Exact match in user's saved places (Home, Work, Gym)
2. Geocode against Maps API
3. Recent destinations: `last_navigated`
4. **No match?** → free-text input on card, with autocomplete from Maps API

**`app` (payment, ride, music, food, etc.):**
1. Explicit in message ("via Venmo")
2. User's preferred app for that intent type (`user_prefs.payment.default = "venmo"`)
3. Show all available app chips on the card; user picks once, we save preference

### Required backend tables

```sql
-- Contacts (existing in most apps; we may need to import)
CREATE TABLE contacts (
  user_id      UUID NOT NULL,
  contact_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  display_name TEXT NOT NULL,
  relationship TEXT,           -- "dad", "mom", "sister", null
  pinned       BOOLEAN DEFAULT false,
  PRIMARY KEY (user_id, contact_id)
);

CREATE TABLE contact_handles (
  contact_id  UUID NOT NULL REFERENCES contacts(contact_id),
  channel     TEXT NOT NULL,   -- "phone", "email", "venmo", "cashapp", "whatsapp"
  handle      TEXT NOT NULL,
  PRIMARY KEY (contact_id, channel, handle)
);

-- User-side handles (for receiving payments)
CREATE TABLE payment_handles (  -- already specced with Samyak
  user_id      UUID NOT NULL,
  app          ENUM('venmo','cashapp','paypal','zelle','upi'),
  handle       TEXT NOT NULL,
  is_default   BOOLEAN DEFAULT false,
  PRIMARY KEY (user_id, app)
);

-- Saved places
CREATE TABLE places (
  user_id  UUID NOT NULL,
  label    TEXT NOT NULL,    -- "home", "work", "gym"
  address  TEXT NOT NULL,
  lat      DOUBLE PRECISION,
  lng      DOUBLE PRECISION,
  PRIMARY KEY (user_id, label)
);

-- Action history (for "last X" defaults & personalization)
CREATE TABLE action_history (
  user_id     UUID NOT NULL,
  ts          TIMESTAMP DEFAULT now(),
  intent      TEXT NOT NULL,
  slots       JSONB,
  app_used    TEXT,
  outcome     TEXT       -- "confirmed" | "dismissed" | "edited"
);

-- User prefs
CREATE TABLE user_prefs (
  user_id           UUID PRIMARY KEY,
  default_payment   TEXT,    -- "venmo"
  default_ride      TEXT,    -- "uber"
  default_food      TEXT,    -- "doordash"
  default_music     TEXT,    -- "spotify"
  language          TEXT     -- "en", "hi-en"
);
```

---

## 7. Confidence model + clarification UX

**Slot confidence is computed per-slot, not per-message:**

| Source | Confidence |
|---|---|
| Regex / dictionary exact match | 1.00 |
| LLM extracted with explicit token in message | 0.95 |
| LLM extracted via paraphrase | 0.75 |
| Context resolver: unique exact match | 0.95 |
| Context resolver: fuzzy match (1 candidate) | 0.80 |
| Context resolver: ambiguous (2+ candidates) | 0.50 |

### Decision matrix

| State | UX |
|---|---|
| All required slots ≥ 0.8 | Render full **Action Card** with a "Confirm" button |
| Required slot < 0.8 but candidates exist | Render **Disambiguation Card** with chip picker ("which dad? Raj / Vinod / + new") |
| Required slot missing entirely | Render **Clarification Card** with one inline text field per missing slot, plus chip suggestions |

**Examples:**

`"call dad"` (1 dad in contacts) →
> ✅ Action Card: "Call Dad (+1 415 555 0184)" `[Call]` `[FaceTime]` `[Message]`

`"call dad"` (2 dads in contacts) →
> ⚠ Disambiguation Card: "Which Dad?" `[Raj — last called 3d ago]` `[Vinod — pinned]`

`"call them"` →
> ❓ Clarification Card: "Call who?" `[Sarah]` `[Mom]` `[Alex]` `[+ Pick contact]`

`"remind me at 9"` →
> ❓ Clarification Card: "What's the reminder for?" `[free-text input]` &nbsp; "When?" `[9am ▼]` `[9pm ▼]`

`"send 20"` (1:1 thread with Alex) →
> ✅ Action Card: "Send $20 to Alex Kim (@alexkim)" `[Venmo]` `[Cash App]` `[PayPal]`
> *(recipient resolved from thread participants → confidence 0.85)*

---

## 8. API contracts

### `/detect` (extended)

**Request:**
```json
{
  "text": "call dad",
  "user_id": "uuid",
  "thread": {
    "thread_id": "uuid",
    "participants": ["uuid1", "uuid2"]
  }
}
```

**Response:**
```json
{
  "intents": [
    {
      "name": "contact",
      "confidence": 0.96,
      "slots": {
        "name":   {"value": "dad", "raw": "dad", "confidence": 0.95, "resolved": {"contact_id": "uuid", "display_name": "Raj Patel", "phone": "+14155550184"}},
        "action": {"value": "call", "confidence": 0.85, "source": "default"}
      },
      "missing_required": [],
      "needs_clarification": false,
      "card_type": "action"
    }
  ],
  "render_hint": {
    "primary_intent": "contact",
    "compact_card": false
  },
  "latency_ms": 187
}
```

**`needs_clarification = true` example:**
```json
{
  "intents": [{
    "name": "contact",
    "slots": {
      "name": {"value": null, "confidence": 0, "candidates": [
        {"contact_id": "u1", "display_name": "Raj Patel", "tag": "Dad"},
        {"contact_id": "u2", "display_name": "Vinod Sharma", "tag": "Dad"}
      ]}
    },
    "missing_required": ["name"],
    "needs_clarification": true,
    "card_type": "disambiguation"
  }]
}
```

### `/resolve` (new — for client-side disambiguation)

When the user picks a chip in a disambiguation card, the client posts back:
```
POST /resolve
{ "intent_id": "...", "slot": "name", "value": "u1" }
```
Server saves the choice to `action_history` and returns the now-fully-filled action card payload.

---

## 9. Cost & latency

### LLM cost (Layer B) at scale

| Volume | Daily LLM calls | Daily cost (gpt-4o-mini @ $0.0002/call) | Monthly |
|---|---:|---:|---:|
| 100 users × 50 msgs/day | 5K | $1 | $30 |
| 1K users × 50 msgs/day | 50K | $10 | $300 |
| 10K users × 50 msgs/day | 500K | $100 | $3,000 |
| 100K users × 50 msgs/day | 5M | $1,000 | $30,000 |

**Optimization tactics (apply in order as scale grows):**
1. **Skip LLM when not needed.** If all required slots are filled by Layer A + C, don't call LLM. (~30% of messages.)
2. **Cache by `(intent_set, normalized_text_hash)`.** Identical phrases hit cache. (~40% hit rate on chat traffic.)
3. **Batch LLM calls** in 50–100 ms windows during peak.
4. **Self-host** Phi-3-mini or Llama 3.2 1B when API cost > $2K/mo.

### Latency

| Stage | P50 | P95 |
|---|---:|---:|
| Classifier (CPU) | 60 ms | 90 ms |
| Layer A (regex) | 2 ms | 5 ms |
| Layer B (LLM API) | 110 ms | 220 ms |
| Layer C (DB lookup) | 8 ms | 25 ms |
| Card decision + serialize | 3 ms | 8 ms |
| **End-to-end** | **~140 ms** | **~250 ms** |

To stay under 200 ms P95: run Layer B in parallel with A and C; quantize the classifier to int8 (cuts P95 by ~30%).

---

## 10. Phased implementation

| Phase | Scope | Time |
|---|---|---|
| **v4.0 — slot filler skeleton** | Layer A (regex/dateparser/money) + Layer C (contact + place resolution). Serve in `/detect` response with current v3 classifier. Card stays the same UI, but now shows real slot values for time/date/money. | 1 week |
| **v4.1 — LLM extractor** | Add Layer B for free-form names/items. Integrate API key, prompt template, error handling, fallback to "needs clarification" on LLM failure. | 1 week |
| **v4.2 — clarification UX** | Disambiguation card, clarification card, `/resolve` endpoint. Wire `action_history` so picks persist. | 1 week |
| **v4.3 — eval harness extension** | Add slot accuracy to `eval_server.py`: per-slot precision/recall, plus full-card pass criterion (intent + all required slots correct). | 3 days |
| **v4.4 — model retrain (intent only)** | Use accumulated failures from harness to fix v3 intent classifier (negation, past tense, multi-intent, code-mixed). Slot filler stays unchanged. | 1 week |
| **v5.0 — joint model** *(optional, post-launch)* | If Layer B cost grows: train a single multi-task RoBERTa with intent head + token-level NER head. Removes LLM dependency for the common slots. | 4–6 weeks |

**Total to v4.4:** ~5 weeks of focused work.

---

## 11. Open questions for the team

1. **API provider** for Layer B — OpenAI gpt-4o-mini, Anthropic Claude Haiku, or Google Gemini Flash? (Cost is similar; pick on data-handling policy.)
2. **Where does `contacts` data come from?** Import from device contacts (iOS / Android permission)? Or only contacts the user adds in-FYOE?
3. **How aggressive is the clarification card?** Risk of "card fatigue" if we ask too often. Threshold for the 0.8 cutoff is tunable; should we A/B?
4. **Multilingual scope for v4.** If India is priority, Layer B prompt must work bilingually. Easy, but increases LLM token count by ~20%.
5. **Privacy.** LLM calls send the user's chat message to a third-party API. Acceptable for v1 with disclosure? Or do we need to redact PII before the call (numbers, emails, names)?
6. **Card fallback on LLM timeout.** If Layer B takes > 500 ms or errors, do we render a degraded card with only Layer A slots, or do we silently drop the intent? (Recommend: degraded render.)

---

## Appendix A — sample full response (alarm + contact, multi-slot)

User: `"remind me to call mom at 6pm tomorrow"`

```json
{
  "intents": [
    {
      "name": "alarm",
      "confidence": 0.98,
      "slots": {
        "time": {"value": "2026-05-03T18:00:00", "raw": "6pm tomorrow", "confidence": 1.0, "source": "regex"},
        "note": {"value": "call mom", "raw": "call mom", "confidence": 0.95, "source": "llm"}
      },
      "missing_required": [],
      "needs_clarification": false,
      "card_type": "action"
    },
    {
      "name": "contact",
      "confidence": 0.92,
      "slots": {
        "name":   {"value": "mom", "raw": "mom", "confidence": 0.95, "resolved": {"contact_id": "u4", "display_name": "Linda Patel", "phone": "+14155550142"}, "source": "context"},
        "action": {"value": "call", "raw": "call", "confidence": 1.0, "source": "regex"}
      },
      "missing_required": [],
      "needs_clarification": false,
      "card_type": "action"
    }
  ],
  "latency_ms": 192
}
```

The client renders **two stacked cards** in the chat thread: a reminder card for tomorrow 6pm, and a contact card with `[Call Mom]` button.

---

*End of spec.*
