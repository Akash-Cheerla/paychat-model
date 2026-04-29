"""
FYOE v3 — expanded intent taxonomy.

Canonical source of truth for:
  - the full 18-intent list
  - per-intent payload schema (what the API returns)
  - per-intent action-URL builder (deep links / web URLs)
  - per-intent training templates (for synthetic data generation)

Both `training/generate_data.py` and `app.py` import from this module so
adding a new intent is a one-place change.

Design principles:
  1. **Multi-label** — a single message can fire multiple intents.
     "dinner at french laundry friday at 8" → calendar + reservation + maps
  2. **Action-ready** — every payload exposes an `action_url` that the
     client can open without further parsing.
  3. **Provider-agnostic** — payloads include enough info that the client
     can pick the user's preferred provider (Venmo vs Zelle, Uber vs Lyft,
     DoorDash vs UberEats).
"""

from typing import Dict, List, Tuple
from urllib.parse import quote_plus


# ═════════════════════════════════════════════════════════════════════
#  Intent taxonomy — ORDER MATTERS (it indexes the model's output heads)
# ═════════════════════════════════════════════════════════════════════

INTENTS_V3: List[str] = [
    # — original 5 (don't reorder; preserves back-compat with v2 weights) —
    "money",        # P2P payments, debts, splits  → Venmo / Zelle / Apple Pay
    "alarm",        # reminders, wake-up           → AlarmClock / Reminders
    "contact",      # phone numbers to save        → Contacts
    "calendar",     # meetings/events with date    → Google Calendar
    "maps",         # places, addresses            → Google Maps

    # — v3 additions (13 new) —
    "food_order",   # food delivery / ordering     → DoorDash / UberEats
    "ride",         # ride-hailing                 → Uber / Lyft
    "travel",       # flights / hotels / trips     → Skyscanner / Booking
    "shopping",     # buy / order things           → Amazon
    "music",        # songs / playlists            → Spotify / Apple Music
    "video",        # movies / TV / streaming      → Netflix / Prime
    "tickets",      # events, concerts, sports     → Ticketmaster / SeatGeek
    "reservation",  # restaurant / salon bookings  → OpenTable / Resy
    "task",         # to-dos                       → Reminders / Notion
    "note",         # save info / link / quote     → Notes
    "bills",        # rent, utilities, subs        → Bill pay
    "health",       # pharmacy, doctor, meds       → CVS / Walgreens / 1mg
    "weather",      # weather lookups              → Weather app
]

NUM_LABELS_V3 = len(INTENTS_V3)


# ═════════════════════════════════════════════════════════════════════
#  Payload schemas (documentation; runtime extractors live in app.py)
# ═════════════════════════════════════════════════════════════════════

PAYLOAD_SCHEMA: Dict[str, Dict[str, str]] = {
    "money":       {"amount": "str?", "trigger_type": "str", "direction": "str"},
    "alarm":       {"label": "str?", "time_iso": "str?", "time_phrase": "str?"},
    "contact":     {"phone": "str?", "name_hint": "str?"},
    "calendar":    {"title": "str?", "start_iso": "str?", "start_phrase": "str?",
                    "duration_minutes": "int"},
    "maps":        {"place": "str?"},

    "food_order":  {"item": "str?", "cuisine": "str?", "provider_hint": "str?"},
    "ride":        {"pickup": "str?", "dropoff": "str?", "provider_hint": "str?"},
    "travel":      {"destination": "str?", "origin": "str?", "trip_type": "str?",
                    "when_phrase": "str?"},
    "shopping":    {"item": "str?", "qty": "str?"},
    "music":       {"track": "str?", "artist": "str?", "playlist": "str?"},
    "video":       {"title": "str?", "kind": "str?"},   # kind ∈ movie | show | trailer
    "tickets":     {"event": "str?", "venue": "str?", "when_phrase": "str?"},
    "reservation": {"venue": "str?", "party_size": "int?", "when_phrase": "str?"},
    "task":        {"title": "str?", "due_phrase": "str?"},
    "note":        {"content": "str?", "url": "str?"},
    "bills":       {"kind": "str?", "amount": "str?", "due_phrase": "str?"},
    "health":      {"need": "str?", "kind": "str?"},    # kind ∈ pharmacy | doctor | meds
    "weather":     {"location": "str?", "when_phrase": "str?"},
}


# ═════════════════════════════════════════════════════════════════════
#  ACTION URL BUILDERS — every payload becomes a one-tap deep link
# ═════════════════════════════════════════════════════════════════════

def _q(v) -> str:
    return quote_plus(str(v)) if v is not None else ""


def build_action_url(intent: str, payload: dict, targeting: dict | None = None) -> str | None:
    """
    Return a single ready-to-fire URL for the given intent + payload.
    The client just opens it. No further parsing required.

    Returns None if there's not enough info to build a useful URL.
    """
    p = payload or {}
    t = targeting or {}
    name = t.get("third_party") or t.get("addressee") or ""

    if intent == "money":
        amt = p.get("amount")
        amt_clean = ""
        if amt:
            # strip leading $ / currency words
            amt_clean = "".join(ch for ch in str(amt) if ch.isdigit() or ch == ".")
        parts = ["txn=pay"]
        if amt_clean:
            parts.append(f"amount={amt_clean}")
        if name:
            parts.append(f"recipients={_q(name)}")
        return "https://venmo.com/?" + "&".join(parts)

    if intent == "alarm":
        # iOS Shortcuts can take a URL like shortcuts://create-alarm
        # but for cross-platform we return a calendar-style web link with the time.
        label = p.get("label") or "Reminder"
        when = p.get("time_iso")
        if not when:
            return None
        return (
            "https://calendar.google.com/calendar/u/0/r/eventedit?"
            f"text={_q(label)}&dates={_q(when.replace(':','').replace('-','') + '00')}/"
            f"{_q(when.replace(':','').replace('-','') + '00')}"
        )

    if intent == "contact":
        phone = p.get("phone")
        if not phone:
            return None
        return "tel:" + str(phone).replace(" ", "")

    if intent == "calendar":
        title = p.get("title") or "Event"
        start = p.get("start_iso")
        params = [f"text={_q(title)}"]
        if start:
            try:
                from datetime import datetime, timedelta
                dt = datetime.fromisoformat(start)
                end = dt + timedelta(minutes=int(p.get("duration_minutes") or 30))
                fmt = "%Y%m%dT%H%M%S"
                params.append(f"dates={dt.strftime(fmt)}/{end.strftime(fmt)}")
            except Exception:
                pass
        return "https://calendar.google.com/calendar/u/0/r/eventedit?" + "&".join(params)

    if intent == "maps":
        place = p.get("place")
        if not place:
            return None
        return "https://www.google.com/maps/search/?api=1&query=" + _q(place)

    if intent == "food_order":
        q = p.get("item") or p.get("cuisine") or "food"
        provider = (p.get("provider_hint") or "").lower()
        if "uber" in provider:
            return "https://www.ubereats.com/search?q=" + _q(q)
        return "https://www.doordash.com/search/store/" + _q(q)

    if intent == "ride":
        pickup = p.get("pickup") or ""
        dropoff = p.get("dropoff") or ""
        provider = (p.get("provider_hint") or "").lower()
        if "lyft" in provider:
            return "https://ride.lyft.com/?destination%5Baddress%5D=" + _q(dropoff)
        return ("https://m.uber.com/ul/?action=setPickup"
                f"&pickup={_q(pickup) or 'my_location'}"
                f"&dropoff%5Bformatted_address%5D={_q(dropoff)}")

    if intent == "travel":
        dest = p.get("destination")
        if not dest:
            return None
        return "https://www.google.com/travel/explore?q=" + _q(dest)

    if intent == "shopping":
        q = p.get("item") or ""
        if not q:
            return None
        return "https://www.amazon.com/s?k=" + _q(q)

    if intent == "music":
        q = p.get("track") or p.get("artist") or p.get("playlist") or ""
        if not q:
            return None
        return "https://open.spotify.com/search/" + _q(q)

    if intent == "video":
        q = p.get("title") or ""
        if not q:
            return None
        return "https://www.netflix.com/search?q=" + _q(q)

    if intent == "tickets":
        q = p.get("event") or p.get("venue") or ""
        if not q:
            return None
        return "https://www.ticketmaster.com/search?q=" + _q(q)

    if intent == "reservation":
        q = p.get("venue") or ""
        if not q:
            return None
        return "https://www.opentable.com/s?term=" + _q(q)

    if intent == "task":
        title = p.get("title") or "Reminder"
        # x-apple-reminderkit doesn't support a public URL scheme — fall back to
        # creating a Google Tasks event-style link. Clients can override.
        return "https://tasks.google.com/embed/?origin=&full=1&q=" + _q(title)

    if intent == "note":
        content = p.get("content") or p.get("url") or ""
        if not content:
            return None
        # Apple Notes / Google Keep don't have a public deep link; we return
        # a generic mailto-style fallback the client can rewrite.
        return "mailto:?body=" + _q(content)

    if intent == "bills":
        kind = (p.get("kind") or "bills").lower()
        return "https://www.google.com/search?q=" + _q(f"pay {kind} online")

    if intent == "health":
        kind = (p.get("kind") or "pharmacy").lower()
        need = p.get("need") or kind
        return "https://www.google.com/maps/search/?api=1&query=" + _q(f"{kind} near me {need}")

    if intent == "weather":
        loc = p.get("location") or ""
        if loc:
            return "https://www.google.com/search?q=" + _q(f"weather in {loc}")
        return "https://www.google.com/search?q=weather"

    return None


# ═════════════════════════════════════════════════════════════════════
#  TRAINING TEMPLATES — synthetic data per new intent
# ═════════════════════════════════════════════════════════════════════
# Each list has 30+ phrasings. {tokens} get filled by generate_data.py:
#   {food}     {cuisine}     {drink}     {provider_food}
#   {place}    {address}     {city}      {city2}       {airport}
#   {song}     {artist}      {movie}     {show}        {ticket_event}
#   {item}     {bill_kind}   {qty}
#   {time}     {day}         {date}      {duration}    {date_phrase}

FOOD_ORDER = [
    "let's order {food} tonight",
    "i'm starving, let's get {food}",
    "should we order {cuisine} for dinner?",
    "ordering {food} on {provider_food}, want anything?",
    "doordash {food}?",
    "uber eats {cuisine}?",
    "order me a {food}",
    "let's get takeout — thinking {cuisine}",
    "im hungry, ordering {food}",
    "wanna get {food} delivered?",
    "ordering {food} now, last call",
    "lunch idea: {food}",
    "craving {food} bad rn",
    "let's split a {food} order",
    "order {qty} {food} please",
    "we should order in tonight, {cuisine}?",
    "ordering some {food}, you in?",
    "doordash {cuisine}?",
    "lets do {food} for dinner",
    "want to order {food}?",
    "thinking {food} from {provider_food}",
    "im gonna order {food}",
    "want to grab {food} on the way?",
    "lets get some {food}",
    "ordering pizza, you want any?",
    "should i order {food}?",
    "{food} for dinner sound good?",
    "let's just doordash",
    "i'll order us {food}",
    "wings tonight?",
    "feel like {cuisine} food",
    "getting {food}, want to split?",
    "order {food} for {qty} people",
    "im ordering {cuisine}, what do you want?",
]

RIDE = [
    "i'll uber over",
    "ubering to {place}",
    "lyft me to {place}",
    "share my ride to {place}",
    "calling an uber to {address}",
    "im taking a lyft",
    "uber to {airport}",
    "ride to {place} please",
    "let's split a uber",
    "uber on the way",
    "lyft eta 5 min",
    "calling uber from {place} to {place2}",
    "send me a ride",
    "book a cab to {place}",
    "i'll grab a lyft",
    "taking an uber, see u in 15",
    "uber from here to {place}",
    "lyft from {address} to {address2}",
    "getting a ride to {airport}",
    "uber pool to the office",
    "i'll just uber",
    "lyft to {place} now",
    "request a ride to {place}",
    "i need an uber",
    "taking a cab to {place}",
    "ola to {place}",   # India
    "rapido to {place}",   # India
    "let's uber together",
    "i'll uber you the location",
    "headed via uber",
]

TRAVEL = [
    "thinking of going to {city} next month",
    "want to plan a trip to {city}?",
    "flights to {city} are cheap rn",
    "should we book {city} for {date_phrase}?",
    "looking at hotels in {city}",
    "vacation idea: {city}",
    "let's plan {city} for {day}",
    "checking flights {city} to {city2}",
    "anyone been to {city}?",
    "booking {city} tickets tonight",
    "trip to {city} {date_phrase}",
    "round trip to {city} {date_phrase}",
    "any hotel recs for {city}?",
    "looking at airbnbs in {city}",
    "weekend trip to {city}?",
    "flights from {city} to {city2}",
    "we should do a trip to {city}",
    "long weekend in {city}",
    "tickets to {city} for {date_phrase}",
    "let's book the {city} trip",
    "hotel near {place} in {city}",
    "going to {city} for {duration}",
    "any of you free for a {city} trip {date_phrase}?",
    "skyscanner says flights to {city} are $300",
    "let's do {city} for new year",
    "trip planning for {city}",
    "thinking goa trip {date_phrase}",
    "vacation to {city}?",
]

SHOPPING = [
    "i need to order {item} from amazon",
    "buy me a {item}",
    "we're out of {item}, can you order?",
    "amazon {item}",
    "ordering a {item} on amazon",
    "need to get a new {item}",
    "shopping list: {item}",
    "let's order {item}",
    "buy {qty} {item}",
    "i'll grab {item} from amazon",
    "we need {item}",
    "running low on {item}",
    "order {item} please",
    "should we order more {item}?",
    "i need {item} for tomorrow",
    "amazon prime: {item}",
    "out of {item}, ordering",
    "buy a new {item}",
    "did you order the {item}?",
    "still need to order {item}",
    "let me order {item} now",
    "{item} on amazon is cheap",
    "i'll get {item} delivered",
    "got {item} on amazon",
    "amazon basket: {item}, {item2}",
    "order me {qty} {item}",
    "shopping cart: {item}",
    "i'm out of {item}",
    "buy more {item}",
]

MUSIC = [
    "send me that {song} song",
    "have you heard {song} by {artist}?",
    "play {song}",
    "this {artist} song is fire",
    "spotify {song}",
    "loving the new {artist} album",
    "playlist: {song}, {song2}",
    "what's that song that goes...",
    "{artist} just dropped a new track",
    "add {song} to the playlist",
    "shazam this song",
    "new {artist} just dropped",
    "play {artist} on spotify",
    "queue {song}",
    "this song slaps — {song}",
    "send the {song} link",
    "apple music {artist}",
    "playlist for the trip: {song}, {song2}",
    "any new music recs?",
    "song of the day: {song}",
    "vibing to {artist}",
    "{song} is stuck in my head",
    "did u listen to {song}?",
    "spotify wrapped season",
    "playing {artist} rn",
    "best {artist} song?",
    "love the new {song}",
    "drop the playlist link",
]

VIDEO = [
    "let's watch {movie} tonight",
    "you seen {show}?",
    "netflix and chill — {movie}?",
    "the new {movie} is on netflix",
    "watching {show} season 2",
    "movie night: {movie}",
    "let's stream {movie}",
    "{show} new season just dropped",
    "wanna watch {movie}?",
    "rewatching {show}",
    "have u seen {movie}?",
    "{movie} trailer is sick",
    "putting on {show}",
    "any good movies on netflix?",
    "let's binge {show}",
    "prime video: {movie}",
    "hbo max — {show}?",
    "movie tonight, thinking {movie}",
    "we should watch {movie} this weekend",
    "did u finish {show}?",
    "the {movie} sequel is out",
    "stream {movie} tonight",
    "netflix {show}",
    "rented {movie} on prime",
    "watching {movie} rn",
    "saw {movie} last night, banger",
    "next ep of {show} when",
]

TICKETS = [
    "should we get tickets to {ticket_event}?",
    "{ticket_event} tickets going fast",
    "ticketmaster {ticket_event}",
    "tickets for {ticket_event} {date_phrase}",
    "concert {date_phrase}",
    "knicks game {date_phrase}?",
    "got tickets to {ticket_event}",
    "let's go to {ticket_event}",
    "wanna go to the {ticket_event} concert?",
    "seatgeek for {ticket_event}",
    "{ticket_event} {date_phrase} — anyone in?",
    "tickets are $80 for {ticket_event}",
    "any tickets left for {ticket_event}?",
    "buying tickets to {ticket_event} now",
    "concert this {day}?",
    "{ticket_event} venue is {place}",
    "lets get the {ticket_event} tickets",
    "tickets to coldplay anyone?",
    "{ticket_event} sold out :(",
    "got 4 tickets to {ticket_event}",
    "{ticket_event} on {date_phrase}",
    "presale starts {date_phrase}",
    "ipl tickets",
    "match tickets {date_phrase}",
    "live show {date_phrase}",
    "comedy show {date_phrase}",
]

RESERVATION = [
    "let's get dinner at {place} {date_phrase}",
    "reservation for {qty} at {place}",
    "book a table at {place} for {date_phrase}",
    "opentable {place}",
    "lunch at {place} {date_phrase}?",
    "should i book {place} for {qty}?",
    "table for {qty} at {place}",
    "dinner reservation {date_phrase}",
    "salon appt {date_phrase}",
    "haircut {date_phrase}",
    "spa booking {date_phrase}",
    "brunch at {place} this {day}?",
    "let's reserve {place} for {date_phrase}",
    "got us a table at {place}",
    "book {place} {time} for {qty}",
    "any tables at {place} {date_phrase}?",
    "make a reservation at {place}",
    "lets do dinner reservations",
    "booking {place} for {qty} ppl",
    "resy at {place}",
    "dinner spot for friday — {place}?",
    "table for two at {place}",
    "reservation tonight at {place}",
    "should we book {place} or {place2}?",
    "yelp says {place} has a wait",
    "booked {place} for {date_phrase}",
    "appointment at the dentist {date_phrase}",
]

TASK = [
    "i need to do {task}",
    "todo: {task}",
    "remember to {task}",
    "add {task} to my todo list",
    "i have to {task} today",
    "task: {task}",
    "im supposed to {task}",
    "still need to {task}",
    "list: {task}, {task2}",
    "to do today: {task}",
    "add it to the list — {task}",
    "make sure to {task}",
    "things to do: {task}",
    "im behind on {task}",
    "{task} by tomorrow",
    "have to finish {task}",
    "follow up on {task}",
    "checklist: {task}",
    "i should {task}",
    "this week: {task}",
    "homework: {task}",
    "personal todo {task}",
    "{task} before {day}",
    "remind me later — {task}",
    "lemme add {task} to notion",
    "i still need to {task}",
]

NOTE = [
    "save this link: {url}",
    "save this for later",
    "note to self: {task}",
    "noting this down",
    "save it in notes",
    "important: {task}",
    "saving this article",
    "remember this — {task}",
    "{url} — read later",
    "save the recipe",
    "writing this down",
    "let me save that",
    "snippet to save: {task}",
    "putting in notes app",
    "this quote is gold",
    "save this idea",
    "interesting article: {url}",
    "bookmarking this",
    "want to save this",
    "for the records: {task}",
    "saving for inspiration",
    "this is worth saving",
    "good {task} idea",
    "let me jot that down",
    "putting this in notes",
]

BILLS = [
    "rent's due {date_phrase}",
    "pay {bill_kind} bill",
    "{bill_kind} payment {date_phrase}",
    "due tomorrow: {bill_kind}",
    "still need to pay {bill_kind}",
    "auto-pay didn't go through",
    "dont forget {bill_kind} bill",
    "subscription renewal {date_phrase}",
    "credit card bill due",
    "wifi bill came",
    "electric bill is high this month",
    "phone bill {date_phrase}",
    "rent transfer {date_phrase}",
    "{bill_kind} due ${qty}",
    "ill pay the {bill_kind} bill",
    "paying rent online",
    "ola/zomato sub",
    "netflix renewal",
    "amex due {date_phrase}",
    "did u pay the {bill_kind}?",
    "split the {bill_kind} bill?",
    "venmo me for {bill_kind}",
    "rent reminder",
    "internet bill due",
    "comcast bill",
]

HEALTH = [
    "pharmacy near me",
    "doctor appointment {date_phrase}",
    "need to refill my meds",
    "running out of {item} (meds)",
    "have a headache",
    "doc appt {date_phrase}",
    "pharmacy run",
    "feeling sick today",
    "fever since morning",
    "should book a doctor",
    "where's the nearest pharmacy",
    "dentist {date_phrase}",
    "ophthalmologist appt",
    "physio {date_phrase}",
    "got a prescription, need pharmacy",
    "telehealth appt {date_phrase}",
    "1mg order",
    "pharmacy at {place}",
    "covid test {date_phrase}",
    "vaccine {date_phrase}",
    "doctor's note needed",
    "annual checkup {date_phrase}",
    "feeling sick, where's the urgent care",
    "i need ibuprofen",
    "got a sore throat",
]

WEATHER = [
    "is it raining",
    "weather today",
    "whats the weather in {city}",
    "is it going to rain {day}",
    "weather tomorrow",
    "forecast for {city}",
    "should i bring an umbrella",
    "is it cold outside",
    "temp in {city}?",
    "snowing in {city}?",
    "weather this weekend",
    "rain expected {date_phrase}",
    "how's the weather there",
    "forecast {date_phrase}",
    "is it sunny",
    "humid today?",
    "weather report",
    "high today in {city}?",
    "windy outside?",
    "what's it like outside",
    "weather alert {city}",
    "raining cats and dogs",
    "is the storm coming",
]


TEMPLATES_V3: Dict[str, List[str]] = {
    "food_order":  FOOD_ORDER,
    "ride":        RIDE,
    "travel":      TRAVEL,
    "shopping":    SHOPPING,
    "music":       MUSIC,
    "video":       VIDEO,
    "tickets":     TICKETS,
    "reservation": RESERVATION,
    "task":        TASK,
    "note":        NOTE,
    "bills":       BILLS,
    "health":      HEALTH,
    "weather":     WEATHER,
}


# ═════════════════════════════════════════════════════════════════════
#  EXTRA TOKEN FILLERS for the new intents
# ═════════════════════════════════════════════════════════════════════

FOODS = [
    "pizza", "sushi", "burgers", "pad thai", "biryani", "ramen", "tacos",
    "chinese food", "wings", "burritos", "salads", "indian food", "sandwiches",
    "noodles", "fried chicken", "chipotle", "shawarma", "bbq", "pho",
    "mediterranean", "korean bbq", "fried rice", "dim sum", "kebabs",
    "fish and chips", "smoothies", "cookies", "milkshake", "donuts",
]
CUISINES = [
    "italian", "indian", "thai", "chinese", "japanese", "mexican", "korean",
    "vietnamese", "mediterranean", "greek", "american", "french",
    "ethiopian", "lebanese", "spanish", "turkish",
]
DRINKS = ["coffee", "boba", "smoothie", "juice", "iced tea", "matcha", "cold brew"]
FOOD_PROVIDERS = ["doordash", "uber eats", "grubhub", "swiggy", "zomato", "postmates"]

CITIES = [
    "Goa", "Tokyo", "Paris", "Lisbon", "Bali", "New York", "London", "Bangkok",
    "Mumbai", "Bangalore", "San Francisco", "Seattle", "Austin", "Miami",
    "Dubai", "Singapore", "Sydney", "Berlin", "Amsterdam", "Barcelona",
    "Rome", "Istanbul", "Cairo", "Cape Town", "Mexico City",
]
AIRPORTS = ["JFK", "LAX", "SFO", "SEA", "BLR", "BOM", "DEL", "LHR", "CDG", "DXB"]

SONGS = [
    "Espresso", "Flowers", "Anti-Hero", "Cruel Summer", "As It Was", "Chaiyya Chaiyya",
    "Levitating", "Heat Waves", "Blinding Lights", "Bad Guy", "Watermelon Sugar",
    "Save Your Tears", "Industry Baby", "Stay", "Peaches",
]
ARTISTS = [
    "Taylor Swift", "Drake", "Bad Bunny", "Sabrina Carpenter", "Travis Scott",
    "Arijit Singh", "AP Dhillon", "Karan Aujla", "The Weeknd", "Billie Eilish",
    "Olivia Rodrigo", "Doja Cat", "SZA", "Post Malone", "Kendrick Lamar",
]
MOVIES = [
    "Dune Part Two", "Oppenheimer", "Inception", "The Batman", "Barbie",
    "Spider-Man", "Top Gun Maverick", "Avengers Endgame", "Pulp Fiction",
    "Jawan", "Pathaan", "Animal", "Salaar", "Fighter", "12th Fail",
]
SHOWS = [
    "Squid Game", "Stranger Things", "Succession", "House of the Dragon",
    "The Bear", "Friends", "Office", "Breaking Bad", "Mirzapur", "Panchayat",
    "Scam 1992", "Family Man", "The Boys", "Wednesday",
]
EVENTS = [
    "Coldplay", "Taylor Swift Eras", "IPL final", "Lakers game", "Knicks game",
    "Comic-Con", "F1 Grand Prix", "Burning Man", "NBA finals", "Super Bowl",
    "World Cup", "Coachella", "Tomorrowland", "Sunburn", "AR Rahman live",
]
SHOPPING_ITEMS = [
    "paper towels", "toilet paper", "shampoo", "phone charger", "headphones",
    "yoga mat", "running shoes", "kitchen sponges", "trash bags", "laundry detergent",
    "protein powder", "cat food", "dog treats", "lightbulbs", "batteries",
    "monitor stand", "keyboard", "mouse", "vitamin d", "multivitamin",
]
BILL_KINDS = [
    "rent", "electricity", "internet", "phone", "credit card", "wifi",
    "netflix", "spotify", "gym", "insurance", "water", "gas", "cable",
]
TASKS = [
    "finish the report", "email Sarah back", "submit timesheet", "clean the apartment",
    "renew passport", "call the bank", "fix the leaking tap", "mail the package",
    "book the flight", "review PR", "study for exam", "prep slides",
    "follow up with client", "do laundry", "grocery run",
]
URLS = [
    "https://nytimes.com/article", "https://medium.com/post", "https://x.com/post",
    "https://substack.com/p", "https://github.com/repo", "https://reddit.com/r/x",
]


# ═════════════════════════════════════════════════════════════════════
#  Mixed multi-intent templates that span v3 (calendar + reservation, etc.)
# ═════════════════════════════════════════════════════════════════════

MIXED_V3: List[Tuple[str, Dict[str, int]]] = [
    # calendar + reservation
    ("dinner at {place} {date_phrase}",
        {"calendar": 1, "reservation": 1}),
    ("book {place} for dinner {date_phrase}",
        {"calendar": 1, "reservation": 1}),
    # calendar + tickets
    ("{ticket_event} concert {date_phrase}, got tickets",
        {"calendar": 1, "tickets": 1}),
    # maps + ride
    ("uber to {place} now",
        {"maps": 1, "ride": 1}),
    # food + money
    ("ordered {food}, venmo me {amount}",
        {"food_order": 1, "money": 1}),
    # travel + calendar
    ("flight to {city} {date_phrase}",
        {"travel": 1, "calendar": 1}),
    # shopping + money
    ("ordered the {item}, you owe me {amount}",
        {"shopping": 1, "money": 1}),
    # music + note
    ("save this song {song}",
        {"music": 1, "note": 1}),
    # health + maps
    ("pharmacy near {place}",
        {"health": 1, "maps": 1}),
    # bills + money
    ("rent due {date_phrase}, transfer me {amount}",
        {"bills": 1, "money": 1}),
    # task + alarm
    ("remind me to {task} at {time}",
        {"task": 1, "alarm": 1}),
    # video + calendar
    ("movie night {date_phrase}, watching {movie}",
        {"video": 1, "calendar": 1}),
]


# ═════════════════════════════════════════════════════════════════════
#  HARD NEGATIVES for v3 intents -- messages that mention an intent's
#  keyword but should NOT fire that intent.
#
#  This is the single biggest quality lever for a real-world superapp:
#  without these, the model learns "any sentence with 'pizza' = food_order",
#  "any sentence with 'paris' = travel". With them, it learns the
#  *actionable* phrasing, not the keyword.
#
#  Each list is consumed by generate_data.py and emitted with all-zero
#  labels (the model learns "this sentence fires NO intent").
# ═════════════════════════════════════════════════════════════════════

NEG_FOOD_TRICKY = [
    "pizza was great last night",
    "I don't even like {cuisine} food",
    "not ordering anything tonight",
    "we just ate, im good",
    "I'm allergic to {food}",
    "remember when we had {food} in vegas",
    "i used to love {food} as a kid",
    "doordash stock dropped 5%",
    "uber eats is overpriced honestly",
    "the {food} place closed down",
    "had {food} for lunch already",
    "burger king was so bad",
    "swiggy is down rn lol",
    "no more {food} for me this month",
    "{cuisine} food gives me heartburn",
    "stopped eating {food} last year",
    "dad makes the best {food}",
    "wish i could eat {food} but on a diet",
]

NEG_RIDE_TRICKY = [
    "uber driver was so rude",
    "saw an uber outside",
    "lyft stock is up",
    "ola cabs was in the news",
    "the uber commercial was funny",
    "rapido bike got hit by a car",
    "uber eats not uber the cab",
    "i used to drive for lyft",
    "uber as a verb is weird",
    "cab drivers in nyc tho",
    "remember when ubers were cheap",
    "watched a doc about uber's founding",
    "lyft just went public",
    "auto rickshaws are wild",
]

NEG_TRAVEL_TRICKY = [
    "i love {city}",
    "{city} is beautiful",
    "miss {city} so much",
    "never been to {city} but want to",  # aspirational, not actionable
    "{city} pizza is the best",
    "{city} weather is crazy",
    "born in {city}",
    "grew up in {city}",
    "lived in {city} for 5 years",
    "my friend moved to {city}",
    "{city} traffic is hell",
    "skyscanner ad on instagram lol",
    "airbnb stock is down",
    "expedia just had a sale",
    "the {city} olympics was amazing",
    "saw a {city} doc on netflix",
    "{city} accent is so cute",
    "never going back to {city}",
    "wanna retire in {city} someday",
]

NEG_SHOPPING_TRICKY = [
    "amazon stock is up 3%",
    "amazon prime show is good",
    "amazon warehouse strike news",
    "i work at amazon",
    "amazon basin documentary",
    "remember when amazon was just books",
    "the amazon rainforest",
    "got amazon shipping issues last week",
    "amazon prime day was insane",
    "shopping cart abandoned haha",
    "i hate shopping",
    "shopping mall on sunday is hell",
    "thinking about ordering... nah",
    "{item} from amazon was defective",
    "returned the {item} already",
    "no more {item} for me",
]

NEG_MUSIC_TRICKY = [
    "{artist} just announced a tour",  # tickets, not music play
    "{artist} divorce news",
    "spotify wrapped season again",
    "spotify is buggy today",
    "apple music is overpriced",
    "shazam'd a song earlier",
    "{song} is overrated tbh",
    "my speaker is broken",
    "concert was so loud",
    "i hate {artist}",
    "stopped listening to {artist}",
    "the {song} music video is wild",
    "{artist} won a grammy",
    "music industry is changing fast",
    "playlist got deleted somehow",
]

NEG_VIDEO_TRICKY = [
    "watched {show} last week",
    "finished {movie} yesterday",
    "{show} ending sucked",
    "netflix stock is tanking",
    "netflix raised prices again",
    "hbo got rebranded to max",
    "the {movie} cast is stacked",
    "remember when {show} came out",
    "rewatched {show} years ago",
    "saw {movie} in theaters",
    "i don't have netflix",
    "cancelled my prime sub",
    "{show} characters are annoying",
    "no spoilers please",
    "spoilers for {show}: it ends",
    "{movie} review is brutal",
    "the {show} intro slaps",
    "couldn't get into {show}",
]

NEG_TICKETS_TRICKY = [
    "{ticket_event} was last night",
    "missed {ticket_event} :(",
    "{ticket_event} got cancelled",
    "tickets sold out, oh well",
    "{ticket_event} was insane",
    "remember the {ticket_event} from 2019",
    "concert was packed",
    "the {ticket_event} after party was wild",
    "ticketmaster lawsuit news",
    "stubhub fees are insane",
    "saw {ticket_event} on tv",
    "{ticket_event} is overrated honestly",
    "couldn't go to {ticket_event} last year",
]

NEG_RESERVATION_TRICKY = [
    "{place} was packed last night",
    "went to {place} yesterday",
    "{place} closed down",
    "no reservations available, oh well",
    "opentable is glitching",
    "wait at {place} was 2 hours",
    "{place} food was meh",
    "no longer going to {place}",
    "they don't take reservations",
    "{place} got bad yelp reviews",
    "remember dinner at {place}",
    "{place} chef left",
]

NEG_TASK_TRICKY = [
    "i should really do laundry",  # rhetorical
    "i really should call mom",  # rhetorical
    "wish i had time to do my taxes",
    "tasks are piling up lol",
    "todo lists don't work for me",
    "notion is bloated",
    "anyone use linear?",
    "add a feature to your app",  # not a personal todo
    "homework was easy back then",
    "remember when we had assignments",
    "no more meetings please",
    "checklist culture is too much",
    "got nothing to do today",
]

NEG_NOTE_TRICKY = [
    "i don't take notes anymore",
    "lost all my notes when phone broke",
    "notes app is so cluttered",
    "evernote is dying",
    "obsidian users are intense",
    "i write everything down already",
    "saved that link weeks ago",  # past
    "cleared my notes folder",
    "no need to save this",
    "didn't bother bookmarking",
    "memory is better than notes anyway",
    "wrote a whole essay about it",
]

NEG_BILLS_TRICKY = [
    "rent prices are insane",  # commentary
    "{bill_kind} bills are ridiculous lately",
    "everyone's electric bill is up",
    "wifi keeps cutting out",
    "phone died lol",
    "got a billing complaint, sorted",
    "no more subscriptions for me",
    "cancelled my {bill_kind}",
    "rent went up again",
    "the {bill_kind} company is the worst",
    "remember when wifi was free",
    "split rent with my ex",
    "hate auto-renew",
    "credit card is maxed",
]

NEG_HEALTH_TRICKY = [
    "had a checkup last week",  # past
    "doctor said i'm fine",
    "pharmacy was closed yesterday",
    "felt sick all weekend",
    "got over the flu finally",
    "1mg ad on whatsapp",
    "watched grey's anatomy lol",
    "house md was such a good show",
    "telehealth is overrated",
    "the cvs ran out",
    "doctor's office was closed",
    "no headache today thankfully",
    "feeling fine actually",
    "scared of dentists tho",
    "the er bill was insane",
]

NEG_WEATHER_TRICKY = [
    "weather was nice yesterday",  # past commentary
    "i love the weather here",
    "weather app is broken",
    "raining cats and dogs is a weird phrase",  # idiom
    "the weather channel is wild",
    "global warming is real",
    "weather was the worst last month",
    "winter is over thank god",
    "miss the {city} weather",
    "weather report is wrong always",
    "no more snow days for me",
    "love thunderstorms tho",
    "growing up in {city} weather was wild",
    "weather small talk is the worst",
]


# Single canonical list -- generate_data.py iterates over this.
NEG_V3_GROUPS: Dict[str, List[str]] = {
    "neg_food_tricky":        NEG_FOOD_TRICKY,
    "neg_ride_tricky":        NEG_RIDE_TRICKY,
    "neg_travel_tricky":      NEG_TRAVEL_TRICKY,
    "neg_shopping_tricky":    NEG_SHOPPING_TRICKY,
    "neg_music_tricky":       NEG_MUSIC_TRICKY,
    "neg_video_tricky":       NEG_VIDEO_TRICKY,
    "neg_tickets_tricky":     NEG_TICKETS_TRICKY,
    "neg_reservation_tricky": NEG_RESERVATION_TRICKY,
    "neg_task_tricky":        NEG_TASK_TRICKY,
    "neg_note_tricky":        NEG_NOTE_TRICKY,
    "neg_bills_tricky":       NEG_BILLS_TRICKY,
    "neg_health_tricky":      NEG_HEALTH_TRICKY,
    "neg_weather_tricky":     NEG_WEATHER_TRICKY,
}
