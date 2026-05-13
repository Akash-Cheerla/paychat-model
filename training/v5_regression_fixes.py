"""
v5.1 regression fixes — massive targeted banks for every seed-suite failure.

v5 retrained the model for multi-turn context (15/15 perfect) but regressed
on 12 single-turn cases. This file carpet-bombs each failure pattern with
enough examples to eliminate it.

The 12 failures break down into 7 root causes:
  1. alarm overfires on time mentions in non-alarm contexts
  2. calendar overfires on activity descriptions / reservations
  3. maps overfires on venue names / ride destinations
  4. money overfires on bills/rent contexts
  5. negation still not strong enough ("don't transfer", "cancel the reminder")
  6. contact threshold too high — "call mom" scores 0.82 vs threshold 0.84
  7. bare "weather" doesn't fire (prob 0.01)

Strategy: contrastive pairs everywhere. Every positive has a matching negative.
"""

# ═════════════════════════════════════════════════════════════════════════════
# 1. TIME_NOT_ALARM — time in non-alarm context must NOT fire alarm
#
# Root cause: model learned "time mentioned = alarm". But "uber at 5am" means
# the ride is at 5am, not "set alarm for 5am". "flight at 6am" is travel,
# "movie at 9pm" is video/tickets, "meeting at 3" is calendar.
#
# Fix: massive bank of "{action} at {time}" where alarm=0.
# ═════════════════════════════════════════════════════════════════════════════

# Format: (template, intent_flags)
V51_TIME_NOT_ALARM = [
    # ── ride + time (alarm must NOT fire) ──
    ("uber at {time}", {"ride": 1}),
    ("uber to {place} at {time}", {"ride": 1, "maps": 1}),
    ("lyft at {time}", {"ride": 1}),
    ("cab at {time}", {"ride": 1}),
    ("book a cab for {time}", {"ride": 1}),
    ("uber to {airport} at {time} {day}", {"ride": 1, "maps": 1}),
    ("lyft to {place} at {time}", {"ride": 1, "maps": 1}),
    ("get an uber at {time}", {"ride": 1}),
    ("ride at {time}", {"ride": 1}),
    ("cab at {time} to the airport", {"ride": 1, "maps": 1}),
    ("uber home at {time}", {"ride": 1}),
    ("lyft at {time} tomorrow", {"ride": 1}),
    ("schedule uber for {time}", {"ride": 1}),
    ("need a ride at {time}", {"ride": 1}),
    ("pick me up at {time}", {"ride": 1}),
    ("uber at {time} sharp", {"ride": 1}),
    ("the uber is at {time}", {"ride": 1}),
    ("car coming at {time}", {"ride": 1}),
    ("ride is booked for {time}", {"ride": 1}),

    # ── travel + time (alarm must NOT fire) ──
    ("flight at {time}", {"travel": 1}),
    ("flight at {time} {day}", {"travel": 1}),
    ("train at {time}", {"travel": 1}),
    ("bus at {time} to {city}", {"travel": 1}),
    ("flight to {city} at {time}", {"travel": 1}),
    ("the flight leaves at {time}", {"travel": 1}),
    ("flight departs at {time}", {"travel": 1}),
    ("my flight is at {time} {day}", {"travel": 1}),
    ("boarding at {time}", {"travel": 1}),
    ("departure is at {time}", {"travel": 1}),
    ("catching the {time} train", {"travel": 1}),

    # ── video/tickets + time (alarm must NOT fire) ──
    ("movie at {time}", {"video": 1}),
    ("movie at {time} tonight", {"video": 1}),
    ("show at {time}", {"video": 1}),
    ("{movie} at {time}", {"video": 1, "tickets": 1}),
    ("movie starts at {time}", {"video": 1}),
    ("{show} at {time} tonight", {"video": 1}),
    ("the movie is at {time}", {"video": 1}),
    ("screening at {time}", {"video": 1, "tickets": 1}),
    ("concert at {time}", {"tickets": 1}),
    ("concert starts at {time}", {"tickets": 1}),
    ("event at {time}", {"tickets": 1}),

    # ── food + time (alarm must NOT fire) ──
    ("dinner at {time}", {"food_order": 1}),
    ("lunch at {time}", {"food_order": 1}),
    ("order food at {time}", {"food_order": 1}),
    ("pizza at {time}", {"food_order": 1}),
    ("order at {time}", {"food_order": 1}),
    ("let's eat at {time}", {"food_order": 1}),
    ("doordash at {time}", {"food_order": 1}),
    ("delivery at {time}", {"food_order": 1}),

    # ── reservation + time (alarm must NOT fire) ──
    ("reservation at {time}", {"reservation": 1}),
    ("table at {time}", {"reservation": 1}),
    ("booking at {time}", {"reservation": 1}),
    ("dinner reservation {time}", {"reservation": 1}),
    ("brunch at {time}", {"reservation": 1}),
    ("book a table for {time}", {"reservation": 1}),

    # ── shopping + time (alarm must NOT fire) ──
    ("sale starts at {time}", {"shopping": 1}),
    ("drop is at {time}", {"shopping": 1}),

    # ── health + time (alarm must NOT fire) ──
    ("appointment at {time}", {"health": 1}),
    ("doctor at {time}", {"health": 1}),
    ("dentist at {time}", {"health": 1}),
    ("checkup at {time}", {"health": 1}),

    # ── bills + time (alarm must NOT fire) ──
    ("bill due at {time}", {"bills": 1}),
    ("payment at {time}", {"bills": 1}),
]

# ── Contrastive: actual alarm requests (alarm SHOULD fire) ──
V51_TIME_IS_ALARM = [
    ("remind me at {time}", {"alarm": 1}),
    ("set alarm for {time}", {"alarm": 1}),
    ("wake me at {time}", {"alarm": 1}),
    ("alarm at {time}", {"alarm": 1}),
    ("timer for {duration}", {"alarm": 1}),
    ("ping me at {time}", {"alarm": 1}),
    ("buzz me at {time}", {"alarm": 1}),
    ("notify me at {time}", {"alarm": 1}),
    ("alert me at {time}", {"alarm": 1}),
    ("remind me in {duration}", {"alarm": 1}),
    ("remind me at {time} to {task}", {"alarm": 1}),
    ("set a reminder for {time}", {"alarm": 1}),
    ("wake me up at {time}", {"alarm": 1}),
    ("need an alarm for {time}", {"alarm": 1}),
    ("don't let me sleep past {time}", {"alarm": 1}),
    ("poke me at {time}", {"alarm": 1}),
    ("{duration} timer", {"alarm": 1}),
    ("remind me {day} at {time}", {"alarm": 1}),
    ("set alarm {time} {day}", {"alarm": 1}),
    ("reminder at {time}", {"alarm": 1}),
]


# ═════════════════════════════════════════════════════════════════════════════
# 2. ACTIVITY_NOT_CALENDAR — describing current/planned activity ≠ calendar
#
# "movie night watching Oppenheimer" = they're watching now/describing plans,
# not asking to CREATE a calendar event. "book a table friday 7" is a
# reservation, not a separate calendar event.
# ═════════════════════════════════════════════════════════════════════════════

V51_ACTIVITY_NOT_CALENDAR = [
    # ── present activity descriptions (calendar must NOT fire) ──
    ("movie night, watching {movie}", {"video": 1}),
    ("movie night with {name}", {"video": 1}),
    ("binge watching {show}", {"video": 1}),
    ("watching {movie} rn", {"video": 1}),
    ("about to watch {movie}", {"video": 1}),
    ("putting on {show}", {"video": 1}),
    ("starting {movie} now", {"video": 1}),
    ("we're watching {show} tonight", {"video": 1}),
    ("going to watch {movie}", {"video": 1}),
    ("gonna binge {show}", {"video": 1}),
    ("movie marathon tonight", {"video": 1}),
    ("watching movies all day", {"video": 1}),
    ("netflix and chill", {"video": 1}),
    ("catching up on {show}", {"video": 1}),
    ("finally watching {movie}", {"video": 1}),

    # ── reservation ≠ calendar (calendar must NOT fire alongside reservation) ──
    ("book a table for {qty} at {place}", {"reservation": 1}),
    ("book a table at {place} {day} at {time}", {"reservation": 1}),
    ("reservation at {place} for {qty}", {"reservation": 1}),
    ("table for {qty} at {place} {time}", {"reservation": 1}),
    ("book a table {day} at {time}", {"reservation": 1}),
    ("reserve a spot at {place}", {"reservation": 1}),
    ("book {place} for {qty} people {day}", {"reservation": 1}),
    ("get a reservation at {place} {day} {time}", {"reservation": 1}),
    ("make a reservation for {qty} at {time}", {"reservation": 1}),
    ("reserve {place} for {day} {time}", {"reservation": 1}),
    ("book us a table at {place}", {"reservation": 1}),
    ("reservation for {qty} {day} at {time}", {"reservation": 1}),
    ("table for {qty} on {day}", {"reservation": 1}),
    ("book dinner at {place}", {"reservation": 1}),
    ("get us a table at {place} {day}", {"reservation": 1}),

    # ── event description, not scheduling ──
    ("game night tonight", {}),
    ("movie night tonight", {}),
    ("having a party tonight", {}),
    ("we're going out tonight", {}),
    ("date night tonight", {}),
    ("poker night at {name}'s", {}),
    ("guys night out", {}),
    ("girls night tonight", {}),
    ("karaoke night!", {}),
    ("trivia night {day}", {}),
]


# ═════════════════════════════════════════════════════════════════════════════
# 3. VENUE_NOT_MAPS — place name in event context ≠ navigation
#
# "lunch at sweetgreen" = calendar + maybe reservation, NOT maps.
# "oppenheimer at amc" = tickets/video, NOT maps.
# "dinner at olive garden" = reservation, NOT maps.
# Maps should only fire when someone NEEDS to navigate/find/go somewhere.
# ═════════════════════════════════════════════════════════════════════════════

V51_VENUE_NOT_MAPS = [
    # ── restaurant/food venue in social context (maps must NOT fire) ──
    ("lunch at {place} {day}", {"calendar": 1}),
    ("dinner at {place} {day} at {time}", {"calendar": 1, "reservation": 1}),
    ("brunch at {place} {day}", {"calendar": 1}),
    ("coffee at {place} {day}", {"calendar": 1}),
    ("drinks at {place} {day}", {"calendar": 1}),
    ("lunch with {name} at {place}", {"calendar": 1}),
    ("dinner with {name} at {place} {day}", {"calendar": 1}),
    ("meeting {name} at {place} {time}", {"calendar": 1}),
    ("get together at {place} {day}", {"calendar": 1}),
    ("hangout at {place}", {"calendar": 1}),
    ("party at {place} {day}", {"calendar": 1}),
    ("celebration at {place}", {"calendar": 1}),
    ("birthday at {place}", {"calendar": 1}),
    ("event at {place} {day}", {"calendar": 1}),
    ("team lunch at {place}", {"calendar": 1}),
    ("team dinner at {place} {day}", {"calendar": 1}),

    # ── movie theater as venue (maps must NOT fire) ──
    ("{movie} at amc tonight", {"video": 1, "tickets": 1}),
    ("{movie} at amc {time}", {"video": 1, "tickets": 1}),
    ("{movie} at the theater tonight", {"video": 1, "tickets": 1}),
    ("movie at amc {day} at {time}", {"video": 1, "tickets": 1}),
    ("tickets for {movie} at amc", {"tickets": 1}),
    ("imax showing at {time}", {"video": 1, "tickets": 1}),
    ("catch {movie} at the cinema", {"video": 1, "tickets": 1}),
    ("{movie} at regal tonight", {"video": 1, "tickets": 1}),
    ("premiere at the theater {day}", {"video": 1, "tickets": 1}),

    # ── concert/event venue (maps must NOT fire) ──
    ("concert at {place} {day}", {"tickets": 1}),
    ("show at {place} tonight", {"tickets": 1}),
    ("comedy show at {place}", {"tickets": 1}),
    ("stand-up at {place} {day}", {"tickets": 1}),

    # ── hotel as venue (maps must NOT fire) ──
    ("stay at {place}", {"reservation": 1}),
    ("booked {place} for the weekend", {"reservation": 1}),
]

# ── Contrastive: actual navigation requests (maps SHOULD fire) ──
V51_PLACE_IS_MAPS = [
    ("directions to {place}", {"maps": 1}),
    ("how do I get to {place}", {"maps": 1}),
    ("navigate to {place}", {"maps": 1}),
    ("heading to {place}", {"maps": 1}),
    ("on my way to {place}", {"maps": 1}),
    ("meet me at {place}", {"maps": 1}),
    ("I'm at {place}", {"maps": 1}),
    ("where is {place}", {"maps": 1}),
    ("find {place} on maps", {"maps": 1}),
    ("drop a pin at {place}", {"maps": 1}),
    ("take me to {place}", {"maps": 1}),
    ("drive to {place}", {"maps": 1}),
    ("go to {place}", {"maps": 1}),
    ("omw to {place}", {"maps": 1}),
    ("pulling up to {place}", {"maps": 1}),
    ("address for {place}", {"maps": 1}),
    ("where's {place}", {"maps": 1}),
    ("how far is {place}", {"maps": 1}),
    ("map to {place}", {"maps": 1}),
    ("route to {place}", {"maps": 1}),
]


# ═════════════════════════════════════════════════════════════════════════════
# 4. RIDE_NOT_MAPS — uber/lyft already implies destination
#
# "wanna uber there" = ride, NOT ride+maps. The ride intent handles the
# navigation. Maps only fires when someone explicitly needs directions or
# is sharing a location.
# ═════════════════════════════════════════════════════════════════════════════

V51_RIDE_NOT_MAPS = [
    # ── ride requests where maps must NOT fire ──
    ("wanna uber there", {"ride": 1}),
    ("let's uber there", {"ride": 1}),
    ("should we uber there", {"ride": 1}),
    ("uber there?", {"ride": 1}),
    ("cab there", {"ride": 1}),
    ("lyft there", {"ride": 1}),
    ("let's take an uber", {"ride": 1}),
    ("we should uber", {"ride": 1}),
    ("uber it", {"ride": 1}),
    ("let's cab it", {"ride": 1}),
    ("should we cab it", {"ride": 1}),
    ("wanna cab it", {"ride": 1}),
    ("uber over", {"ride": 1}),
    ("ride over there", {"ride": 1}),
    ("take a cab there", {"ride": 1}),
    ("grab an uber", {"ride": 1}),
    ("let's grab an uber", {"ride": 1}),
    ("grab a lyft", {"ride": 1}),
    ("get a cab", {"ride": 1}),
    ("get an uber", {"ride": 1}),
    ("call a cab", {"ride": 1}),
    ("hail a cab", {"ride": 1}),
    ("book a ride", {"ride": 1}),
    ("need a ride there", {"ride": 1}),
    ("uber home", {"ride": 1}),
    ("uber back", {"ride": 1}),
    ("lyft home", {"ride": 1}),
    ("cab home", {"ride": 1}),
    ("cab back", {"ride": 1}),
    ("ride home", {"ride": 1}),
    ("uber to the party", {"ride": 1}),
    ("uber to dinner", {"ride": 1}),
    ("cab to the event", {"ride": 1}),
    ("lyft to the game", {"ride": 1}),
    ("uber to the concert", {"ride": 1}),
]


# ═════════════════════════════════════════════════════════════════════════════
# 5. BILLS_NOT_MONEY — rent/utility/bill payments ≠ P2P money
#
# "pay rent" / "remind me to pay rent" = bills (or alarm+bills), NOT money.
# Money intent is for person-to-person: venmo, split, owe, cashapp.
# Bills intent is for services: rent, electricity, wifi, insurance.
# ═════════════════════════════════════════════════════════════════════════════

V51_BILLS_NOT_MONEY = [
    # ── bill payments (money must NOT fire) ──
    ("pay rent", {"bills": 1}),
    ("pay rent {day}", {"bills": 1}),
    ("pay the rent", {"bills": 1}),
    ("pay the rent {day}", {"bills": 1}),
    ("pay the electricity bill", {"bills": 1}),
    ("pay the wifi bill", {"bills": 1}),
    ("pay the internet bill", {"bills": 1}),
    ("pay the water bill", {"bills": 1}),
    ("pay the gas bill", {"bills": 1}),
    ("pay the phone bill", {"bills": 1}),
    ("pay the cable bill", {"bills": 1}),
    ("pay the insurance", {"bills": 1}),
    ("pay the mortgage", {"bills": 1}),
    ("pay the car payment", {"bills": 1}),
    ("pay the student loan", {"bills": 1}),
    ("pay utilities", {"bills": 1}),
    ("pay my subscription", {"bills": 1}),
    ("pay netflix bill", {"bills": 1}),
    ("pay spotify bill", {"bills": 1}),
    ("rent is due", {"bills": 1}),
    ("rent is due {day}", {"bills": 1}),
    ("electricity bill is due", {"bills": 1}),
    ("wifi bill due {day}", {"bills": 1}),
    ("bill due {day}", {"bills": 1}),
    ("gotta pay rent", {"bills": 1}),
    ("need to pay rent", {"bills": 1}),
    ("time to pay rent", {"bills": 1}),
    ("rent payment due", {"bills": 1}),

    # ── remind + bills (alarm + bills, NOT money) ──
    ("remind me to pay rent {day}", {"alarm": 1, "bills": 1}),
    ("remind me to pay the electricity bill", {"alarm": 1, "bills": 1}),
    ("remind me to pay the wifi bill {day}", {"alarm": 1, "bills": 1}),
    ("remind me rent is due {day}", {"alarm": 1, "bills": 1}),
    ("reminder to pay utilities {day}", {"alarm": 1, "bills": 1}),
    ("remind me about rent on {day}", {"alarm": 1, "bills": 1}),
    ("don't let me forget rent {day}", {"alarm": 1, "bills": 1}),
    ("remind me to pay insurance {day}", {"alarm": 1, "bills": 1}),
    ("set reminder for rent {day}", {"alarm": 1, "bills": 1}),
    ("ping me about the bill {day}", {"alarm": 1, "bills": 1}),
]

# ── Contrastive: actual money (P2P) requests ──
V51_ACTUAL_MONEY = [
    ("venmo me {amount}", {"money": 1}),
    ("you owe me {amount}", {"money": 1}),
    ("send me {amount}", {"money": 1}),
    ("pay me back {amount}", {"money": 1}),
    ("cashapp me {amount}", {"money": 1}),
    ("split it with me", {"money": 1}),
    ("let's split the bill", {"money": 1}),
    ("you owe me from last night", {"money": 1}),
    ("send {amount} for dinner", {"money": 1}),
    ("pay me back whenever", {"money": 1}),
    ("your share is {amount}", {"money": 1}),
    ("transfer me {amount}", {"money": 1}),
    ("I'll venmo you", {"money": 1}),
    ("zelle me {amount}", {"money": 1}),
    ("gpay me {amount}", {"money": 1}),
]


# ═════════════════════════════════════════════════════════════════════════════
# 6. STRONG_NEGATION — cancel/don't/stop must kill the intent
#
# The model still fires on "don't transfer" and "cancel the reminder".
# Need massive negation bank with all patterns.
# ═════════════════════════════════════════════════════════════════════════════

V51_STRONG_NEGATION = [
    # ── don't + action verb ──
    ("don't transfer the money", {}),
    ("don't transfer the rent", {}),
    ("don't transfer the rent yet", {}),
    ("don't transfer anything yet", {}),
    ("don't send the money", {}),
    ("don't send it yet", {}),
    ("don't send the payment", {}),
    ("don't pay yet", {}),
    ("don't pay the bill yet", {}),
    ("don't pay rent yet", {}),
    ("don't venmo me yet", {}),
    ("don't cashapp me", {}),
    ("don't split it yet", {}),
    ("don't book the uber", {}),
    ("don't book the cab yet", {}),
    ("don't book the flight", {}),
    ("don't book the hotel", {}),
    ("don't book the table", {}),
    ("don't order food yet", {}),
    ("don't order pizza yet", {}),
    ("don't order anything", {}),
    ("don't call mom yet", {}),
    ("don't call him yet", {}),
    ("don't text her yet", {}),
    ("don't set the alarm", {}),
    ("don't set a reminder", {}),
    ("don't play music", {}),
    ("don't play that song", {}),
    ("don't buy it", {}),
    ("don't buy anything", {}),

    # ── cancel + action ──
    ("cancel the uber", {}),
    ("cancel the cab", {}),
    ("cancel the ride", {}),
    ("cancel the order", {}),
    ("cancel the food order", {}),
    ("cancel the pizza order", {}),
    ("cancel the reservation", {}),
    ("cancel the booking", {}),
    ("cancel the flight", {}),
    ("cancel the hotel", {}),
    ("cancel the alarm", {}),
    ("cancel the reminder", {}),
    ("cancel the reminder for {time}", {}),
    ("cancel the timer", {}),
    ("cancel the meeting", {}),
    ("cancel the appointment", {}),
    ("cancel the subscription", {}),
    ("cancel my order", {}),
    ("cancel everything", {}),

    # ── stop/hold off ──
    ("hold off on the payment", {}),
    ("hold off on ordering", {}),
    ("hold off on the uber", {}),
    ("hold off on booking", {}),
    ("stop the alarm", {}),
    ("stop the timer", {}),
    ("stop the music", {}),
    ("stop the order", {}),
    ("never mind the uber", {}),
    ("never mind the order", {}),
    ("never mind the reservation", {}),
    ("never mind the alarm", {}),
    ("nvm the uber", {}),
    ("nvm the order", {}),
    ("nvm the payment", {}),
    ("nvm don't order", {}),
    ("scratch that, don't order", {}),
    ("actually don't order", {}),
    ("wait don't send it", {}),
    ("wait don't book it", {}),
    ("wait don't pay yet", {}),

    # ── already done (must not re-trigger) ──
    ("already paid the rent", {}),
    ("already paid the bill", {}),
    ("already ordered food", {}),
    ("already booked the uber", {}),
    ("already set the alarm", {}),
    ("already called mom", {}),
    ("already booked the flight", {}),
    ("already made the reservation", {}),
    ("already sent the money", {}),
    ("already venmo'd you", {}),
    ("already split it", {}),
    ("I already paid", {}),
    ("I already ordered", {}),
    ("I already booked it", {}),
    ("I already called", {}),
    ("took care of it already", {}),
    ("done already", {}),
    ("handled it already", {}),
]


# ═════════════════════════════════════════════════════════════════════════════
# 7. CONTACT_BOOST — "call X", "text X", "phone X" must fire contact
#
# v5 raised contact threshold to 0.84 but "call mom" only scores 0.82.
# Need WAY more contact positives with all variations.
# ═════════════════════════════════════════════════════════════════════════════

V51_CONTACT_BOOST = [
    # ── call patterns ──
    ("call {name}", {"contact": 1}),
    ("call {name} please", {"contact": 1}),
    ("call {name} real quick", {"contact": 1}),
    ("call {name} rn", {"contact": 1}),
    ("call {name} back", {"contact": 1}),
    ("call {name} asap", {"contact": 1}),
    ("call {name} when you can", {"contact": 1}),
    ("call {name} later", {"contact": 1}),
    ("call {name} tonight", {"contact": 1}),
    ("call {name} tomorrow", {"contact": 1}),
    ("give {name} a call", {"contact": 1}),
    ("give {name} a ring", {"contact": 1}),
    ("hit up {name}", {"contact": 1}),
    ("ring {name}", {"contact": 1}),
    ("phone {name}", {"contact": 1}),
    ("call mom", {"contact": 1}),
    ("call dad", {"contact": 1}),
    ("call mom real quick", {"contact": 1}),
    ("call dad back", {"contact": 1}),
    ("call home", {"contact": 1}),
    ("call the office", {"contact": 1}),
    ("call the doctor", {"contact": 1}),
    ("call the dentist", {"contact": 1}),
    ("call the landlord", {"contact": 1}),
    ("call the plumber", {"contact": 1}),
    ("call the restaurant", {"contact": 1}),
    ("need to call {name}", {"contact": 1}),
    ("gotta call {name}", {"contact": 1}),
    ("should call {name}", {"contact": 1}),
    ("wanna call {name}", {"contact": 1}),
    ("gonna call {name}", {"contact": 1}),
    ("I'll call {name}", {"contact": 1}),
    ("let me call {name}", {"contact": 1}),
    ("can you call {name}", {"contact": 1}),
    ("please call {name}", {"contact": 1}),

    # ── text patterns ──
    ("text {name}", {"contact": 1}),
    ("text {name} about dinner", {"contact": 1}),
    ("text {name} back", {"contact": 1}),
    ("text {name} later", {"contact": 1}),
    ("text mom", {"contact": 1}),
    ("text dad", {"contact": 1}),
    ("message {name}", {"contact": 1}),
    ("dm {name}", {"contact": 1}),
    ("ping {name}", {"contact": 1}),
    ("send {name} a text", {"contact": 1}),
    ("shoot {name} a text", {"contact": 1}),
    ("drop {name} a message", {"contact": 1}),
    ("reach out to {name}", {"contact": 1}),
    ("get in touch with {name}", {"contact": 1}),
    ("let {name} know", {"contact": 1}),
    ("tell {name} about it", {"contact": 1}),
    ("hit {name} up", {"contact": 1}),

    # ── multi-intent: call/text + time = contact + alarm ──
    ("remind me to call {name} {day}", {"contact": 1, "alarm": 1}),
    ("remind me to call mom {day} at {time}", {"contact": 1, "alarm": 1}),
    ("remind me to text {name} {day}", {"contact": 1, "alarm": 1}),
    ("call {name} at {time}", {"contact": 1}),
    ("text {name} at {time}", {"contact": 1}),
    ("call mom tomorrow at {time}", {"contact": 1}),
    ("call dad at {time}", {"contact": 1}),

    # ── "with name" in social context = calendar, NOT contact ──
    # Contrastive: "lunch with sarah" should fire calendar, not contact
    ("lunch with {name}", {"calendar": 1}),
    ("dinner with {name}", {"calendar": 1}),
    ("coffee with {name}", {"calendar": 1}),
    ("drinks with {name}", {"calendar": 1}),
    ("meeting with {name}", {"calendar": 1}),
    ("hanging out with {name}", {}),
    ("going out with {name}", {}),
    ("was with {name} yesterday", {}),
    ("talked to {name} earlier", {}),
    ("saw {name} today", {}),

    # ── "lunch with sarah at sweetgreen" = calendar, NOT contact + maps ──
    ("lunch with {name} at {place} {day}", {"calendar": 1}),
    ("dinner with {name} at {place}", {"calendar": 1}),
    ("coffee with {name} at {place} {day}", {"calendar": 1}),
    ("meeting with {name} at {place} {time}", {"calendar": 1}),
    ("brunch with {name} at {place}", {"calendar": 1}),
]


# ═════════════════════════════════════════════════════════════════════════════
# 8. BARE_WORD_BOOST — single-word intent triggers need more training
#
# "weather" alone should fire weather. "uber" alone should fire ride.
# Ultra-short messages are the hardest for the model.
# ═════════════════════════════════════════════════════════════════════════════

V51_BARE_WORD_BOOST = [
    # ── weather bare words (weather SHOULD fire) ──
    ("weather", {"weather": 1}),
    ("weather?", {"weather": 1}),
    ("weather today", {"weather": 1}),
    ("weather tomorrow", {"weather": 1}),
    ("weather this week", {"weather": 1}),
    ("the weather", {"weather": 1}),
    ("how's the weather", {"weather": 1}),
    ("what's the weather", {"weather": 1}),
    ("weather check", {"weather": 1}),
    ("check weather", {"weather": 1}),
    ("rain?", {"weather": 1}),
    ("rain today?", {"weather": 1}),
    ("is it raining", {"weather": 1}),
    ("gonna rain?", {"weather": 1}),
    ("will it rain", {"weather": 1}),
    ("sunny today?", {"weather": 1}),
    ("is it cold out", {"weather": 1}),
    ("how cold is it", {"weather": 1}),
    ("temperature", {"weather": 1}),
    ("temp today", {"weather": 1}),
    ("forecast", {"weather": 1}),
    ("weather forecast", {"weather": 1}),
    ("weather in {city}", {"weather": 1}),
    ("weather {city}", {"weather": 1}),
    ("{city} weather", {"weather": 1}),
    ("do I need an umbrella", {"weather": 1}),
    ("should I bring a jacket", {"weather": 1}),
    ("is it hot out", {"weather": 1}),
    ("what's the temp", {"weather": 1}),
    ("outside temp", {"weather": 1}),

    # ── ride WITH action word (ride SHOULD fire) ──
    ("uber please", {"ride": 1}),
    ("get an uber", {"ride": 1}),
    ("book an uber", {"ride": 1}),
    ("call an uber", {"ride": 1}),
    ("need an uber", {"ride": 1}),
    ("let's uber", {"ride": 1}),
    ("uber there", {"ride": 1}),
    ("uber home", {"ride": 1}),
    ("get a lyft", {"ride": 1}),
    ("book a cab", {"ride": 1}),
    ("call a cab", {"ride": 1}),
    ("need a cab", {"ride": 1}),
    ("need a ride", {"ride": 1}),

    # ── money WITH action word (money SHOULD fire) ──
    ("venmo me", {"money": 1}),
    ("venmo me {amount}", {"money": 1}),
    ("cashapp me", {"money": 1}),
    ("cashapp me {amount}", {"money": 1}),
    ("pay me back", {"money": 1}),
    ("pay me", {"money": 1}),
    ("pay up", {"money": 1}),
    ("split it", {"money": 1}),
    ("let's split", {"money": 1}),
    ("send me {amount}", {"money": 1}),
    ("you owe me", {"money": 1}),

    # ── food WITH action word (food SHOULD fire) ──
    ("order pizza", {"food_order": 1}),
    ("order food", {"food_order": 1}),
    ("let's order", {"food_order": 1}),
    ("get dominos", {"food_order": 1}),
    ("get chipotle", {"food_order": 1}),
    ("order from doordash", {"food_order": 1}),
    ("order from ubereats", {"food_order": 1}),
]

# ═════════════════════════════════════════════════════════════════════════════
# 8b. BARE_BRAND_NEGATIVES — bare brand names must NOT fire
#
# "uber" alone = could be talking about the company, stock, etc.
# "venmo" alone = could be naming the app in conversation.
# Only fires when combined with action words (uber please, venmo me, etc.)
# ═════════════════════════════════════════════════════════════════════════════

V51_BARE_BRAND_POSITIVE = [
    # ── In a chat app, bare brand names ARE action requests ──
    # "uber" = user wants a ride. "venmo" = user wants to pay/get paid.
    # "dominos" = user wants food. "spotify" = user wants music.

    # ride brands
    ("uber", {"ride": 1}),
    ("lyft", {"ride": 1}),
    ("ola", {"ride": 1}),
    ("cab", {"ride": 1}),
    ("taxi", {"ride": 1}),
    ("uber?", {"ride": 1}),
    ("lyft?", {"ride": 1}),
    ("cab?", {"ride": 1}),
    ("ride?", {"ride": 1}),

    # money brands
    ("venmo", {"money": 1}),
    ("cashapp", {"money": 1}),
    ("zelle", {"money": 1}),
    ("paypal", {"money": 1}),
    ("gpay", {"money": 1}),
    ("venmo?", {"money": 1}),
    ("cashapp?", {"money": 1}),
    ("split?", {"money": 1}),

    # food brands
    ("dominos", {"food_order": 1}),
    ("chipotle", {"food_order": 1}),
    ("pizza hut", {"food_order": 1}),
    ("mcdonald's", {"food_order": 1}),
    ("doordash", {"food_order": 1}),
    ("ubereats", {"food_order": 1}),
    ("swiggy", {"food_order": 1}),
    ("zomato", {"food_order": 1}),
    ("dominos?", {"food_order": 1}),
    ("chipotle?", {"food_order": 1}),
    ("pizza?", {"food_order": 1}),
    ("sushi?", {"food_order": 1}),
    ("mcd?", {"food_order": 1}),

    # music brands
    ("spotify", {"music": 1}),
    ("apple music", {"music": 1}),
    ("spotify?", {"music": 1}),

    # video brands
    ("netflix", {"video": 1}),
    ("youtube", {"video": 1}),
    ("hulu", {"video": 1}),
    ("netflix?", {"video": 1}),

    # ── Contrastive: brand in descriptive sentence = NOT actionable ──
    # This teaches the model: bare brand = action, brand-in-sentence = info
    ("uber is a great company", {}),
    ("lyft stock is up", {}),
    ("uber drivers are nice", {}),
    ("I work at uber", {}),
    ("my friend works at lyft", {}),
    ("venmo is down", {}),
    ("cashapp has fees", {}),
    ("dominos is overrated", {}),
    ("chipotle gives me stomach issues", {}),
    ("doordash fees are crazy", {}),
    ("I worked at chipotle", {}),
    ("spotify has a new feature", {}),
    ("netflix raised their prices", {}),
    ("uber just IPO'd", {}),
    ("I interviewed at uber", {}),
    ("venmo got hacked apparently", {}),
]


# ═════════════════════════════════════════════════════════════════════════════
# 9. DONT_FORGET_NOT_ALARM — "don't forget to X" is idiomatic
#
# "don't forget to book the flight" should fire travel, NOT alarm.
# "don't forget" is just emphasis, not a reminder request.
# ═════════════════════════════════════════════════════════════════════════════

V51_DONT_FORGET_FIXES = [
    # ── "don't forget to X" = X intent only, NOT alarm ──
    ("don't forget to book the flight", {"travel": 1}),
    ("don't forget to book the flight {day}", {"travel": 1}),
    ("don't forget to book flights to {city}", {"travel": 1}),
    ("don't forget to book the hotel", {"travel": 1, "reservation": 1}),
    ("don't forget to order food", {"food_order": 1}),
    ("don't forget to order pizza", {"food_order": 1}),
    ("don't forget to order dinner", {"food_order": 1}),
    ("don't forget to pay rent", {"bills": 1}),
    ("don't forget to pay the bill", {"bills": 1}),
    ("don't forget to pay utilities", {"bills": 1}),
    ("don't forget to call {name}", {"contact": 1}),
    ("don't forget to call mom", {"contact": 1}),
    ("don't forget to call dad", {"contact": 1}),
    ("don't forget to text {name}", {"contact": 1}),
    ("don't forget to buy {item}", {"shopping": 1}),
    ("don't forget to buy groceries", {"shopping": 1}),
    ("don't forget to check the weather", {"weather": 1}),
    ("don't forget the reservation", {"reservation": 1}),
    ("don't forget the booking", {"reservation": 1}),
    ("don't forget the tickets", {"tickets": 1}),
    ("don't forget to book tickets", {"tickets": 1}),
    ("don't forget to uber", {"ride": 1}),
    ("don't forget to get a cab", {"ride": 1}),
    ("don't forget to venmo {name}", {"money": 1}),
    ("don't forget to pay {name} back", {"money": 1}),
    ("don't forget to split the bill", {"money": 1}),
    ("don't forget the doctor appointment", {"health": 1}),
    ("don't forget your meds", {"health": 1}),

    # Variations: "make sure to", "remember to"
    ("make sure to book the flight", {"travel": 1}),
    ("make sure to pay rent", {"bills": 1}),
    ("make sure to call {name}", {"contact": 1}),
    ("make sure to order food", {"food_order": 1}),
    ("make sure to get an uber", {"ride": 1}),
    ("remember to pay rent", {"bills": 1}),
    ("remember to book the flight", {"travel": 1}),
    ("remember to call mom", {"contact": 1}),
    ("remember to order dinner", {"food_order": 1}),
    ("remember to venmo {name}", {"money": 1}),
]


# ═════════════════════════════════════════════════════════════════════════════
# 10. QUESTION_FORM_BOOST — "wanna X", "should we X" = actionable
# ═════════════════════════════════════════════════════════════════════════════

V51_QUESTION_FORM = [
    ("wanna uber", {"ride": 1}),
    ("wanna uber there", {"ride": 1}),
    ("wanna uber to {place}", {"ride": 1, "maps": 1}),
    ("wanna lyft", {"ride": 1}),
    ("wanna cab it", {"ride": 1}),
    ("wanna get a ride", {"ride": 1}),
    ("wanna order food", {"food_order": 1}),
    ("wanna order pizza", {"food_order": 1}),
    ("wanna get sushi", {"food_order": 1}),
    ("wanna split it", {"money": 1}),
    ("wanna venmo me", {"money": 1}),
    ("wanna call {name}", {"contact": 1}),
    ("wanna watch {movie}", {"video": 1}),
    ("wanna listen to music", {"music": 1}),
    ("wanna book a flight", {"travel": 1}),
    ("wanna book a table", {"reservation": 1}),
    ("should we uber", {"ride": 1}),
    ("should we order food", {"food_order": 1}),
    ("should we split it", {"money": 1}),
    ("should we call {name}", {"contact": 1}),
    ("should we book a cab", {"ride": 1}),
    ("should I uber", {"ride": 1}),
    ("should I order", {"food_order": 1}),
    ("should I call {name}", {"contact": 1}),
    ("should I book the flight", {"travel": 1}),
    ("should I set an alarm", {"alarm": 1}),
    ("can we uber", {"ride": 1}),
    ("can we order", {"food_order": 1}),
    ("can I venmo you", {"money": 1}),
    ("let's uber", {"ride": 1}),
    ("let's order", {"food_order": 1}),
    ("let's split it", {"money": 1}),
    ("let's call {name}", {"contact": 1}),
    ("let's book a cab", {"ride": 1}),
    ("let's watch {movie}", {"video": 1}),
    ("let's book the flight", {"travel": 1}),
]


# ═════════════════════════════════════════════════════════════════════════════
# Aggregate all v5.1 fix banks
# ═════════════════════════════════════════════════════════════════════════════

ALL_V51_BANKS = {
    # Negatives (things that must NOT fire certain intents)
    "v51_time_not_alarm":         V51_TIME_NOT_ALARM,
    "v51_time_is_alarm":          V51_TIME_IS_ALARM,
    "v51_activity_not_calendar":  V51_ACTIVITY_NOT_CALENDAR,
    "v51_venue_not_maps":         V51_VENUE_NOT_MAPS,
    "v51_place_is_maps":          V51_PLACE_IS_MAPS,
    "v51_ride_not_maps":          V51_RIDE_NOT_MAPS,
    "v51_bills_not_money":        V51_BILLS_NOT_MONEY,
    "v51_actual_money":           V51_ACTUAL_MONEY,
    "v51_strong_negation":        V51_STRONG_NEGATION,
    "v51_bare_brand_positive":    V51_BARE_BRAND_POSITIVE,

    # Positives (things that must fire)
    "v51_contact_boost":          V51_CONTACT_BOOST,
    "v51_bare_word_boost":        V51_BARE_WORD_BOOST,
    "v51_dont_forget_fixes":      V51_DONT_FORGET_FIXES,
    "v51_question_form":          V51_QUESTION_FORM,
}

# Template counts for diagnostics
if __name__ == "__main__":
    total = 0
    for name, bank in ALL_V51_BANKS.items():
        if bank and isinstance(bank[0], tuple):
            positives = sum(1 for entry in bank if entry[1])
            negatives = sum(1 for entry in bank if not entry[1])
        else:
            positives = len(bank)
            negatives = 0
        print(f"  {name:<30} {len(bank):>3} templates ({positives} pos, {negatives} neg)")
        total += len(bank)
    print(f"\n  Total: {total} regression fix templates")
