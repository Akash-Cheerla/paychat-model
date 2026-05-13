"""
v5.1 SMART MODEL — teaches the model to UNDERSTAND intent, not match keywords.

Design principle: every pattern has a contrastive pair.
  "uber to JFK at 5am" → ride+maps (time belongs to ride, NOT alarm)
  "remind me at 5am"   → alarm (time belongs to alarm)

Same word "at 5am", completely different outcome based on WHAT it modifies.
THAT is what makes the model smart.

7 core skills:
  1. Time ownership — who does the time belong to?
  2. Place ownership — venue vs navigation vs just talking?
  3. Bare word intelligence — brand in chat = action, brand in sentence = info
  4. Negation spectrum — "don't X" vs "don't forget to X" vs "cancel X"
  5. Person ownership — my action vs their action vs gossip
  6. Bill vs P2P money — rent/utilities vs venmo/split
  7. Action modifiers — "wanna", "should we", "let's" = actionable questions
"""

# ═══════════════════════════════════════════════════════════════════════
# SKILL 1: TIME OWNERSHIP
#
# The model must learn: WHO does the time belong to?
#   "uber at 5am"     → 5am = ride departure time (NOT alarm)
#   "remind me at 5am" → 5am = alarm time
#   "movie at 9pm"    → 9pm = showtime (NOT alarm)
#   "flight at 6am"   → 6am = departure (NOT alarm)
#
# Rule: time belongs to the ACTION VERB, not to alarm by default.
# alarm only fires when the user EXPLICITLY asks for reminder/alarm/wake.
# ═══════════════════════════════════════════════════════════════════════

# TIME belongs to RIDE (alarm must NOT fire)
SMART_TIME_RIDE = [
    ("uber at {time}", {"ride": 1}),
    ("uber at {time} {day}", {"ride": 1}),
    ("uber to {place} at {time}", {"ride": 1, "maps": 1}),
    ("uber to {airport} at {time} {day}", {"ride": 1, "maps": 1}),
    ("lyft at {time}", {"ride": 1}),
    ("lyft to {place} at {time}", {"ride": 1, "maps": 1}),
    ("cab at {time}", {"ride": 1}),
    ("cab to {place} at {time}", {"ride": 1, "maps": 1}),
    ("book uber for {time}", {"ride": 1}),
    ("book uber for {time} {day}", {"ride": 1}),
    ("schedule uber for {time}", {"ride": 1}),
    ("get a cab at {time}", {"ride": 1}),
    ("need a ride at {time}", {"ride": 1}),
    ("pick me up at {time}", {"ride": 1}),
    ("the uber is at {time}", {"ride": 1}),
    ("ride at {time}", {"ride": 1}),
    ("uber {time} {day}", {"ride": 1}),
    ("car at {time}", {"ride": 1}),
    ("uber to the airport at {time}", {"ride": 1, "maps": 1}),
    ("lyft to {airport} at {time} {day}", {"ride": 1, "maps": 1}),
    ("grab an uber at {time}", {"ride": 1}),
    ("uber to {place} {time} {day}", {"ride": 1, "maps": 1}),
    ("let's uber at {time}", {"ride": 1}),
    ("we need an uber at {time}", {"ride": 1}),
    ("book a ride for {time}", {"ride": 1}),
    ("uber arriving at {time}", {"ride": 1}),
    ("pick up at {time}", {"ride": 1}),
    ("cab for {time} tomorrow", {"ride": 1}),
    ("let's get a lyft at {time}", {"ride": 1}),
    ("book cab {time}", {"ride": 1}),
]

# TIME belongs to TRAVEL (alarm must NOT fire)
SMART_TIME_TRAVEL = [
    ("flight at {time}", {"travel": 1}),
    ("flight at {time} {day}", {"travel": 1}),
    ("flight to {city} at {time}", {"travel": 1}),
    ("train at {time}", {"travel": 1}),
    ("bus at {time}", {"travel": 1}),
    ("the flight leaves at {time}", {"travel": 1}),
    ("departure at {time}", {"travel": 1}),
    ("boarding at {time}", {"travel": 1}),
    ("my flight is at {time} {day}", {"travel": 1}),
    ("catching the {time} train", {"travel": 1}),
    ("train to {city} at {time}", {"travel": 1}),
    ("bus departs at {time}", {"travel": 1}),
    ("flight lands at {time}", {"travel": 1}),
    ("arriving at {time}", {"travel": 1}),
    ("check in at {time}", {"travel": 1}),
    ("takeoff is at {time}", {"travel": 1}),
    ("the bus is at {time}", {"travel": 1}),
    ("ferry at {time}", {"travel": 1}),
    ("layover until {time}", {"travel": 1}),
    ("connecting flight at {time}", {"travel": 1}),
]

# TIME belongs to VIDEO/TICKETS (alarm must NOT fire)
SMART_TIME_VIDEO = [
    ("movie at {time}", {"video": 1}),
    ("movie at {time} tonight", {"video": 1}),
    ("{movie} at {time}", {"video": 1, "tickets": 1}),
    ("show at {time}", {"video": 1}),
    ("{show} starts at {time}", {"video": 1}),
    ("screening at {time}", {"video": 1, "tickets": 1}),
    ("the movie is at {time}", {"video": 1}),
    ("concert at {time}", {"tickets": 1}),
    ("concert starts at {time}", {"tickets": 1}),
    ("event at {time}", {"tickets": 1}),
    ("game at {time}", {"tickets": 1}),
    ("match at {time}", {"tickets": 1}),
    ("showtime is {time}", {"video": 1}),
    ("it starts at {time}", {"video": 1}),
    ("premiere at {time}", {"video": 1, "tickets": 1}),
]

# TIME belongs to FOOD (alarm must NOT fire)
SMART_TIME_FOOD = [
    ("dinner at {time}", {"food_order": 1}),
    ("lunch at {time}", {"food_order": 1}),
    ("order food for {time}", {"food_order": 1}),
    ("pizza at {time}", {"food_order": 1}),
    ("delivery at {time}", {"food_order": 1}),
    ("doordash for {time}", {"food_order": 1}),
    ("food arriving at {time}", {"food_order": 1}),
    ("order for {time}", {"food_order": 1}),
    ("breakfast at {time}", {"food_order": 1}),
    ("takeout at {time}", {"food_order": 1}),
]

# TIME belongs to RESERVATION (alarm must NOT fire)
SMART_TIME_RESERVATION = [
    ("reservation at {time}", {"reservation": 1}),
    ("table at {time}", {"reservation": 1}),
    ("booking for {time}", {"reservation": 1}),
    ("dinner reservation {time}", {"reservation": 1}),
    ("brunch at {time}", {"reservation": 1}),
    ("book a table for {time}", {"reservation": 1}),
    ("reservation {day} at {time}", {"reservation": 1}),
    ("table for {qty} at {time}", {"reservation": 1}),
    ("seated at {time}", {"reservation": 1}),
    ("check in at {time}", {"reservation": 1}),
]

# TIME belongs to HEALTH (alarm must NOT fire)
SMART_TIME_HEALTH = [
    ("appointment at {time}", {"health": 1}),
    ("doctor at {time}", {"health": 1}),
    ("dentist at {time}", {"health": 1}),
    ("checkup at {time}", {"health": 1}),
    ("therapy at {time}", {"health": 1}),
    ("appointment {day} at {time}", {"health": 1}),
    ("doctor's at {time} {day}", {"health": 1}),
    ("my appointment is at {time}", {"health": 1}),
    ("session at {time}", {"health": 1}),
    ("lab work at {time}", {"health": 1}),
]

# TIME belongs to CALENDAR (alarm must NOT fire)
SMART_TIME_CALENDAR = [
    ("meeting at {time}", {"calendar": 1}),
    ("meeting {day} at {time}", {"calendar": 1}),
    ("standup at {time}", {"calendar": 1}),
    ("call at {time}", {"calendar": 1}),
    ("1:1 at {time}", {"calendar": 1}),
    ("sync at {time}", {"calendar": 1}),
    ("interview at {time}", {"calendar": 1}),
    ("class at {time}", {"calendar": 1}),
    ("lecture at {time}", {"calendar": 1}),
    ("yoga at {time}", {"calendar": 1}),
    ("gym at {time}", {"calendar": 1}),
]

# Contrastive: TIME actually IS for alarm (alarm SHOULD fire)
SMART_TIME_ALARM = [
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
    ("don't let me sleep past {time}", {"alarm": 1}),
    ("{duration} timer", {"alarm": 1}),
    ("remind me {day} at {time}", {"alarm": 1}),
    ("reminder at {time}", {"alarm": 1}),
    ("wake up call at {time}", {"alarm": 1}),
    ("alarm {time} {day}", {"alarm": 1}),
    ("poke me at {time}", {"alarm": 1}),
]

# Multi-intent: action + EXPLICIT alarm request (alarm SHOULD fire alongside)
SMART_TIME_MULTI_WITH_ALARM = [
    ("uber to {airport} at {time}, remind me to pack tonight", {"ride": 1, "maps": 1, "alarm": 1}),
    ("flight at {time}, set alarm for {time}", {"travel": 1, "alarm": 1}),
    ("meeting at {time}, remind me {duration} before", {"calendar": 1, "alarm": 1}),
    ("dinner at {time}, remind me to make reservation", {"food_order": 1, "alarm": 1}),
    ("movie at {time}, remind me to buy tickets", {"video": 1, "alarm": 1}),
    ("doctor at {time}, set a reminder", {"health": 1, "alarm": 1}),
    ("{event} at {time}, wake me at {time}", {"calendar": 1, "alarm": 1}),
    ("flight {day} at {time}, alarm for {time}", {"travel": 1, "alarm": 1}),
    ("uber at {time}, also remind me to pack", {"ride": 1, "alarm": 1}),
    ("concert at {time}, don't let me forget", {"tickets": 1, "alarm": 1}),
]


# ═══════════════════════════════════════════════════════════════════════
# SKILL 2: PLACE OWNERSHIP
#
# The model must learn: is this place a DESTINATION or a VENUE?
#   "lunch at sweetgreen"     → venue for calendar event (NOT maps)
#   "directions to sweetgreen" → navigation (maps)
#   "oppenheimer at amc"      → theater venue (NOT maps)
#   "heading to amc"          → navigation (maps)
#   "meet me at sweetgreen"   → telling someone WHERE (maps)
# ═══════════════════════════════════════════════════════════════════════

# PLACE is a VENUE for an event (maps must NOT fire)
SMART_PLACE_VENUE = [
    # restaurant/cafe as social event venue
    ("lunch at {place} {day}", {"calendar": 1}),
    ("dinner at {place} {day}", {"calendar": 1}),
    ("brunch at {place} {day}", {"calendar": 1}),
    ("coffee at {place} {day}", {"calendar": 1}),
    ("drinks at {place} {day}", {"calendar": 1}),
    ("lunch with {name} at {place}", {"calendar": 1}),
    ("dinner with {name} at {place} {day}", {"calendar": 1}),
    ("team lunch at {place}", {"calendar": 1}),
    ("team dinner at {place} {day}", {"calendar": 1}),
    ("party at {place} {day}", {"calendar": 1}),
    ("birthday at {place}", {"calendar": 1}),
    ("celebration at {place}", {"calendar": 1}),
    ("get together at {place}", {"calendar": 1}),
    ("meetup at {place} {day}", {"calendar": 1}),
    ("event at {place} {day}", {"calendar": 1}),
    ("hangout at {place} {day}", {"calendar": 1}),

    # restaurant as reservation venue
    ("book a table at {place}", {"reservation": 1}),
    ("book a table at {place} {day} at {time}", {"reservation": 1}),
    ("reservation at {place}", {"reservation": 1}),
    ("reservation at {place} for {qty}", {"reservation": 1}),
    ("table for {qty} at {place}", {"reservation": 1}),
    ("reserve {place} for {day}", {"reservation": 1}),
    ("book {place} for {qty} people", {"reservation": 1}),
    ("get a table at {place}", {"reservation": 1}),
    ("dinner reservation at {place}", {"reservation": 1}),
    ("book us in at {place}", {"reservation": 1}),

    # theater/venue for entertainment
    ("{movie} at amc", {"video": 1, "tickets": 1}),
    ("{movie} at amc tonight", {"video": 1, "tickets": 1}),
    ("{movie} at amc at {time}", {"video": 1, "tickets": 1}),
    ("{movie} at the theater", {"video": 1, "tickets": 1}),
    ("{movie} at regal", {"video": 1, "tickets": 1}),
    ("{movie} at imax", {"video": 1, "tickets": 1}),
    ("movie at amc {day}", {"video": 1, "tickets": 1}),
    ("show at {place} tonight", {"tickets": 1}),
    ("concert at {place} {day}", {"tickets": 1}),
    ("comedy show at {place}", {"tickets": 1}),
    ("stand-up at {place}", {"tickets": 1}),
    ("game at {place}", {"tickets": 1}),
]

# PLACE is a DESTINATION (maps SHOULD fire)
SMART_PLACE_NAVIGATION = [
    ("directions to {place}", {"maps": 1}),
    ("how do I get to {place}", {"maps": 1}),
    ("navigate to {place}", {"maps": 1}),
    ("heading to {place}", {"maps": 1}),
    ("on my way to {place}", {"maps": 1}),
    ("omw to {place}", {"maps": 1}),
    ("driving to {place}", {"maps": 1}),
    ("going to {place}", {"maps": 1}),
    ("take me to {place}", {"maps": 1}),
    ("route to {place}", {"maps": 1}),
    ("how far is {place}", {"maps": 1}),
    ("where is {place}", {"maps": 1}),
    ("where's {place}", {"maps": 1}),
    ("find {place} on maps", {"maps": 1}),
    ("map to {place}", {"maps": 1}),
    ("address for {place}", {"maps": 1}),
    ("drop a pin at {place}", {"maps": 1}),
    ("pulling up to {place}", {"maps": 1}),
    ("meet me at {place}", {"maps": 1}),
    ("I'm at {place}", {"maps": 1}),
    ("waiting at {place}", {"maps": 1}),
    ("come to {place}", {"maps": 1}),
    ("the address is {address}", {"maps": 1}),
    ("we're at {address}", {"maps": 1}),
]

# RIDE already implies destination (maps must NOT double-fire)
SMART_RIDE_NOT_MAPS = [
    ("wanna uber there", {"ride": 1}),
    ("let's uber there", {"ride": 1}),
    ("uber there?", {"ride": 1}),
    ("cab there", {"ride": 1}),
    ("lyft there", {"ride": 1}),
    ("let's take an uber", {"ride": 1}),
    ("should we uber", {"ride": 1}),
    ("uber it", {"ride": 1}),
    ("cab it", {"ride": 1}),
    ("let's cab it", {"ride": 1}),
    ("uber over", {"ride": 1}),
    ("uber back", {"ride": 1}),
    ("uber home", {"ride": 1}),
    ("lyft home", {"ride": 1}),
    ("cab home", {"ride": 1}),
    ("ride home", {"ride": 1}),
    ("grab an uber", {"ride": 1}),
    ("grab a lyft", {"ride": 1}),
    ("get a cab", {"ride": 1}),
    ("call a cab", {"ride": 1}),
    ("hail a cab", {"ride": 1}),
    ("book a ride", {"ride": 1}),
    ("need a ride", {"ride": 1}),
    ("uber to dinner", {"ride": 1}),
    ("uber to the party", {"ride": 1}),
    ("uber to the event", {"ride": 1}),
    ("uber to the concert", {"ride": 1}),
    ("uber to the game", {"ride": 1}),
    ("cab to the airport", {"ride": 1}),
    ("lyft to the station", {"ride": 1}),
]

# RIDE with explicit destination (maps SHOULD fire alongside)
SMART_RIDE_WITH_MAPS = [
    ("uber to {place}", {"ride": 1, "maps": 1}),
    ("uber to {airport}", {"ride": 1, "maps": 1}),
    ("lyft to {place}", {"ride": 1, "maps": 1}),
    ("cab to {place}", {"ride": 1, "maps": 1}),
    ("uber to {address}", {"ride": 1, "maps": 1}),
    ("ride to {place}", {"ride": 1, "maps": 1}),
    ("uber to {city}", {"ride": 1, "maps": 1}),
    ("uber me to {place}", {"ride": 1, "maps": 1}),
    ("book uber to {airport}", {"ride": 1, "maps": 1}),
    ("cab to {address}", {"ride": 1, "maps": 1}),
]


# ═══════════════════════════════════════════════════════════════════════
# SKILL 3: BARE WORD INTELLIGENCE
#
# In a chat between friends:
#   "uber"                → ride (they want a ride)
#   "uber is hiring"      → nothing (talking ABOUT uber)
#   "uber drivers are nice" → nothing (discussing the service)
#   "dominos"             → food_order (they want food)
#   "dominos is overrated" → nothing (opinion about the brand)
#
# Rule: bare brand = action request. Brand in descriptive sentence = info.
# ═══════════════════════════════════════════════════════════════════════

# Bare brand names = action (SHOULD fire)
SMART_BARE_ACTION = [
    # ride
    ("uber", {"ride": 1}),
    ("uber?", {"ride": 1}),
    ("lyft", {"ride": 1}),
    ("lyft?", {"ride": 1}),
    ("ola", {"ride": 1}),
    ("ola?", {"ride": 1}),
    ("cab", {"ride": 1}),
    ("cab?", {"ride": 1}),
    ("taxi", {"ride": 1}),
    ("taxi?", {"ride": 1}),

    # money
    ("venmo", {"money": 1}),
    ("venmo?", {"money": 1}),
    ("cashapp", {"money": 1}),
    ("cashapp?", {"money": 1}),
    ("zelle", {"money": 1}),
    ("zelle?", {"money": 1}),
    ("gpay", {"money": 1}),
    ("gpay?", {"money": 1}),
    ("paypal", {"money": 1}),
    ("paypal?", {"money": 1}),
    ("split?", {"money": 1}),

    # food
    ("dominos", {"food_order": 1}),
    ("dominos?", {"food_order": 1}),
    ("chipotle", {"food_order": 1}),
    ("chipotle?", {"food_order": 1}),
    ("pizza", {"food_order": 1}),
    ("pizza?", {"food_order": 1}),
    ("sushi", {"food_order": 1}),
    ("sushi?", {"food_order": 1}),
    ("doordash", {"food_order": 1}),
    ("doordash?", {"food_order": 1}),
    ("ubereats", {"food_order": 1}),
    ("ubereats?", {"food_order": 1}),
    ("swiggy", {"food_order": 1}),
    ("swiggy?", {"food_order": 1}),
    ("zomato", {"food_order": 1}),
    ("zomato?", {"food_order": 1}),
    ("mcd", {"food_order": 1}),
    ("mcd?", {"food_order": 1}),
    ("kfc", {"food_order": 1}),
    ("kfc?", {"food_order": 1}),
    ("biryani", {"food_order": 1}),
    ("biryani?", {"food_order": 1}),
    ("noodles", {"food_order": 1}),
    ("noodles?", {"food_order": 1}),
    ("burger", {"food_order": 1}),
    ("burgers", {"food_order": 1}),
    ("tacos", {"food_order": 1}),
    ("tacos?", {"food_order": 1}),

    # music
    ("spotify", {"music": 1}),
    ("spotify?", {"music": 1}),
    ("apple music", {"music": 1}),
    ("music", {"music": 1}),
    ("music?", {"music": 1}),

    # video
    ("netflix", {"video": 1}),
    ("netflix?", {"video": 1}),
    ("youtube", {"video": 1}),
    ("youtube?", {"video": 1}),
    ("hulu", {"video": 1}),
    ("hulu?", {"video": 1}),

    # weather
    ("weather", {"weather": 1}),
    ("weather?", {"weather": 1}),
    ("rain?", {"weather": 1}),
    ("forecast", {"weather": 1}),
    ("temperature", {"weather": 1}),
]

# Brand in descriptive sentence = NOT action (must NOT fire)
SMART_BARE_INFO = [
    # ride brands in sentences
    ("uber is a great company", {}),
    ("uber is hiring", {}),
    ("uber drivers are nice", {}),
    ("lyft stock went up", {}),
    ("I work at uber", {}),
    ("my friend works at lyft", {}),
    ("uber vs lyft", {}),
    ("uber has a new feature", {}),
    ("lyft is cheaper than uber", {}),
    ("the uber app is slow", {}),
    ("uber just laid off people", {}),
    ("uber IPO was huge", {}),
    ("I interviewed at uber", {}),
    ("uber surge pricing is crazy", {}),
    ("lyft drivers make less", {}),

    # money brands in sentences
    ("venmo is down", {}),
    ("venmo has a new update", {}),
    ("is venmo safe", {}),
    ("cashapp vs venmo", {}),
    ("cashapp has fees", {}),
    ("zelle doesn't work", {}),
    ("venmo got hacked", {}),
    ("I don't trust cashapp", {}),
    ("paypal owns venmo", {}),
    ("does venmo charge fees", {}),
    ("venmo has social features", {}),
    ("cashapp bitcoin is interesting", {}),
    ("zelle is linked to my bank", {}),

    # food brands in sentences
    ("dominos is overrated", {}),
    ("chipotle gives me stomach issues", {}),
    ("doordash fees are crazy", {}),
    ("ubereats has a promo", {}),
    ("swiggy vs zomato", {}),
    ("dominos has a buy one get one", {}),
    ("I worked at chipotle", {}),
    ("mcdonald's is trash", {}),
    ("kfc is better in india", {}),
    ("pizza hut closed down", {}),
    ("doordash drivers don't get tips", {}),

    # music brands in sentences
    ("spotify has a new feature", {}),
    ("spotify raised prices", {}),
    ("apple music sounds better", {}),
    ("spotify wrapped was fun", {}),
    ("spotify vs apple music", {}),

    # video brands in sentences
    ("netflix raised their prices", {}),
    ("netflix is losing subscribers", {}),
    ("youtube has too many ads", {}),
    ("hulu has a good deal", {}),
    ("netflix original movies are mid", {}),

    # weather in figurative use
    ("the weather of opinion has shifted", {}),
    ("weathering the storm", {}),
    ("under the weather", {}),
    ("fair weather friend", {}),
]


# ═══════════════════════════════════════════════════════════════════════
# SKILL 4: NEGATION SPECTRUM
#
# Not all negation is the same:
#   "don't forget to X"  → DO X (emphasis, not negation)
#   "don't X"            → DON'T X (actual negation)
#   "cancel the X"       → DON'T X (cancellation)
#   "stop X"             → DON'T X (stop command)
#   "already X'd"        → DON'T X (already done)
# ═══════════════════════════════════════════════════════════════════════

# "don't forget to X" = DO X (the intent of X fires, alarm does NOT)
SMART_DONT_FORGET = [
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
    ("don't forget the reservation", {"reservation": 1}),
    ("don't forget the tickets", {"tickets": 1}),
    ("don't forget to uber", {"ride": 1}),
    ("don't forget to venmo {name}", {"money": 1}),
    ("don't forget to split the bill", {"money": 1}),
    ("don't forget the doctor appointment", {"health": 1}),
    ("don't forget your meds", {"health": 1}),
    ("don't forget to check the weather", {"weather": 1}),
    ("make sure to book the flight", {"travel": 1}),
    ("make sure to pay rent", {"bills": 1}),
    ("make sure to call {name}", {"contact": 1}),
    ("make sure to order food", {"food_order": 1}),
    ("remember to pay rent", {"bills": 1}),
    ("remember to book the flight", {"travel": 1}),
    ("remember to call mom", {"contact": 1}),
    ("remember to order dinner", {"food_order": 1}),
]

# "don't X" / "cancel X" = DON'T X (nothing fires)
SMART_ACTUAL_NEGATION = [
    # don't + action
    ("don't transfer the money", {}),
    ("don't transfer the rent yet", {}),
    ("don't send the money", {}),
    ("don't send it yet", {}),
    ("don't pay yet", {}),
    ("don't pay the bill yet", {}),
    ("don't venmo me", {}),
    ("don't cashapp me", {}),
    ("don't book the uber", {}),
    ("don't book the cab", {}),
    ("don't book the flight", {}),
    ("don't book the hotel", {}),
    ("don't book the table", {}),
    ("don't order food", {}),
    ("don't order pizza", {}),
    ("don't order anything", {}),
    ("don't call mom yet", {}),
    ("don't call him", {}),
    ("don't text her", {}),
    ("don't set the alarm", {}),
    ("don't set a reminder", {}),
    ("don't play music", {}),
    ("don't buy it", {}),
    ("I'm not paying for pizza", {}),
    ("I'm not going to call mom", {}),
    ("I won't be ordering food", {}),
    ("no need to book an uber", {}),
    ("no need to set an alarm", {}),
    ("I don't need a reminder", {}),
    ("I don't need an uber", {}),

    # cancel / stop
    ("cancel the uber", {}),
    ("cancel the cab", {}),
    ("cancel the order", {}),
    ("cancel the food order", {}),
    ("cancel the reservation", {}),
    ("cancel the flight", {}),
    ("cancel the alarm", {}),
    ("cancel the reminder", {}),
    ("cancel the reminder for {time}", {}),
    ("cancel the timer", {}),
    ("cancel the meeting", {}),
    ("cancel the appointment", {}),
    ("cancel my order", {}),
    ("cancel everything", {}),
    ("stop the alarm", {}),
    ("stop the timer", {}),
    ("stop the music", {}),
    ("never mind the uber", {}),
    ("never mind the order", {}),
    ("nvm the uber", {}),
    ("nvm the order", {}),
    ("nvm the payment", {}),
    ("scratch that", {}),
    ("actually don't order", {}),
    ("wait don't send it", {}),
    ("wait don't book it", {}),
    ("forget it", {}),
    ("forget about it", {}),
    ("hold off on that", {}),

    # already done
    ("already paid the rent", {}),
    ("already ordered food", {}),
    ("already booked the uber", {}),
    ("already set the alarm", {}),
    ("already called mom", {}),
    ("already booked the flight", {}),
    ("already made the reservation", {}),
    ("already sent the money", {}),
    ("already venmo'd you", {}),
    ("I already paid", {}),
    ("I already ordered", {}),
    ("I already booked it", {}),
    ("took care of it already", {}),
    ("done already", {}),
    ("handled it", {}),
]


# ═══════════════════════════════════════════════════════════════════════
# SKILL 5: PERSON OWNERSHIP
#
#   "she paid 200"    → nothing (third person gossip)
#   "pay her 200"     → money (YOUR action)
#   "he ordered pizza" → nothing (describing someone else)
#   "order pizza"     → food_order (YOUR action)
# ═══════════════════════════════════════════════════════════════════════

SMART_THIRD_PERSON = [
    # Third person = gossip (nothing fires)
    ("she paid {amount} for that bag", {}),
    ("he paid {amount} for dinner", {}),
    ("they paid {amount} for the tickets", {}),
    ("she ordered sushi last night", {}),
    ("he ordered an uber home", {}),
    ("they booked a flight to Tokyo", {}),
    ("she called the doctor", {}),
    ("he set an alarm for 6am", {}),
    ("she venmo'd him {amount}", {}),
    ("he booked a table at {place}", {}),
    ("they split the bill", {}),
    ("she ubered to the airport", {}),
    ("he texted mom earlier", {}),
    ("they watched {movie} last night", {}),
    ("she's on the phone", {}),
    ("he's cooking dinner", {}),
    ("they're watching tv", {}),
    ("she bought tickets to {movie}", {}),
    ("he's taking a cab", {}),
    ("she made a reservation at {place}", {}),

    # First/second person = YOUR action (fires)
    ("pay her {amount}", {"money": 1}),
    ("pay him {amount}", {"money": 1}),
    ("send her {amount}", {"money": 1}),
    ("send him {amount}", {"money": 1}),
    ("venmo her {amount}", {"money": 1}),
    ("venmo him {amount}", {"money": 1}),
    ("call her", {"contact": 1}),
    ("call him", {"contact": 1}),
    ("text her", {"contact": 1}),
    ("text him", {"contact": 1}),
    ("order for them", {"food_order": 1}),
    ("book it for them", {"reservation": 1}),
    ("get him an uber", {"ride": 1}),
    ("send her the location", {"maps": 1}),
]


# ═══════════════════════════════════════════════════════════════════════
# SKILL 6: BILLS vs P2P MONEY
#
#   "pay rent"            → bills (service payment)
#   "venmo me for rent"   → money + bills (P2P + bill context)
#   "remind me to pay rent" → alarm + bills (NOT money)
#   "split rent with me"  → money (P2P splitting)
# ═══════════════════════════════════════════════════════════════════════

SMART_BILLS = [
    # Pure bills (money must NOT fire)
    ("pay rent", {"bills": 1}),
    ("pay rent {day}", {"bills": 1}),
    ("pay the rent", {"bills": 1}),
    ("pay the electricity bill", {"bills": 1}),
    ("pay the wifi bill", {"bills": 1}),
    ("pay the internet bill", {"bills": 1}),
    ("pay the water bill", {"bills": 1}),
    ("pay the gas bill", {"bills": 1}),
    ("pay the phone bill", {"bills": 1}),
    ("pay insurance", {"bills": 1}),
    ("pay the mortgage", {"bills": 1}),
    ("pay utilities", {"bills": 1}),
    ("pay my subscription", {"bills": 1}),
    ("rent is due", {"bills": 1}),
    ("rent is due {day}", {"bills": 1}),
    ("bill due {day}", {"bills": 1}),
    ("gotta pay rent", {"bills": 1}),
    ("need to pay rent", {"bills": 1}),
    ("time to pay rent", {"bills": 1}),

    # Bills + alarm (money must NOT fire)
    ("remind me to pay rent {day}", {"alarm": 1, "bills": 1}),
    ("remind me to pay the electricity bill", {"alarm": 1, "bills": 1}),
    ("remind me to pay the wifi bill", {"alarm": 1, "bills": 1}),
    ("reminder to pay utilities", {"alarm": 1, "bills": 1}),
    ("remind me about rent on {day}", {"alarm": 1, "bills": 1}),
    ("set reminder for rent {day}", {"alarm": 1, "bills": 1}),
    ("ping me about the bill {day}", {"alarm": 1, "bills": 1}),

    # Bills + money (P2P involved)
    ("rent due {day}, transfer {amount} to landlord", {"bills": 1, "money": 1}),
    ("send {amount} to landlord for rent", {"bills": 1, "money": 1}),
    ("venmo landlord {amount} for rent", {"bills": 1, "money": 1}),
    ("transfer rent {amount}", {"bills": 1, "money": 1}),
    ("send rent money {amount}", {"bills": 1, "money": 1}),
    ("split rent with me", {"money": 1}),
    ("split the rent", {"money": 1}),
    ("your half of rent is {amount}", {"money": 1}),
]


# ═══════════════════════════════════════════════════════════════════════
# SKILL 7: ACTION MODIFIERS
#
# "wanna", "should we", "let's", "can you" = actionable requests
# These are how real people talk — questions that ARE commands.
# ═══════════════════════════════════════════════════════════════════════

SMART_ACTION_MODIFIERS = [
    ("wanna uber", {"ride": 1}),
    ("wanna uber there", {"ride": 1}),
    ("wanna order food", {"food_order": 1}),
    ("wanna order pizza", {"food_order": 1}),
    ("wanna get sushi", {"food_order": 1}),
    ("wanna split it", {"money": 1}),
    ("wanna call {name}", {"contact": 1}),
    ("wanna watch {movie}", {"video": 1}),
    ("wanna listen to music", {"music": 1}),
    ("wanna book a flight", {"travel": 1}),
    ("wanna book a table", {"reservation": 1}),
    ("should we uber", {"ride": 1}),
    ("should we order food", {"food_order": 1}),
    ("should we order pizza", {"food_order": 1}),
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
    ("let's order pizza", {"food_order": 1}),
    ("let's split it", {"money": 1}),
    ("let's call {name}", {"contact": 1}),
    ("let's book a cab", {"ride": 1}),
    ("let's watch {movie}", {"video": 1}),
    ("let's book the flight", {"travel": 1}),
    ("let's get food", {"food_order": 1}),
    ("let's get an uber", {"ride": 1}),
    ("let's get a cab", {"ride": 1}),
]


# ═══════════════════════════════════════════════════════════════════════
# BONUS: PRESENT ACTIVITY — describing what you're doing ≠ command
# ═══════════════════════════════════════════════════════════════════════

SMART_PRESENT_ACTIVITY = [
    ("I'm watching tv", {}),
    ("I'm watching {movie}", {}),
    ("I'm cooking dinner", {}),
    ("I'm eating pizza", {}),
    ("I'm on the phone", {}),
    ("I'm driving", {}),
    ("I'm at the gym", {}),
    ("I'm listening to music", {}),
    ("I'm studying", {}),
    ("I'm working", {}),
    ("I'm in a meeting", {}),
    ("I'm shopping", {}),
    ("I'm at the doctor", {}),
    ("I'm on my way", {}),
    ("watching tv rn", {}),
    ("eating rn", {}),
    ("driving rn", {}),
    ("cooking rn", {}),
    ("movie night, watching {movie}", {"video": 1}),
    ("binge watching {show}", {"video": 1}),
    ("about to watch {movie}", {"video": 1}),
    ("putting on {show}", {"video": 1}),
    ("gonna binge {show}", {"video": 1}),
    ("catching up on {show}", {"video": 1}),
    ("netflix and chill", {"video": 1}),
]


# ═══════════════════════════════════════════════════════════════════════
# BONUS: CONTACT BOOST — all the ways people say "call someone"
# ═══════════════════════════════════════════════════════════════════════

SMART_CONTACT = [
    ("call {name}", {"contact": 1}),
    ("call {name} please", {"contact": 1}),
    ("call {name} rn", {"contact": 1}),
    ("call {name} real quick", {"contact": 1}),
    ("call {name} back", {"contact": 1}),
    ("call {name} asap", {"contact": 1}),
    ("call {name} later", {"contact": 1}),
    ("call mom", {"contact": 1}),
    ("call dad", {"contact": 1}),
    ("call home", {"contact": 1}),
    ("call the doctor", {"contact": 1}),
    ("call the dentist", {"contact": 1}),
    ("call the restaurant", {"contact": 1}),
    ("give {name} a call", {"contact": 1}),
    ("give {name} a ring", {"contact": 1}),
    ("hit up {name}", {"contact": 1}),
    ("ring {name}", {"contact": 1}),
    ("phone {name}", {"contact": 1}),
    ("text {name}", {"contact": 1}),
    ("text {name} back", {"contact": 1}),
    ("text mom", {"contact": 1}),
    ("text dad", {"contact": 1}),
    ("message {name}", {"contact": 1}),
    ("dm {name}", {"contact": 1}),
    ("ping {name}", {"contact": 1}),
    ("send {name} a text", {"contact": 1}),
    ("shoot {name} a text", {"contact": 1}),
    ("reach out to {name}", {"contact": 1}),
    ("let {name} know", {"contact": 1}),
    ("need to call {name}", {"contact": 1}),
    ("gotta call {name}", {"contact": 1}),
    ("I'll call {name}", {"contact": 1}),
    ("let me call {name}", {"contact": 1}),
    ("remind me to call {name}", {"contact": 1, "alarm": 1}),
    ("remind me to call mom {day}", {"contact": 1, "alarm": 1}),
    ("remind me to text {name}", {"contact": 1, "alarm": 1}),
]


# ═══════════════════════════════════════════════════════════════════════
# Aggregate all smart model banks
# ═══════════════════════════════════════════════════════════════════════

ALL_SMART_BANKS = {
    # Skill 1: Time ownership
    "smart_time_ride":          SMART_TIME_RIDE,
    "smart_time_travel":        SMART_TIME_TRAVEL,
    "smart_time_video":         SMART_TIME_VIDEO,
    "smart_time_food":          SMART_TIME_FOOD,
    "smart_time_reservation":   SMART_TIME_RESERVATION,
    "smart_time_health":        SMART_TIME_HEALTH,
    "smart_time_calendar":      SMART_TIME_CALENDAR,
    "smart_time_alarm":         SMART_TIME_ALARM,
    "smart_time_multi_alarm":   SMART_TIME_MULTI_WITH_ALARM,

    # Skill 2: Place ownership
    "smart_place_venue":        SMART_PLACE_VENUE,
    "smart_place_navigation":   SMART_PLACE_NAVIGATION,
    "smart_ride_not_maps":      SMART_RIDE_NOT_MAPS,
    "smart_ride_with_maps":     SMART_RIDE_WITH_MAPS,

    # Skill 3: Bare word intelligence
    "smart_bare_action":        SMART_BARE_ACTION,
    "smart_bare_info":          SMART_BARE_INFO,

    # Skill 4: Negation
    "smart_dont_forget":        SMART_DONT_FORGET,
    "smart_actual_negation":    SMART_ACTUAL_NEGATION,

    # Skill 5: Person ownership
    "smart_third_person":       SMART_THIRD_PERSON,

    # Skill 6: Bills vs money
    "smart_bills":              SMART_BILLS,

    # Skill 7: Action modifiers
    "smart_action_modifiers":   SMART_ACTION_MODIFIERS,

    # Bonus
    "smart_present_activity":   SMART_PRESENT_ACTIVITY,
    "smart_contact":            SMART_CONTACT,
}

if __name__ == "__main__":
    total = 0
    for name, bank in ALL_SMART_BANKS.items():
        if bank and isinstance(bank[0], tuple) and len(bank[0]) == 2:
            positives = sum(1 for entry in bank if entry[1])
            negatives = sum(1 for entry in bank if not entry[1])
        else:
            positives = len(bank)
            negatives = 0
        print(f"  {name:<30} {len(bank):>3} templates ({positives} pos, {negatives} neg)")
        total += len(bank)
    print(f"\n  Total: {total} smart model templates")
