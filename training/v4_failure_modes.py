"""
v4 failure-mode templates — the specific gaps the seed-suite baseline exposed.

Baseline (v3): 43/82 on the seed suite (52.4%).
Critical failure modes:
  - negation     (25%) — "I'm not paying for X" still fired money
  - past_tense   (43%) — "paid rent already" still fired bills
  - multi_intent (0%)  — compound messages collapse to dominant intent
  - code_mixed   (25%) — Hindi-English Roman script barely works
  - contact      (0% recall) — "call dad" → fired `task` instead of `contact`
  - edge_pronoun (33%) — pronouns confused the contact intent
  - ambiguous overfire — "set it for 6" fires alarm at 0.95 (should be silent)
  - dont_forget — "don't forget to call mom" suppressed by "don't"

Each bank below is a list of templates (or (template, label_flags) tuples)
ready for fill() + augment() in generate_data.py.
"""

# ───────────────────────────────────────────────────────────────────────
#  STRONG NEGATIONS — must NOT fire any intent
# ───────────────────────────────────────────────────────────────────────
V4_NEGATION = [
    # money / bills negation
    "I'm not paying for {food}",
    "I'm not paying you back",
    "won't venmo you anything",
    "stopped using venmo months ago",
    "don't transfer the {amount} yet",
    "no I won't pay for that",
    "I'm not sending any money",
    "not gonna venmo you for it",
    "I won't be paying that bill",
    "no plans to pay rent yet",
    "skip rent this month",
    "won't pay the {bill_kind} bill",
    "don't pay him back",
    "I'm not splitting that with you",
    "no need to send money",
    "not going to chip in",

    # alarm / reminder negation
    "don't remind me to {task}",
    "don't remind me about it",
    "cancel the reminder for {time}",
    "forget the alarm",
    "no need to remind me",
    "I won't set an alarm",
    "stop setting alarms",
    "don't ping me about it",
    "no reminders please",
    "skip the reminder",
    "delete the alarm",
    "remove the {time} reminder",
    "I don't need a reminder",

    # contact negation
    "I won't call him today",
    "I'm not calling {addressee}",
    "stop texting them",
    "don't call me back",
    "I'm not reaching out",
    "no need to ring {addressee}",
    "won't be texting {addressee}",
    "don't message {addressee}",

    # ride negation
    "no need to book an uber",
    "won't take an uber",
    "not ubering home",
    "skip the lyft",
    "no uber needed",
    "I'll walk no need for a ride",
    "not booking any ride",

    # food negation
    "I'm not ordering takeout",
    "don't doordash",
    "skip the food order",
    "no doordash tonight",
    "not getting {food} today",
    "no need for delivery",
    "I'm cooking not ordering",

    # music / video / tickets / reservation negation
    "won't be playing music",
    "no music tonight",
    "not watching tv tonight",
    "skip the movie",
    "don't book any tickets",
    "no tickets needed",
    "cancel our reservation at {place}",
    "no need to reserve a table",

    # health negation
    "forget the doctor appointment",
    "cancel the dentist",
    "no need for a doctor",
    "skip the appointment",
    "I'm not seeing a doctor",

    # travel negation
    "not flying anywhere",
    "canceled the flight",
    "no travel plans",
    "scrapped the {city} trip",
    "won't be going to {city}",

    # calendar negation
    "no meeting today",
    "cancel the lunch",
    "skip the event",
    "calling off the meeting",

    # weather negation
    "don't tell me the weather",

    # shopping negation
    "not buying anything today",
    "don't add to cart",
    "skip the order",
    "not shopping today",
    "won't order the {item}",
]


# ───────────────────────────────────────────────────────────────────────
#  PAST TENSE — completed actions; must NOT fire
# ───────────────────────────────────────────────────────────────────────
V4_PAST_TENSE = [
    # money past
    "paid {amount} yesterday",
    "venmoed {addressee} yesterday",
    "already sent the money",
    "transferred rent last friday",
    "I paid him back last week",
    "covered the bill last night",
    "settled up with {addressee} earlier",
    "spent {amount} on groceries today",
    "transferred {amount} to mom",
    "I just paid him",
    "already venmoed you",
    "split the bill last night already",

    # alarm past
    "set my alarm last night",
    "the reminder went off this morning",
    "alarm woke me up at {time}",
    "set a reminder yesterday",
    "the alarm just rang",
    "got the {time} alarm earlier",

    # contact past
    "called dad already",
    "texted mom yesterday",
    "messaged sarah last week",
    "facetimed alex on sunday",
    "called {addressee} earlier today",
    "texted dad this morning",
    "rang the office an hour ago",
    "spoke to {addressee} last night",
    "we just got off the phone",

    # ride past
    "ubered home last night",
    "took an uber yesterday",
    "lyft dropped me off earlier",
    "got off the uber",
    "rode here in a lyft",

    # food past
    "ordered sushi already",
    "had doordash for lunch",
    "we ate at sweetgreen yesterday",
    "got chipotle on the way",
    "the food just arrived",
    "ate pizza last night",

    # music past
    "listened to {artist} this morning",
    "played that song earlier",
    "had spotify on all day",

    # video past
    "watched stranger things last week",
    "saw {movie} on opening night",
    "binged the show yesterday",
    "watched it last night",
    "finished the season already",

    # tickets past
    "bought tickets last week",
    "already got the tickets",
    "got my taylor swift tickets",
    "we have the tickets",

    # travel past
    "flew to {city} last month",
    "got back from {city} yesterday",
    "visited {city} on vacation",
    "I flew to {city} for work",
    "vacation in {city} was great",

    # health past
    "saw the doctor on monday",
    "had my appointment yesterday",
    "the dentist was last week",
    "got my check-up done",

    # bills past
    "paid rent friday",
    "already sent the rent",
    "settled the electric bill",
    "the {bill_kind} bill was paid yesterday",
    "rent's already taken care of",

    # reservation past
    "booked the table earlier",
    "we had reservations last night",
    "ate at our reserved spot yesterday",

    # shopping past
    "bought {item} yesterday",
    "ordered the {item} already",
    "got my amazon delivery",
    "the package arrived",

    # generic completion phrases
    "took care of it earlier",
    "handled it already",
    "all done with that",
    "wrapped up the call",
    "got it sorted last night",
]


# ───────────────────────────────────────────────────────────────────────
#  MULTI-INTENT — compound messages, must fire 2-3 intents
# ───────────────────────────────────────────────────────────────────────
V4_MULTI_INTENT = [
    # ride + maps + (alarm if early)
    ("uber to {airport} at 5am tomorrow",          {"ride": 1, "maps": 1, "alarm": 1}),
    ("lyft to {place} at {time}",                  {"ride": 1, "maps": 1}),
    ("get me an uber to {airport}",                {"ride": 1, "maps": 1}),
    ("ride from home to {place}",                  {"ride": 1, "maps": 1}),
    ("uber to {place} please",                     {"ride": 1, "maps": 1}),
    ("book a lyft to {airport} for {time}",        {"ride": 1, "maps": 1, "alarm": 1}),
    ("uber needed to {place} at {time} tomorrow",  {"ride": 1, "maps": 1, "alarm": 1}),

    # alarm + bills
    ("remind me to pay rent {day}",                {"alarm": 1, "bills": 1}),
    ("ping me to pay the {bill_kind} bill",        {"alarm": 1, "bills": 1}),
    ("remind me {day} that rent is due",           {"alarm": 1, "bills": 1}),
    ("alarm to pay {bill_kind} {date}",            {"alarm": 1, "bills": 1}),

    # alarm + contact
    ("remind me to call {addressee} at {time}",        {"alarm": 1, "contact": 1}),
    ("ping me to text {addressee} later",              {"alarm": 1, "contact": 1}),
    ("set a reminder to call {addressee}",             {"alarm": 1, "contact": 1}),
    ("remind me {day} to facetime {addressee}",        {"alarm": 1, "contact": 1}),
    ("don't let me forget to call mom {day}",          {"alarm": 1, "contact": 1}),

    # food + alarm
    ("doordash {food} at {time} and remind me to take meds at {time2}",
                                                   {"food_order": 1, "alarm": 1}),
    ("order {food} at {time} and remind me to eat",
                                                   {"food_order": 1, "alarm": 1}),

    # calendar + maps + contact
    ("lunch with {addressee} at {place} {day} at {time}",
                                                   {"calendar": 1, "maps": 1, "contact": 1}),
    ("dinner with {addressee} at {place} {time}",
                                                   {"calendar": 1, "maps": 1, "contact": 1}),
    ("coffee with {addressee} at {place}",         {"calendar": 1, "maps": 1, "contact": 1}),
    ("meeting with {addressee} at {place} {day}",  {"calendar": 1, "maps": 1, "contact": 1}),

    # calendar + reservation
    ("book a table at {place} for {qty} {day} at {time}",
                                                   {"calendar": 1, "reservation": 1}),
    ("reserve {place} for {time} {day}",           {"calendar": 1, "reservation": 1}),
    ("dinner reservation at {place} for {time}",   {"calendar": 1, "reservation": 1}),

    # video + tickets + maps
    ("{movie} at amc tonight at {time}",           {"video": 1, "tickets": 1, "maps": 1}),
    ("{movie} at the imax tonight at {time}",      {"video": 1, "tickets": 1, "maps": 1}),
    ("see {movie} at the theater {day}",           {"video": 1, "tickets": 1}),
    ("buy tickets for {movie}",                    {"tickets": 1, "video": 1}),
    ("{movie} showing at {time} - get tickets",    {"tickets": 1, "video": 1}),

    # bills + money (the compound that's broken in v3)
    ("rent due {day} transfer {amount} to landlord",   {"bills": 1, "money": 1}),
    ("rent is due {day} gonna transfer {amount}",      {"bills": 1, "money": 1}),
    ("electricity bill is here paying {amount} via venmo",
                                                       {"bills": 1, "money": 1}),
    ("transfer the gym membership {amount} today",     {"bills": 1, "money": 1}),
    ("internet bill {amount} due tomorrow will send via paypal",
                                                       {"bills": 1, "money": 1}),
    ("send {amount} for the {bill_kind} bill",         {"bills": 1, "money": 1}),
    ("paying {amount} for {bill_kind} now",            {"bills": 1, "money": 1}),
    ("transfer {amount} for rent friday",              {"bills": 1, "money": 1}),

    # ride + food
    ("just landed at {airport} uber me home and order {food} on the way",
                                                   {"ride": 1, "maps": 1, "food_order": 1}),
    ("grab an uber and pick up {food}",            {"ride": 1, "food_order": 1}),

    # health + calendar
    ("book me a doctor appointment for {day}",     {"health": 1, "calendar": 1}),
    ("schedule the dentist for {day}",             {"health": 1, "calendar": 1}),
    ("dentist {day} at {time}",                    {"health": 1, "calendar": 1}),
    ("doctor visit {day} at {time}",               {"health": 1, "calendar": 1}),

    # travel + calendar
    ("flying to {city} on {date}",                 {"travel": 1, "calendar": 1}),
    ("flight to {city} {date}",                    {"travel": 1}),

    # contact + alarm (recurring call)
    ("ring mom every {day}",                       {"contact": 1, "alarm": 1}),
    ("remind me to call dad every {day}",          {"contact": 1, "alarm": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  CODE-MIXED — Hindi-English Roman script (India market)
# ───────────────────────────────────────────────────────────────────────
V4_CODE_MIXED = [
    # money
    ("yaar venmo me {amount}",                    {"money": 1}),
    ("bhai {amount} bhej de venmo pe",            {"money": 1}),
    ("paytm pe {amount} send kar",                {"money": 1}),
    ("upi pe {amount} transfer kar de",           {"money": 1}),
    ("phonepe se {amount} bhej",                  {"money": 1}),
    ("gpay pe {amount} bhejna hai",               {"money": 1}),
    ("yaar mujhe {amount} chahiye",               {"money": 1}),

    # contact
    ("mom ko phone karna hai",                    {"contact": 1}),
    ("dad ko call kar le",                        {"contact": 1}),
    ("{addressee} ko message kar do",             {"contact": 1}),
    ("bhai ko whatsapp kar",                      {"contact": 1}),
    ("papa ko phone laga",                        {"contact": 1}),
    ("mummy ko call karo",                        {"contact": 1}),

    # ride + maps + alarm
    ("kal subah {time} baje uber book kar dena to {airport}",
                                                  {"ride": 1, "maps": 1, "alarm": 1}),
    ("uber chahiye to {place}",                   {"ride": 1, "maps": 1}),
    ("ola book kar do airport ke liye",           {"ride": 1, "maps": 1}),
    ("rapido se {place} tak chalna hai",          {"ride": 1, "maps": 1}),

    # alarm
    ("mujhe yaad dilana {time} baje",             {"alarm": 1}),
    ("kal subah jagana {time} pe",                {"alarm": 1}),
    ("alarm laga do {time} ke liye",              {"alarm": 1}),
    ("yaad dilana yaar",                          {"alarm": 1}),

    # food
    ("shaam ko {food} mangwana doordash se",      {"food_order": 1}),
    ("zomato pe pizza order kar de",              {"food_order": 1}),
    ("swiggy se khana mangao",                    {"food_order": 1}),
    ("biryani order karwa",                       {"food_order": 1}),

    # travel
    ("weekend pe {city} jaane ka plan",           {"travel": 1}),
    ("{city} ki flight book karwa",               {"travel": 1}),
    ("goa trip plan kar",                         {"travel": 1}),

    # bills + money
    ("rent ka transfer kar do {day} tak",         {"bills": 1, "money": 1}),
    ("bijli ka bill pay karna hai",               {"bills": 1}),
    ("EMI bharna hai is mahine",                  {"bills": 1}),

    # health + calendar
    ("doctor ka appointment book karwa do {day} ke liye",
                                                  {"health": 1, "calendar": 1}),
    ("dentist ke paas jana hai {day}",            {"health": 1, "calendar": 1}),

    # music
    ("spotify pe {artist} chala do",              {"music": 1}),
    ("gaana laga de",                             {"music": 1}),
    ("kuch achha play kar",                       {"music": 1}),

    # shopping
    ("amazon pe {item} order karwa do",           {"shopping": 1}),
    ("flipkart se {item} order karna",            {"shopping": 1}),

    # weather
    ("kal {city} mein baarish hogi kya",          {"weather": 1}),
    ("aaj ka mausam kaisa hai",                   {"weather": 1}),

    # video
    ("netflix pe show dekhna hai",                {"video": 1}),
    ("hotstar pe match dekho",                    {"video": 1}),

    # task
    ("yaad rakhna {task}",                        {"task": 1}),

    # negative chitchat (code-mixed) — must NOT fire
    ("paani lao please",                          {}),
    ("kya haal hai",                              {}),
    ("achha bhai",                                {}),
    ("theek hai bhai",                            {}),
    ("kuch nahi yaar",                            {}),
    ("haan haan",                                 {}),
    ("nahi yaar",                                 {}),
]


# ───────────────────────────────────────────────────────────────────────
#  CONTACT INTENT FIX — the v3 contact head has 0% recall, every "call X"
#  fired `task`. Drowning the model with explicit contact examples.
# ───────────────────────────────────────────────────────────────────────
V4_CONTACT_FIX = [
    # call + name
    "call {addressee}",
    "call {addressee} please",
    "call {addressee} now",
    "call {addressee} back",
    "give {addressee} a call",
    "give {addressee} a ring",
    "ring {addressee}",
    "ring {addressee} up",
    "phone {addressee}",
    "dial {addressee}",
    "calling {addressee}",
    "calling {addressee} now",
    "I need to call {addressee}",
    "gotta call {addressee}",
    "let me call {addressee}",
    "let me give {addressee} a quick call",
    "get {addressee} on the phone",
    "put {addressee} on the line",

    # text + name
    "text {addressee}",
    "text {addressee} please",
    "text {addressee} back",
    "shoot {addressee} a text",
    "drop {addressee} a message",
    "send {addressee} a text",
    "message {addressee}",
    "message {addressee} now",
    "messaging {addressee}",
    "texting {addressee}",
    "sms {addressee}",
    "send a text to {addressee}",

    # facetime / video
    "facetime {addressee}",
    "video call {addressee}",
    "ft {addressee}",
    "facetiming {addressee}",
    "let's facetime {addressee}",

    # whatsapp / imessage / dm
    "whatsapp {addressee}",
    "imessage {addressee}",
    "dm {addressee}",
    "whatsapp dad",
    "message {addressee} on whatsapp",

    # specific family / common names
    "call dad",
    "call mom",
    "call mum",
    "call dad please",
    "call mom now",
    "call dad back",
    "text dad",
    "text mom",
    "facetime mom",
    "facetime dad",
    "ring dad",
    "ring mom",
    "phone mom",
    "give mom a call",
    "give dad a call",
    "call sarah",
    "text alex",
    "call jordan",
    "text priya",
    "ring my brother",
    "call my sister",
    "text my wife",
    "call my husband",
    "facetime grandma",
    "call grandma",
    "call uncle",
    "text aunt",
    "call my boss",
    "text my friend",
    "phone my dentist",

    # pronoun forms — still contact intent (slot resolver handles "who?")
    "call him",
    "call her",
    "call them",
    "text him",
    "text her",
    "text them",
    "message him",
    "message her",
    "message them",
    "facetime him",
    "facetime her",
    "facetime them",
    "ring him",
    "ring her",
    "phone him",
    "phone her",

    # imperatives that imply contact
    "give them a ring",
    "give them a call",
    "shoot them a text",
    "shoot her a message",
    "drop him a line",

    # multi-recipient
    "need to call dad and mom",
    "call dad and mom and uncle",
    "text both my parents",
    "message everyone in the group",
    "call the whole family",
]


# ───────────────────────────────────────────────────────────────────────
#  "DON'T FORGET" — negation marker but action SHOULD fire
# ───────────────────────────────────────────────────────────────────────
V4_DONT_FORGET = [
    ("don't forget to call {addressee}",            {"contact": 1, "alarm": 1}),
    ("don't forget to text {addressee} {day}",      {"contact": 1, "alarm": 1}),
    ("don't forget to pay rent {day}",              {"bills": 1, "alarm": 1}),
    ("don't forget the dentist appointment",        {"health": 1, "alarm": 1}),
    ("don't forget to book the flight for {date}",  {"travel": 1, "alarm": 1}),
    ("don't forget to grab {item} on the way",      {"shopping": 1, "alarm": 1}),
    ("don't forget to order {food} at {time}",      {"food_order": 1, "alarm": 1}),
    ("don't forget the meeting at {time}",          {"calendar": 1, "alarm": 1}),
    ("don't forget our reservation at {place}",     {"reservation": 1, "alarm": 1}),
    ("don't forget to set an alarm for {time}",     {"alarm": 1}),
    ("remember to pay the {bill_kind} bill {day}",  {"bills": 1, "alarm": 1}),
    ("remember to call {addressee} {day}",          {"contact": 1, "alarm": 1}),
    ("make sure to text {addressee} later",         {"contact": 1, "alarm": 1}),
    ("make sure you book the {city} flight",       {"travel": 1, "alarm": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  AMBIGUOUS "QUIET" — model fires alarm/money at 0.95 on these; should be silent
# ───────────────────────────────────────────────────────────────────────
V4_AMBIGUOUS_QUIET = [
    "set it for {time}",
    "do it tomorrow",
    "remind me about it",
    "send {amount}",
    "play that thing",
    "check on her",
    "check on him",
    "call back",
    "text back",
    "let me know",
    "go ahead",
    "do whatever",
    "send it",
    "play it again",
    "watch it later",
    "order it",
    "book it",
    "remind me",
    "set a thing",
    "make a thing",
    "do that",
    "remember it",
    "the appointment thing",
    "that thing tomorrow",
    "the meeting",
    "the bill",
    "ok do it",
    "fine just do it",
]


# ───────────────────────────────────────────────────────────────────────
#  QUERIES (information requests) — must NOT fire actions
# ───────────────────────────────────────────────────────────────────────
V4_QUERY_NOT_ACTION = [
    "how much did I spend on food this month",
    "how much did I send {addressee} last week",
    "what's my balance",
    "where is my {item}",
    "did I pay rent",
    "what time is my flight",
    "when is the meeting",
    "who called me",
    "what was that song",
    "what movie did I watch",
    "how many times did I call mom",
    "where did I go yesterday",
    "what did I order from doordash",
    "did I set the alarm",
    "is the rent paid",
    "have I paid the {bill_kind} bill",
    "what's on my calendar today",
    "do I have any reminders",
    "any meetings scheduled",
]
