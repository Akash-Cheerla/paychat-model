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


# ═══════════════════════════════════════════════════════════════════════
#  v4.1 BANKS — teammate feedback from real-world chat testing
# ═══════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────
#  WORD AMOUNTS — "five dollars", "twenty bucks" (v4 only trained on
#  numeric amounts like "$20", "50 bucks"; fails on written-out numbers)
# ───────────────────────────────────────────────────────────────────────
V41_WORD_AMOUNTS = [
    # send/pay with word amounts
    ("venmo me five dollars",                           {"money": 1}),
    ("pay me ten bucks",                                {"money": 1}),
    ("send twenty dollars to {addressee}",              {"money": 1}),
    ("you owe me fifteen dollars",                      {"money": 1}),
    ("cashapp me fifty bucks for dinner",               {"money": 1}),
    ("transfer thirty dollars to {addressee}",          {"money": 1}),
    ("pay {addressee} twenty five dollars",             {"money": 1}),
    ("zelle me forty bucks",                            {"money": 1}),
    ("send me a hundred dollars",                       {"money": 1}),
    ("I owe you seventy five bucks",                    {"money": 1}),
    ("give me five bucks for coffee",                   {"money": 1}),
    ("you owe me two hundred dollars",                  {"money": 1}),
    ("can you send ten dollars",                        {"money": 1}),
    ("split it three ways thats twenty each",           {"money": 1}),
    ("here's fifty for the groceries",                  {"money": 1}),
    ("pay back the thirty dollars you borrowed",        {"money": 1}),
    ("I need five dollars for the bus",                 {"money": 1}),
    ("lend me sixty bucks till friday",                 {"money": 1}),
    ("chip in fifteen each",                            {"money": 1}),
    ("your share is forty five dollars",                {"money": 1}),
    # word amounts in bill context
    ("rent is twelve hundred this month",               {"bills": 1}),
    ("electric bill is eighty dollars",                 {"bills": 1}),
    ("pay two hundred for the {bill_kind} bill",        {"bills": 1, "money": 1}),
    # word amounts that should NOT fire (past tense)
    ("paid him fifty bucks yesterday",                  {}),
    ("already sent twenty dollars",                     {}),
    ("gave her ten bucks last week",                    {}),
]


# ───────────────────────────────────────────────────────────────────────
#  PRESENT-PERFECT / PENDING — "haven't paid yet", "still need to pay"
#  These are ACTIONABLE (unlike past tense which is completed).
#  v4 incorrectly treated these as past tense negatives.
# ───────────────────────────────────────────────────────────────────────
V41_PRESENT_PERFECT_ACTIONABLE = [
    # bills — haven't paid yet = actionable
    ("I still haven't paid my {bill_kind} bill",        {"bills": 1}),
    ("haven't paid rent yet",                           {"bills": 1}),
    ("I need to pay the {bill_kind} bill",              {"bills": 1}),
    ("still need to pay rent",                          {"bills": 1}),
    ("gotta pay the {bill_kind} bill",                  {"bills": 1}),
    ("rent is overdue I should pay",                    {"bills": 1}),
    ("forgot to pay {bill_kind}",                       {"bills": 1, "alarm": 1}),
    ("haven't sent the rent yet",                       {"bills": 1, "money": 1}),
    ("still owe {amount} for {bill_kind}",              {"bills": 1, "money": 1}),
    ("the electricity bill is due and I haven't paid",  {"bills": 1}),
    ("my {bill_kind} bill is overdue",                  {"bills": 1}),
    # money — haven't sent yet = actionable
    ("I still haven't paid {addressee} back",           {"money": 1}),
    ("haven't venmoed {addressee} yet",                 {"money": 1}),
    ("still need to send {addressee} the money",        {"money": 1}),
    ("I should pay {addressee} back",                   {"money": 1}),
    ("forgot to venmo {addressee}",                     {"money": 1}),
    # contact — haven't called yet = actionable
    ("I still haven't called {addressee}",              {"contact": 1}),
    ("haven't texted mom back yet",                     {"contact": 1}),
    ("I need to call {addressee} back",                 {"contact": 1}),
    ("still need to call dad",                          {"contact": 1}),
    # tasks — haven't done yet = actionable
    ("I still haven't finished {task}",                 {"task": 1}),
    ("still need to {task}",                            {"task": 1}),
    ("haven't {task} yet",                              {"task": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  OTP / SPAM / NOTIFICATION NOISE — must NOT fire any intent.
#  Real chat contains forwarded OTPs, app notifications, random noise.
# ───────────────────────────────────────────────────────────────────────
V41_NOISE_NEGATIVES = [
    # OTP messages
    "Your OTP is 482910. Do not share with anyone.",
    "123456 is your verification code for Amazon",
    "Your one-time password is 789012",
    "Google verification code: 654321",
    "PayPal: Your security code is 112233. It expires in 10 minutes.",
    "Uber code: 5544",
    "Your login code is 998877",
    "Enter 223344 to verify your account",
    "OTP for transaction: 556677",
    "2FA code: 334455",

    # notification-style noise
    "Your package has been delivered",
    "Your order has been shipped",
    "Battery low 10%",
    "Storage almost full",
    "Update available for 3 apps",
    "Screenshot saved",
    "WiFi connected to Home",
    "Do not disturb is on",
    "Bluetooth connected",
    "Low data warning: 90% used",

    # spam / marketing
    "FLASH SALE 50% off everything today only!!",
    "Congratulations you've won a $1000 gift card!",
    "Click here to claim your prize",
    "Limited time offer free shipping on all orders",
    "You have been selected for an exclusive deal",
    "Earn cash back on every purchase sign up now",
    "Unsubscribe from this list",

    # random conversation noise
    "lmaooo",
    "bruh",
    "no way",
    "fr fr",
    "that's crazy",
    "wow",
    "lol what",
    "hahaha",
    "ok",
    "k",
    "bet",
    "nah",
    "wdym",
    "idk",
    "same",
    "real",
    "ong",
    "deadass",
    "nvm",
    "mb",
    "smh",
    "istg",
    "💀",
    "😭",
    "🔥",
    "👍",
    "aight",
    "yo",
    "sup",
    "nm wbu",
    "nothing much",
    "just chilling",
    "im bored",
    "wanna hang",
    "what time is it",
    "where are you",
    "im coming",
    "almost there",
    "running late",
    "be there in 5",
    "one sec",
    "hold on",
    "my bad",
    "sorry wrong chat",
    "ignore that",
    "that was for someone else",
]


# ───────────────────────────────────────────────────────────────────────
#  SARCASM / IMPOSSIBLE / UNREALISTIC — must NOT fire real actions.
#  "book a flight to Mars" is not a valid travel intent.
# ───────────────────────────────────────────────────────────────────────
V41_SARCASM_IMPOSSIBLE = [
    "book a flight to Mars",
    "uber to the moon",
    "fly me to Jupiter",
    "reserve a table on the sun",
    "order food from Antarctica",
    "book me a hotel on Mars",
    "flight to Narnia next week",
    "uber to Hogwarts please",
    "directions to the end of the rainbow",
    "navigate to Atlantis",
    "call Santa Claus",
    "text the tooth fairy",
    "venmo God 100 bucks",
    "pay the universe back",
    "set an alarm for the year 3000",
    "remind me in a million years",
    "sure let me just fly to the moon real quick",
    "yeah I'll just teleport there",
    "oh yeah let me just print money",
    "lol yeah let me book a spaceship",
    "I'll just time travel back",
    "right because I can totally fly",
    "yeah and pigs can fly",
    "sure I'll just swim across the ocean",
]


# ───────────────────────────────────────────────────────────────────────
#  MULTI-INTENT FALSE POSITIVE FIXES — ride should NOT fire food, etc.
#  Targeted hard negatives for specific cross-intent confusion.
# ───────────────────────────────────────────────────────────────────────
V41_MULTI_INTENT_FIXES = [
    # ride without food (v4 confused "cab" with food_order)
    ("book a cab to {place}",                           {"ride": 1, "maps": 1}),
    ("get me a cab to {airport}",                       {"ride": 1, "maps": 1}),
    ("book a cab and pay later",                        {"ride": 1}),
    ("cab to {place} please",                           {"ride": 1, "maps": 1}),
    ("get a cab home",                                  {"ride": 1, "maps": 1}),
    ("take a cab to {place}",                           {"ride": 1, "maps": 1}),
    ("need a cab to the office",                        {"ride": 1, "maps": 1}),
    ("hail a cab",                                      {"ride": 1}),
    ("call me a cab",                                   {"ride": 1}),
    ("i need a cab right now",                          {"ride": 1}),
    ("cab from {place} to {place2}",                    {"ride": 1, "maps": 1}),
    ("taxi to {place}",                                 {"ride": 1, "maps": 1}),
    ("auto to {place}",                                 {"ride": 1, "maps": 1}),
    ("rickshaw to {place}",                             {"ride": 1, "maps": 1}),

    # pay + ride (not food)
    ("uber to {place} and pay with card",               {"ride": 1, "maps": 1}),
    ("cab there and I'll pay cash",                     {"ride": 1}),
    ("take a lyft and split the fare",                  {"ride": 1, "money": 1}),

    # shopping without food
    ("order {item} and pay on delivery",                {"shopping": 1}),
    ("buy {item} and pay later",                        {"shopping": 1}),
    ("order {item} from amazon",                        {"shopping": 1}),

    # alarm without calendar (just a reminder, not a meeting)
    ("remind me to eat",                                {"alarm": 1}),
    ("remind me to drink water",                        {"alarm": 1}),
    ("alarm at {time} to take meds",                    {"alarm": 1}),
    ("ping me to stretch at {time}",                    {"alarm": 1}),
    ("remind me to lock the door",                      {"alarm": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  SMS-SPEAK / HEAVY TYPO TEMPLATES — "ordr fud from mcd" etc.
#  v4 augment() only did light single-char drops. Real chat is messier.
# ───────────────────────────────────────────────────────────────────────
V41_SMS_SPEAK = [
    # food
    ("ordr fud from {provider_food}",                   {"food_order": 1}),
    ("ordr {food} pls",                                 {"food_order": 1}),
    ("gt me sum {food}",                                {"food_order": 1}),
    ("fud delivery pls",                                {"food_order": 1}),
    ("wnt {food} tn",                                   {"food_order": 1}),
    ("ordring {food} rn",                               {"food_order": 1}),
    ("delivr {food} to home",                           {"food_order": 1}),
    ("can u ordr {food}",                               {"food_order": 1}),

    # ride
    ("buk a cab to {place}",                            {"ride": 1, "maps": 1}),
    ("ubr to {airport}",                                {"ride": 1, "maps": 1}),
    ("ned a ride to {place}",                           {"ride": 1, "maps": 1}),
    ("gt me a lyft",                                    {"ride": 1}),
    ("rideshr to {place}",                              {"ride": 1, "maps": 1}),

    # money
    ("vmo me {amount}",                                 {"money": 1}),
    ("py me bck {amount}",                              {"money": 1}),
    ("snd {amount} to {addressee}",                     {"money": 1}),
    ("trnsfr {amount} pls",                             {"money": 1}),
    ("u owe me {amount} bro",                           {"money": 1}),

    # alarm
    ("rmnd me at {time}",                               {"alarm": 1}),
    ("set alrm for {time}",                             {"alarm": 1}),
    ("wke me up at {time}",                             {"alarm": 1}),
    ("remindme {time}",                                 {"alarm": 1}),

    # contact
    ("cal {addressee}",                                 {"contact": 1}),
    ("txt {addressee} pls",                             {"contact": 1}),
    ("msg {addressee}",                                 {"contact": 1}),
    ("fon {addressee}",                                 {"contact": 1}),

    # shopping
    ("ordr {item} frm amazon",                          {"shopping": 1}),
    ("buy {item} onln",                                 {"shopping": 1}),
    ("gt {item} delivrd",                               {"shopping": 1}),

    # travel
    ("buk flyt to {city}",                              {"travel": 1}),
    ("flt to {city} nxt wk",                            {"travel": 1}),

    # calendar
    ("mtng with {addressee} tmrw",                      {"calendar": 1}),
    ("schdl lunch {day}",                               {"calendar": 1}),

    # maps
    ("hw to gt to {place}",                             {"maps": 1}),
    ("dirctons to {place}",                             {"maps": 1}),

    # music
    ("ply {song}",                                      {"music": 1}),
    ("ply sum music",                                   {"music": 1}),

    # bills
    ("py my {bill_kind} bil",                           {"bills": 1}),
    ("rnt due {day}",                                   {"bills": 1}),
]
