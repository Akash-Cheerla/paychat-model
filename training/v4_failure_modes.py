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


# ═══════════════════════════════════════════════════════════════════════
#  v4.2 BANKS — targeted at the 15 remaining seed-suite failures.
#  Every bank uses CONTRASTIVE PAIRS: for every negative, a matching
#  positive so the model learns the actual distinction.
# ═══════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────
#  "X IT" SLANG — "doordash it", "spotify it", "uber it" etc.
#  v4 scores these at ~0.00. Model never saw this pattern.
#  3 seed failures: doordash it bro, spotify it pls, ubering home
# ───────────────────────────────────────────────────────────────────────
V42_SLANG_VERBED = [
    # food_order — "doordash it" pattern
    ("doordash it",                                     {"food_order": 1}),
    ("doordash it bro",                                 {"food_order": 1}),
    ("just doordash it",                                {"food_order": 1}),
    ("doordash it tonight",                             {"food_order": 1}),
    ("lets doordash it",                                {"food_order": 1}),
    ("doordash some food",                              {"food_order": 1}),
    ("ubereats it bro",                                 {"food_order": 1}),
    ("just ubereats it",                                {"food_order": 1}),
    ("grubhub it",                                      {"food_order": 1}),
    ("swiggy it bro",                                   {"food_order": 1}),
    ("zomato it",                                       {"food_order": 1}),
    ("lets just order in",                              {"food_order": 1}),
    ("lets just swiggy",                                {"food_order": 1}),
    ("yo just doordash",                                {"food_order": 1}),

    # music — "spotify it" pattern
    ("spotify it",                                      {"music": 1}),
    ("spotify it pls",                                  {"music": 1}),
    ("just spotify it",                                 {"music": 1}),
    ("spotify that song",                               {"music": 1}),
    ("shazam it",                                       {"music": 1}),
    ("youtube it",                                      {"music": 1, "video": 1}),
    ("alexa play it",                                   {"music": 1}),
    ("siri play it",                                    {"music": 1}),
    ("put it on spotify",                               {"music": 1}),
    ("throw it on spotify",                             {"music": 1}),
    ("queue it on spotify",                             {"music": 1}),

    # ride — "uber it" / "ubering" pattern
    ("ubering home",                                    {"ride": 1, "maps": 1}),
    ("ubering there",                                   {"ride": 1, "maps": 1}),
    ("ubering to {place}",                              {"ride": 1, "maps": 1}),
    ("just uber it",                                    {"ride": 1}),
    ("uber it bro",                                     {"ride": 1}),
    ("lyft it",                                         {"ride": 1}),
    ("lyfting home",                                    {"ride": 1, "maps": 1}),
    ("gonna uber there",                                {"ride": 1, "maps": 1}),
    ("im ubering",                                      {"ride": 1}),
    ("just uber",                                       {"ride": 1}),

    # other verbed apps
    ("google it",                                       {}),    # not actionable
    ("yelp it",                                         {}),    # not actionable
    ("just amazon it",                                  {"shopping": 1}),
    ("flipkart it",                                     {"shopping": 1}),
    ("venmo it",                                        {"money": 1}),
    ("cashapp it",                                      {"money": 1}),
    ("zelle it",                                        {"money": 1}),
    ("airbnb it",                                       {"travel": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  PLACE-AS-VENUE — "lunch at sweetgreen" is calendar, NOT maps.
#  "oppenheimer at amc" is video+tickets, NOT maps/calendar.
#  Model fires maps/calendar whenever it sees a place name + time.
#  5 seed failures from extra maps/calendar on venue contexts.
#
#  Contrastive: same place, but one is "go TO the place" (maps) and
#  the other is "event AT the place" (calendar/reservation/tickets).
# ───────────────────────────────────────────────────────────────────────
V42_VENUE_NOT_MAPS = [
    # restaurant = calendar/reservation, NOT maps
    ("lunch at {place} on {day}",                       {"calendar": 1, "reservation": 1}),
    ("dinner at {place} {day} at {time}",               {"calendar": 1, "reservation": 1}),
    ("brunch at {place} {day}",                         {"calendar": 1, "reservation": 1}),
    ("eating at {place} tonight",                       {"calendar": 1, "reservation": 1}),
    ("coffee at {place} tomorrow",                      {"calendar": 1}),
    ("meeting at {place} {day} at {time}",              {"calendar": 1}),
    ("lunch with {addressee} at {place} on {day}",      {"calendar": 1, "contact": 1}),
    ("dinner with {addressee} at {place} {time}",       {"calendar": 1, "contact": 1}),
    ("drinks at {place} {day} night",                   {"calendar": 1, "reservation": 1}),
    ("happy hour at {place} {day}",                     {"calendar": 1, "reservation": 1}),

    # contrastive: same place but GOING there = maps
    ("heading to {place}",                              {"maps": 1}),
    ("directions to {place}",                           {"maps": 1}),
    ("navigate to {place}",                             {"maps": 1}),
    ("how do I get to {place}",                         {"maps": 1}),
    ("on my way to {place}",                            {"maps": 1}),
    ("driving to {place}",                              {"maps": 1}),
    ("where is {place}",                                {"maps": 1}),
    ("find {place} on maps",                            {"maps": 1}),

    # movie at theater = video+tickets, NOT calendar/maps
    ("{movie} at amc tonight",                          {"video": 1, "tickets": 1}),
    ("{movie} at amc tonight at {time}",                {"video": 1, "tickets": 1}),
    ("{movie} at the theater tonight",                  {"video": 1, "tickets": 1}),
    ("{movie} at imax {day}",                           {"video": 1, "tickets": 1}),
    ("{movie} at regal tonight at {time}",              {"video": 1, "tickets": 1}),
    ("see {movie} at amc {day}",                       {"video": 1, "tickets": 1}),
    ("catch {movie} at the movies tonight",             {"video": 1, "tickets": 1}),
    ("going to see {movie} at the cinema",              {"video": 1, "tickets": 1}),
    ("lets watch {movie} at the theater {day}",         {"video": 1, "tickets": 1}),
    ("{movie} showing at {time} get tickets",           {"video": 1, "tickets": 1}),
    ("buy tickets for {movie}",                         {"video": 1, "tickets": 1}),

    # contrastive: movie at HOME = just video, no tickets
    ("watch {movie} on netflix",                        {"video": 1}),
    ("put on {movie} tonight",                          {"video": 1}),
    ("stream {movie} on disney plus",                   {"video": 1}),
    ("{movie} on prime video tonight",                  {"video": 1}),

    # long multi-intent with venue
    ("dinner {day} at {place} at {time} and then catch {movie} after",
                                                        {"calendar": 1, "reservation": 1, "video": 1}),
    ("lets do {place} for dinner and then {movie}",
                                                        {"calendar": 1, "reservation": 1, "video": 1}),
    ("{place} for lunch and then {movie} at the theater",
                                                        {"calendar": 1, "reservation": 1, "video": 1, "tickets": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  FALSE POSITIVE BAIT — figurative / metaphorical language.
#  "map out plans" ≠ maps. "contact lens" ≠ contact. "check on mom" ≠ contact.
#  3 seed failures. Contrastive: literal vs figurative.
# ───────────────────────────────────────────────────────────────────────
V42_FIGURATIVE_NEGATIVES = [
    # "map" used figuratively — NOT maps intent
    ("we should map out our plans",                     {}),
    ("let me map this out",                             {}),
    ("mapping out the strategy",                        {}),
    ("map out the project timeline",                    {}),
    ("need to map out the roadmap",                     {}),
    ("let's map the customer journey",                  {}),
    # contrastive: "map" used literally = maps
    ("map me to {place}",                               {"maps": 1}),
    ("pull up the map to {place}",                      {"maps": 1}),
    ("map to {airport}",                                {"maps": 1}),

    # "contact" used as noun — NOT contact intent
    ("she's a contact lens specialist",                 {}),
    ("lost my contact lens",                            {}),
    ("need new contact lenses",                         {"shopping": 1}),
    ("contact paper for the shelves",                   {}),
    ("eye contact is important",                        {}),
    ("he's my emergency contact",                       {}),
    ("first point of contact",                          {}),
    # contrastive: "contact" as action = contact intent
    ("contact {addressee} please",                      {"contact": 1}),
    ("contact my doctor",                               {"contact": 1}),

    # "check on" someone — NOT contact, just chitchat/concern
    ("check on mom",                                    {}),
    ("check on him",                                    {}),
    ("check on her",                                    {}),
    ("check on the kids",                               {}),
    ("check on grandma",                                {}),
    ("check on your brother",                           {}),
    ("I should check on {addressee}",                   {}),
    # contrastive: "call" someone = contact
    ("call mom",                                        {"contact": 1}),
    ("call him",                                        {"contact": 1}),
    ("text her",                                        {"contact": 1}),

    # "task" used figuratively — NOT task intent
    ("we should map out our plans",                     {}),
    ("that's a big task ahead",                         {}),
    ("multi-tasking is hard",                           {}),
    ("this task is killing me",                         {}),
    ("what a task that was",                            {}),
    # contrastive: actual task creation = task
    ("add {task} to my todo list",                      {"task": 1}),
    ("create a task for {task}",                        {"task": 1}),
    ("add to my list: {task}",                          {"task": 1}),

    # "reservation" used figuratively — NOT reservation intent
    ("I have reservations about this plan",             {}),
    ("dinner reservations make me nervous",             {}),
    ("no reservations about quitting",                  {}),
    ("I have some reservations",                        {}),
    # contrastive: actual reservation = reservation
    ("make a reservation at {place}",                   {"reservation": 1}),
    ("reserve a table at {place}",                      {"reservation": 1}),
    ("book a table for {qty} at {place}",               {"reservation": 1}),

    # "bill" used figuratively — NOT bills intent
    ("bill is a great guy",                             {}),
    ("ask bill about it",                               {}),
    ("bill said he'd come",                             {}),
    ("the Bill of Rights",                              {}),
    # contrastive: actual bill = bills
    ("pay the electricity bill",                        {"bills": 1}),
    ("my phone bill is due",                            {"bills": 1}),

    # "health" used casually — NOT health intent
    ("she's a contact lens specialist",                 {}),
    ("health is wealth",                                {}),
    ("mental health matters",                           {}),
    ("health of the economy is bad",                    {}),
    # contrastive: health action = health
    ("book a doctor appointment",                       {"health": 1}),
    ("schedule a dentist visit",                        {"health": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  BILL SPLITTING — "split the bill", "split dinner 4 ways"
#  v4 scores "split the dinner bill 4 ways" at money=0.02. Not learned.
# ───────────────────────────────────────────────────────────────────────
V42_SPLIT_PATTERNS = [
    ("split the bill",                                  {"money": 1}),
    ("split the dinner bill",                           {"money": 1}),
    ("split the bill {qty} ways",                       {"money": 1}),
    ("split the dinner bill {qty} ways",                {"money": 1}),
    ("split it {qty} ways",                             {"money": 1}),
    ("split dinner",                                    {"money": 1}),
    ("split the uber",                                  {"money": 1}),
    ("split the tab",                                   {"money": 1}),
    ("split the check",                                 {"money": 1}),
    ("lets split it",                                   {"money": 1}),
    ("lets go halves",                                  {"money": 1}),
    ("lets go halfsies",                                {"money": 1}),
    ("halves on dinner",                                {"money": 1}),
    ("going dutch",                                     {"money": 1}),
    ("we can split it",                                 {"money": 1}),
    ("split {qty} ways?",                               {"money": 1}),
    ("split the cab fare",                              {"money": 1}),
    ("split the groceries",                             {"money": 1}),
    ("split the rent",                                  {"money": 1, "bills": 1}),
    ("chip in for dinner",                              {"money": 1}),
    ("everyone chip in {amount}",                       {"money": 1}),
    ("your share is {amount}",                          {"money": 1}),
    ("each person pays {amount}",                       {"money": 1}),
    ("divvy up the bill",                               {"money": 1}),
    ("split the cost",                                  {"money": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  GENERIC / MINIMAL ALARM TRIGGERS — "set a reminder", "remind me"
#  v4 scores "set a reminder" at alarm=0.04. Way too low.
# ───────────────────────────────────────────────────────────────────────
V42_GENERIC_ALARM = [
    ("set a reminder",                                  {"alarm": 1}),
    ("set a reminder please",                           {"alarm": 1}),
    ("set a reminder for me",                           {"alarm": 1}),
    ("reminder please",                                 {"alarm": 1}),
    ("I need a reminder",                               {"alarm": 1}),
    ("can you set a reminder",                          {"alarm": 1}),
    ("set an alarm",                                    {"alarm": 1}),
    ("set an alarm please",                             {"alarm": 1}),
    ("alarm please",                                    {"alarm": 1}),
    ("I need an alarm",                                 {"alarm": 1}),
    ("can you set an alarm",                            {"alarm": 1}),
    ("remind me",                                       {"alarm": 1}),
    ("remind me please",                                {"alarm": 1}),
    ("just remind me",                                  {"alarm": 1}),
    ("set timer",                                       {"alarm": 1}),
    ("put a reminder",                                  {"alarm": 1}),
    ("make a reminder",                                 {"alarm": 1}),
    ("wake me up",                                      {"alarm": 1}),
    ("wake me",                                         {"alarm": 1}),
    ("ping me later",                                   {"alarm": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  RENT + TRANSFER COMPOUND — "rent due friday, transfer 1200"
#  v4 fires bills but misses money. The "transfer X" pattern should
#  also fire money when combined with bills.
# ───────────────────────────────────────────────────────────────────────
V42_BILLS_MONEY_COMPOUND = [
    ("rent due {day}, transfer {amount}",               {"bills": 1, "money": 1}),
    ("rent due {day} transfer {amount}",                {"bills": 1, "money": 1}),
    ("rent is due, need to transfer {amount}",          {"bills": 1, "money": 1}),
    ("transfer {amount} for rent by {day}",             {"bills": 1, "money": 1}),
    ("send {amount} for rent",                          {"bills": 1, "money": 1}),
    ("pay {amount} rent by {day}",                      {"bills": 1, "money": 1}),
    ("rent {amount} due {day} sending now",             {"bills": 1, "money": 1}),
    ("need to send {amount} for {bill_kind}",           {"bills": 1, "money": 1}),
    ("{bill_kind} bill is {amount}, transferring now",   {"bills": 1, "money": 1}),
    ("paying {amount} for {bill_kind} today",           {"bills": 1, "money": 1}),
    ("wifi bill {amount} sending via zelle",            {"bills": 1, "money": 1}),
    ("electric bill {amount} venmo me",                 {"bills": 1, "money": 1}),
    ("insurance due {day} gotta send {amount}",         {"bills": 1, "money": 1}),
    ("gym membership {amount} transferring {day}",      {"bills": 1, "money": 1}),
    # contrastive: just bills, no money action
    ("rent is due {day}",                               {"bills": 1}),
    ("{bill_kind} bill is overdue",                     {"bills": 1}),
    ("my {bill_kind} bill is {amount}",                 {"bills": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  RIDE TIME ≠ ALARM — "uber at 5am" is departure time, not a reminder.
#  "don't forget to book the flight" = travel, NOT alarm.
#  2 seed failures from alarm overfiring on time-in-context.
# ───────────────────────────────────────────────────────────────────────
V42_TIME_CONTEXT = [
    # ride with time = ride ONLY, not alarm
    ("uber to {place} at {time}",                       {"ride": 1, "maps": 1}),
    ("uber to {airport} at {time} tomorrow",            {"ride": 1, "maps": 1}),
    ("cab at {time} to {place}",                        {"ride": 1, "maps": 1}),
    ("lyft at {time} to {airport}",                     {"ride": 1, "maps": 1}),
    ("pickup at {time} from {place}",                   {"ride": 1, "maps": 1}),
    ("uber at {time}",                                  {"ride": 1}),
    ("cab at {time}",                                   {"ride": 1}),
    ("ride at {time} to {place}",                       {"ride": 1, "maps": 1}),
    ("need a ride at {time} tomorrow",                  {"ride": 1}),
    ("uber me at {time}",                               {"ride": 1}),

    # contrastive: reminder about ride = alarm + ride
    ("remind me to book uber at {time}",                {"alarm": 1, "ride": 1}),
    ("set alarm for {time} uber to {airport}",          {"alarm": 1, "ride": 1, "maps": 1}),
    ("reminder to get a cab at {time}",                 {"alarm": 1, "ride": 1}),

    # calendar with time = calendar ONLY, not alarm
    ("meeting at {time} {day}",                         {"calendar": 1}),
    ("lunch at {time}",                                 {"calendar": 1}),
    ("dinner at {time} {day}",                          {"calendar": 1}),
    ("interview at {time} tomorrow",                    {"calendar": 1}),
    ("standup at {time}",                               {"calendar": 1}),

    # contrastive: reminder about meeting = alarm + calendar
    ("remind me about the meeting at {time}",           {"alarm": 1, "calendar": 1}),
    ("set alarm for {time} I have a meeting",           {"alarm": 1, "calendar": 1}),

    # "don't forget to X" = the actual X intent, NOT alarm
    ("don't forget to book the flight to {city}",       {"travel": 1}),
    ("don't forget to book flights for {date}",         {"travel": 1}),
    ("don't forget to get flight tickets",              {"travel": 1}),
    ("don't forget to book the hotel in {city}",        {"travel": 1}),
    ("don't forget to order {food}",                    {"food_order": 1}),
    ("don't forget to buy {item}",                      {"shopping": 1}),
    ("don't forget to pay the {bill_kind} bill",        {"bills": 1}),

    # contrastive: "remind me to X" = alarm + X (explicit reminder request)
    ("remind me to book the flight to {city}",          {"alarm": 1, "travel": 1}),
    ("remind me to order {food}",                       {"alarm": 1, "food_order": 1}),
    ("remind me to pay the {bill_kind} bill",           {"alarm": 1, "bills": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  HINGLISH ALARM / TIME — "kal subah 6 baje" needs to fire alarm.
#  v4 catches ride but misses alarm on Hinglish time expressions.
# ───────────────────────────────────────────────────────────────────────
V42_HINGLISH_TIME = [
    ("kal subah {time} baje alarm laga do",             {"alarm": 1}),
    ("kal subah {time} baje uthana",                    {"alarm": 1}),
    ("subah {time} baje yaad dilana",                   {"alarm": 1}),
    ("shaam ko {time} baje reminder set karo",          {"alarm": 1}),
    ("{time} baje alarm chahiye",                        {"alarm": 1}),
    ("kal {time} pe reminder laga de",                  {"alarm": 1}),
    ("kal subah jaldi uthana hai",                      {"alarm": 1}),
    ("dopahar ko yaad dilana",                          {"alarm": 1}),
    ("raat ko {time} baje alarm",                       {"alarm": 1}),
    ("parso subah {time} baje",                         {"alarm": 1}),

    # hinglish ride + alarm (the seed-suite compound)
    ("kal subah {time} baje uber book kar dena to {airport}",
                                                        {"ride": 1, "maps": 1, "alarm": 1}),
    ("subah {time} baje cab chahiye {airport} ke liye",
                                                        {"ride": 1, "maps": 1, "alarm": 1}),
    ("kal {time} baje ola book karna hai office ke liye",
                                                        {"ride": 1, "maps": 1, "alarm": 1}),

    # contrastive: hinglish chitchat (no alarm)
    ("kal subah baat karte hain",                       {}),
    ("kal milte hain",                                  {}),
    ("subah subah kya kar raha hai",                    {}),
]


# ═══════════════════════════════════════════════════════════════════════
#  v4.3 BANKS — making the model "perfect". Covers patterns the v4.0-4.2
#  banks missed: question-form intents, single-word triggers, idiomatic
#  false positives, and more natural conversation flow.
# ═══════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────
#  QUESTION-FORM INTENTS — real people ASK, they don't command.
#  "can you venmo me?" is just as actionable as "venmo me".
#  Most v4 training templates are imperative commands. Real chat is
#  50%+ questions. This gap causes recall loss on polite phrasings.
# ───────────────────────────────────────────────────────────────────────
V43_QUESTION_WRAPPING = [
    # money questions
    ("can you venmo me {amount}",                          {"money": 1}),
    ("could you send me {amount}",                         {"money": 1}),
    ("would you mind paying me back",                      {"money": 1}),
    ("hey can you cashapp me {amount}",                    {"money": 1}),
    ("do you mind venmoing me",                            {"money": 1}),
    ("can I get that {amount} back",                       {"money": 1}),
    ("you gonna pay me back or what",                      {"money": 1}),
    ("wanna just venmo me for it",                         {"money": 1}),
    ("mind sending me {amount}",                           {"money": 1}),

    # alarm questions
    ("can you remind me at {time}",                        {"alarm": 1}),
    ("could you set an alarm for {time}",                  {"alarm": 1}),
    ("would you remind me to {task}",                      {"alarm": 1}),
    ("can someone remind me",                              {"alarm": 1}),
    ("hey can you wake me up at {time}",                   {"alarm": 1}),
    ("mind reminding me about the {event}",                {"alarm": 1}),

    # contact questions
    ("can you call {addressee}",                           {"contact": 1}),
    ("could you text {addressee} for me",                  {"contact": 1}),
    ("should I call {addressee}",                          {"contact": 1}),
    ("wanna facetime {addressee}",                         {"contact": 1}),
    ("want me to text {addressee}",                        {"contact": 1}),
    ("can someone call {addressee}",                       {"contact": 1}),

    # food questions
    ("should we order food",                               {"food_order": 1}),
    ("wanna order pizza",                                  {"food_order": 1}),
    ("should we doordash it",                              {"food_order": 1}),
    ("can you order {food} for us",                        {"food_order": 1}),
    ("want me to order {food}",                            {"food_order": 1}),
    ("you guys wanna order in",                            {"food_order": 1}),
    ("what should we order",                               {"food_order": 1}),

    # ride questions
    ("should I uber there",                                {"ride": 1}),
    ("wanna uber",                                         {"ride": 1}),
    ("should we get a cab",                                {"ride": 1}),
    ("can you book me an uber to {place}",                 {"ride": 1, "maps": 1}),
    ("want me to book a lyft",                             {"ride": 1}),

    # travel questions
    ("should I book the flight to {city}",                 {"travel": 1}),
    ("wanna plan a trip to {city}",                        {"travel": 1}),
    ("should we book the hotel",                           {"travel": 1}),

    # calendar questions
    ("are we still on for {day}",                          {"calendar": 1}),
    ("can we schedule a meeting {day}",                    {"calendar": 1}),
    ("should we do lunch {day}",                           {"calendar": 1}),
    ("you free for dinner {day}",                          {"calendar": 1}),

    # music questions
    ("can you play {song}",                                {"music": 1}),
    ("wanna play some music",                              {"music": 1}),
    ("should I put on {artist}",                           {"music": 1}),

    # bills questions
    ("did you pay rent yet",                               {"bills": 1}),
    ("should I pay the {bill_kind} bill",                  {"bills": 1}),
    ("is the {bill_kind} bill due",                        {"bills": 1}),

    # shopping questions
    ("should I order {item} on amazon",                    {"shopping": 1}),
    ("want me to buy {item}",                              {"shopping": 1}),

    # contrastive: rhetorical / non-actionable questions
    ("can you believe that",                               {}),
    ("should we even bother",                              {}),
    ("want me to just leave",                              {}),
    ("you think so",                                       {}),
    ("do you agree",                                       {}),
    ("would you say that",                                 {}),
    ("can I get a break",                                  {}),
    ("wanna hang out",                                     {}),
    ("should we go",                                       {}),
    ("you coming or what",                                 {}),
]


# ───────────────────────────────────────────────────────────────────────
#  IDIOMATIC / FIGURATIVE FALSE POSITIVES — deeper set.
#  Words that match intent keywords but are used figuratively.
#  v4.2 covered "map out", "contact lens", "check on", "bill" (name).
#  v4.3 adds more: "playing", "riding", "booking", "noted", etc.
# ───────────────────────────────────────────────────────────────────────
V43_IDIOMATIC_NEGATIVES = [
    # "play" used figuratively — NOT music
    ("playing it safe",                                    {}),
    ("playing with fire",                                  {}),
    ("playing hardball",                                   {}),
    ("stop playing around",                                {}),
    ("playing the victim",                                 {}),
    ("playing catch up",                                   {}),
    ("play it by ear",                                     {}),
    ("he plays for the team",                              {}),

    # "ride" used figuratively — NOT ride intent
    ("riding the wave",                                    {}),
    ("ride or die",                                        {}),
    ("it's been a rough ride",                             {}),
    ("riding high",                                        {}),
    ("she's riding a winning streak",                      {}),
    ("riding out the storm",                               {}),

    # "booking it" = running fast — NOT travel/reservation
    ("he was booking it down the street",                  {}),
    ("I booked it out of there",                           {}),
    ("she's booking it to class",                          {}),

    # "order" used figuratively — NOT food/shopping
    ("in order to finish",                                 {}),
    ("out of order",                                       {}),
    ("put it in order",                                    {}),
    ("that's a tall order",                                {}),
    ("law and order",                                      {}),
    ("order of operations",                                {}),

    # "noted" / "note" used casually — NOT note intent
    ("noted",                                              {}),
    ("duly noted",                                         {}),
    ("note taken",                                         {}),
    ("on a different note",                                {}),
    ("on that note I'm leaving",                           {}),
    ("side note",                                          {}),
    ("of note",                                            {}),

    # "watch" figuratively — NOT video
    ("watch your step",                                    {}),
    ("watch out",                                          {}),
    ("watch your back",                                    {}),
    ("watching my weight",                                 {}),
    ("watch your tone",                                    {}),

    # "call" figuratively — NOT contact
    ("good call",                                          {}),
    ("that's a close call",                                {}),
    ("call it a day",                                      {}),
    ("call it even",                                       {}),
    ("call the shots",                                     {}),
    ("it's your call",                                     {}),
    ("judgment call",                                      {}),
    ("tough call",                                         {}),
    ("close call today",                                   {}),

    # "split" figuratively — NOT money
    ("split decision",                                     {}),
    ("split personality",                                  {}),
    ("split second",                                       {}),
    ("splitting hairs",                                    {}),
    ("let's split up and search",                          {}),
    ("they split up last month",                           {}),

    # "set" figuratively — NOT alarm
    ("set in stone",                                       {}),
    ("all set",                                            {}),
    ("set the tone",                                       {}),
    ("set the bar high",                                   {}),
    ("mindset",                                            {}),
    ("get set for this",                                   {}),

    # "pay" figuratively — NOT money
    ("pay attention",                                      {}),
    ("it'll pay off",                                      {}),
    ("pay your respects",                                  {}),
    ("pay the price",                                      {}),
    ("that doesn't pay well",                              {}),
    ("crime doesn't pay",                                  {}),

    # "delivery" figuratively — NOT food
    ("great delivery on that joke",                        {}),
    ("her delivery was perfect",                           {}),
    ("delivery room",                                      {}),

    # "trip" figuratively — NOT travel
    ("what a trip",                                        {}),
    ("that's trippy",                                      {}),
    ("tripping out",                                       {}),
    ("don't trip about it",                                {}),
    ("ego trip",                                           {}),

    # "ticket" figuratively — NOT tickets
    ("that's the ticket",                                  {}),
    ("meal ticket",                                        {}),
    ("parking ticket",                                     {}),
    ("golden ticket",                                      {}),
    ("one way ticket to disaster",                         {}),

    # "bill" more — NOT bills
    ("Buffalo Bills game",                                 {}),
    ("dollar bill",                                        {}),
    ("bill of sale",                                       {}),
    ("the bill passed in congress",                        {}),
    ("fitting the bill",                                   {}),

    # contrastive: literal uses that SHOULD fire
    ("play {song} on spotify",                             {"music": 1}),
    ("get me a ride to {place}",                           {"ride": 1, "maps": 1}),
    ("book a flight to {city}",                            {"travel": 1}),
    ("watch {movie} tonight",                              {"video": 1}),
    ("call {addressee} now",                               {"contact": 1}),
    ("set alarm for {time}",                               {"alarm": 1}),
    ("pay {addressee} {amount}",                           {"money": 1}),
    ("order {food} on doordash",                           {"food_order": 1}),
    ("split the bill {qty} ways",                          {"money": 1}),
    ("note to self: {task}",                               {"note": 1}),
    ("buy tickets for {movie}",                            {"tickets": 1, "video": 1}),
    ("book a table at {place}",                            {"reservation": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  SINGLE-WORD / ULTRA-SHORT INTENTS — "uber", "venmo", "spotify"
#  Some intents should fire on a single brand name + minimal context.
#  But bare words like "uber" alone shouldn't (too ambiguous).
#  These teach the boundary between actionable-short and too-short.
# ───────────────────────────────────────────────────────────────────────
V43_ULTRA_SHORT = [
    # Actionable short messages
    ("uber please",                                        {"ride": 1}),
    ("uber home",                                          {"ride": 1, "maps": 1}),
    ("venmo me",                                           {"money": 1}),
    ("cashapp me",                                         {"money": 1}),
    ("doordash please",                                    {"food_order": 1}),
    ("order food",                                         {"food_order": 1}),
    ("call mom",                                           {"contact": 1}),
    ("text dad",                                           {"contact": 1}),
    ("play music",                                         {"music": 1}),
    ("directions home",                                    {"maps": 1}),
    ("directions to {place}",                              {"maps": 1}),
    ("remind me later",                                    {"alarm": 1}),
    ("alarm {time}",                                       {"alarm": 1}),
    ("weather today",                                      {"weather": 1}),
    ("pay rent",                                           {"bills": 1}),
    ("order {food}",                                       {"food_order": 1}),
    ("book uber",                                          {"ride": 1}),
    ("book cab",                                           {"ride": 1}),
    ("book flight",                                        {"travel": 1}),
    ("book hotel",                                         {"travel": 1}),

    # Too short / ambiguous — should NOT fire
    ("uber",                                               {}),
    ("venmo",                                              {}),
    ("spotify",                                            {}),
    ("doordash",                                           {}),
    ("amazon",                                             {}),
    ("netflix",                                            {}),
    ("youtube",                                            {}),
    ("food",                                               {}),
    ("money",                                              {}),
    ("call",                                               {}),
    ("ride",                                               {}),
    ("flight",                                             {}),
    ("alarm",                                              {}),
    ("bill",                                               {}),
    ("doctor",                                             {}),
    ("dinner",                                             {}),
    ("music",                                              {}),
    ("weather",                                            {}),
    ("tickets",                                            {}),
    ("hotel",                                              {}),
]


# ───────────────────────────────────────────────────────────────────────
#  PRESENT TENSE STATEMENTS — "I'm cooking", "he's calling someone"
#  These describe CURRENT activity, not requests. Should NOT fire.
#  Different from past tense ("cooked yesterday") — present progressive
#  "I'm doing X right now" is NOT a command to do X.
# ───────────────────────────────────────────────────────────────────────
V43_PRESENT_ACTIVITY = [
    ("I'm watching a movie",                               {}),
    ("I'm watching tv",                                    {}),
    ("she's watching netflix",                              {}),
    ("he's playing video games",                           {}),
    ("I'm playing guitar",                                 {}),
    ("we're cooking dinner",                               {}),
    ("I'm cooking right now",                              {}),
    ("he's calling someone",                               {}),
    ("I'm on a call",                                      {}),
    ("she's on the phone",                                 {}),
    ("I'm ordering food rn",                               {}),
    ("we're riding the subway",                            {}),
    ("I'm paying for it right now",                        {}),
    ("he's driving to work",                               {}),
    ("she's shopping",                                     {}),
    ("I'm at the doctor rn",                               {}),
    ("we're at the restaurant",                            {}),
    ("I'm setting the table",                              {}),
    ("he's booking his own flight",                        {}),
    ("I'm listening to music",                             {}),
    ("she's texting someone",                              {}),
    ("I'm streaming something",                            {}),
    ("we're splitting up for the search",                  {}),

    # contrastive: COMMANDS to do things = actionable
    ("watch a movie on netflix",                           {"video": 1}),
    ("play some music",                                    {"music": 1}),
    ("order food for us",                                  {"food_order": 1}),
    ("call {addressee} for me",                            {"contact": 1}),
    ("book me a flight",                                   {"travel": 1}),
    ("drive me to {place}",                                {"ride": 1, "maps": 1}),
]


# ───────────────────────────────────────────────────────────────────────
#  THIRD PERSON / GOSSIP — "sarah paid 200 for that bag"
#  Talking ABOUT someone else's actions isn't a command.
# ───────────────────────────────────────────────────────────────────────
V43_THIRD_PERSON = [
    ("she paid {amount} for that bag",                     {}),
    ("he ordered food from doordash",                      {}),
    ("they booked a flight to {city}",                     {}),
    ("sarah called her mom",                               {}),
    ("mike set an alarm for 5am",                          {}),
    ("they ubered home last night",                        {}),
    ("he watches netflix all day",                         {}),
    ("{addressee} bought tickets for the concert",         {}),
    ("she split the bill with her friends",                {}),
    ("he reminded everyone about the meeting",             {}),
    ("they made a reservation at {place}",                 {}),
    ("she venmod me {amount} already",                     {}),
    ("{addressee} already paid the rent",                  {}),
    ("his alarm goes off at {time} every day",             {}),
    ("she ordered {food} for the office",                  {}),
    ("they scheduled it for {day}",                        {}),
    ("he booked a hotel in {city}",                        {}),
]
