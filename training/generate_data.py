"""
Training Data Generator for PayChat Multi-Intent Detector.

Intents (multi-label — a message can fire multiple):
  - money     — payments, debts, splits, venmo/cashapp/zelle
  - alarm     — reminders, wake-me-up, set-alarm, ping-me
  - contact   — phone numbers to save (US + India)
  - calendar  — meetings, events, appointments with a date/time
  - maps      — places, addresses, meet-me-at, directions

Each example has a `labels` dict with one 0/1 per intent.
A single message can fire multiple intents (e.g. "meet me at blue bottle 3pm" -> maps + calendar).

Output: train.json / val.json / test.json / full_dataset.json
"""

import json
import random
import re
from pathlib import Path

random.seed(42)

INTENTS = ["money", "alarm", "contact", "calendar", "maps"]


# ═════════════════════════════════════════════════════════════════════
#  MONEY TEMPLATES (unchanged from the money-only model)
# ═════════════════════════════════════════════════════════════════════

OWING_DEBT = [
    "you owe me {amount} from last week",
    "hey don't forget you owe me {amount}",
    "still waiting on that {amount} you owe me bro",
    "you never paid me back the {amount} for groceries",
    "when are you gonna pay me back?",
    "pay me back whenever you get a chance",
    "you still owe me from the concert",
    "bro you literally owe me like {amount}",
    "I covered you last time, you owe me one",
    "I paid for everything last night, you owe me",
    "don't forget about the money you owe me",
    "you been owing me {amount} since forever",
    "remember you borrowed {amount}? lmk when ur free to pay",
    "I need that money back asap",
    "can you please just pay me back already",
    "you owe me for the pizza last week",
    "still haven't gotten my money back smh",
    "pay me back the {amount} you borrowed",
    "settle up with me when you can",
    "I kept the receipt - you owe {amount}",
    "quick reminder: you owe me {amount} for the Lyft",
    "hey I covered the tip, that's {amount} from you",
    "you said you'd pay me back ages ago",
    "your share was {amount} btw",
    "just a heads up you still owe me",
    "where's my money dude",
    "where's my money bro",
    "give me my money back",
    "I want my money back",
    "need my money back asap",
    "you gonna pay me or what",
    "bruh pay up",
    "pay up already",
    "cough up the {amount}",
    "you still haven't paid me",
    "bro I lent you {amount} remember",
    "that {amount} I lent you last month",
    "return my {amount} please",
    "when am I getting my money",
    "hello? my money?",
    "yo my {amount}??",
    "dude the {amount}",
    "gimme my money",
    "gimme my {amount} back",
    "you borrowed {amount} and never paid back",
    "how about that {amount} you owe",
    "still waiting on my {amount}",
    "u owe me big time",
    "pay up or else lol",
    "do I need to remind you about the {amount}",
    "seriously tho pay me back",
]

BILL_SPLITTING = [
    "let's split the bill tonight",
    "can we go halves on this?",
    "should we split dinner?",
    "let's split it evenly",
    "we can split the cost",
    "just split it with me {amount} each",
    "splitting the Airbnb {amount} ways",
    "we should split rent this month",
    "split the tab with me?",
    "going halves on the pizza?",
    "divide the bill - your share is {amount}",
    "let's do {amount} each for the Uber",
    "splitting the groceries {amount} ways tonight",
    "we split the utility bill evenly right?",
    "your half comes out to {amount}",
    "I calculated your share - it's {amount}",
    "everyone owes {amount} for the group gift",
    "splitting the hotel room 3 ways so {amount} each",
    "let's divide this evenly, about {amount} a person",
    "we're splitting the cable bill right?",
    "can you chip in {amount} for the supplies?",
    "splitting the dinner check - you're {amount}",
    "let's cut costs and split the subscription",
    "your portion of the bill is {amount}",
    "we're splitting everything down the middle",
    "let's go dutch",
    "going dutch on this one",
    "50/50 on the bill?",
    "50 50 on dinner?",
    "wanna go halfsies?",
    "halfsies on the pizza?",
    "let's just split it",
    "we splitting this or what",
    "how we splitting this",
    "what's my share",
    "what do I owe for dinner",
    "how much is my part",
    "what's everyone's share",
    "chip in for the gift?",
    "everyone pitch in {amount}",
    "pitch in for the food?",
    "throw in {amount} each",
    "we all throwing in?",
    "let's pool money for this",
]

DIRECT_AMOUNTS = [
    "that'll be {amount}",
    "can you send me {amount} real quick",
    "I need {amount} from you",
    "just send {amount} whenever",
    "hit me with {amount}",
    "send over {amount} when you get a chance",
    "drop me {amount} for the tickets",
    "it's only {amount} no big deal",
    "I'm out {amount} because of last night",
    "that cost me {amount} total",
    "I spent {amount} on your behalf",
    "total came to {amount} so far",
    "we're looking at {amount} each",
    "I fronted {amount} for everyone",
    "it was like {amount} altogether",
    "literally just {amount} don't overthink it",
    "my bill was {amount} what was yours",
    "charged {amount} on my card for all of us",
    "send me {amount} and we're even",
    "it adds up to {amount} with tax",
    "mine was {amount} yours was more",
    "I'm short {amount} can you help",
    "{amount} and we're square",
    "only {amount} each, super reasonable",
    "knocked {amount} off the total already",
]

VENMO_CASHAPP = [
    "just venmo me",
    "I'll venmo you later",
    "send it through venmo",
    "venmo me {amount} whenever",
    "you can cashapp me",
    "send on cashapp",
    "I'll cashapp you back tonight",
    "zelle me if you have it",
    "just zelle the money over",
    "I'll send it on venmo rn",
    "cashapp me {amount}?",
    "venmo me @username for the food",
    "throw it on venmo when u can",
    "I only have venmo and cashapp",
    "my venmo is @handle",
    "what's your venmo handle?",
    "do you have cashapp set up?",
    "drop it to my cashapp",
    "just venmo me the {amount}",
    "I'll zelle you {amount} tonight",
    "does anyone use apple pay? venmo me otherwise",
    "send {amount} via cashapp or venmo either works",
    "my cashapp is $handle, send whenever",
    "request me on venmo, I'll approve",
    "sent you {amount} on venmo, check it",
    "lemme venmo you rn",
    "lemme cashapp you real quick",
    "I'll venmo you back tonight",
    "let me venmo you for that",
    "venmo you later for the food",
    "cashapp you the {amount} now",
    "apple pay me {amount}",
    "just apple pay me",
    "paypal me the {amount}",
    "send it on paypal",
    "gpay me {amount}",
    "google pay me",
    "I'll venmo request you",
    "sending you a venmo request",
    "accept my venmo request",
    "check your venmo I sent {amount}",
    "just sent on cashapp",
    "I venmo'd you",
    "venmo'd you {amount} just now",
    "cashapped you the {amount}",
    "zelle'd you {amount}",
    "wire me the {amount}",
    "transfer me {amount}",
    "do a bank transfer for {amount}",
]

MIXED_MONEY = [
    "you owe me {amount}, just venmo me",
    "split it with me? should be {amount} each, cashapp me",
    "we can split the bill - your half is {amount}, send on venmo",
    "you owe me {amount} for the Uber, venmo: @handle",
    "hey I covered dinner, everyone venmo me {amount}",
    "can you zelle me {amount}? we're splitting the Airbnb",
    "I paid {amount} for parking, split it with me?",
    "you owe me {amount}, just cashapp me when you can",
    "let's split the subscription - {amount} each, I'll venmo request",
    "still owe me {amount} from the grocery run, my venmo is @name",
    "I'll cover it don't worry",
    "I'll cover it no worries",
    "don't worry I got this one",
    "my treat tonight",
    "my treat don't worry about it",
    "it's on me tonight",
    "this one's on me",
    "I got you on the tab",
    "I got you bro don't worry",
    "i got you on this one",
    "ima send you the money",
    "ima send you the {amount}",
    "ima pay you back tonight",
    "imma venmo you later",
    "imma cashapp you the {amount}",
    "shall I send you the remaining?",
    "should I send the rest?",
    "want me to send the remaining {amount}?",
    "I'll take care of the bill",
    "let me take care of this",
    "let me get you back for that",
    "I'll get you back for the {amount}",
    "drinks on me",
    "dinner's on me",
    "lunch is on me today",
    "I'll pay for everything",
    "I'll foot the bill",
    "let me foot the bill",
    "I'll pick up the tab",
    "let me pick up the tab",
    "here's {amount} for gas",
    "take {amount} for helping",
    "here's some money for food",
    "I'll spot you {amount}",
    "let me spot you",
    "spot me {amount}?",
    "can you spot me",
    "lend me {amount}?",
    "can I borrow {amount}",
    "loan me {amount} till Friday",
    "can you loan me some money",
    "how much do I owe",
    "how much do I owe you",
    "what do I owe you",
    "do I owe you anything",
    "are we even",
    "are we square",
    "we good on money?",
]


# ═════════════════════════════════════════════════════════════════════
#  ALARM TEMPLATES
# ═════════════════════════════════════════════════════════════════════

ALARM_REMIND_ME = [
    "remind me to {task} at {time}",
    "remind me to {task} {time}",
    "remind me at {time} to {task}",
    "remind me in {duration} to {task}",
    "remind me about {task} {time}",
    "can you remind me to {task} {time}",
    "pls remind me to {task} at {time}",
    "remind me to {task}",
    "need a reminder to {task} {time}",
    "set a reminder for {task} at {time}",
    "set a reminder to {task} {time}",
    "reminder: {task} at {time}",
    "hey remind me to {task} later",
    "don't let me forget to {task}",
    "remember to {task} {time}",
    "note to self - {task} at {time}",
    "remind me to {task} in {duration}",
    "remind me later to {task}",
    "nudge me {time} about {task}",
    "poke me in {duration} for {task}",
]

ALARM_SET_ALARM = [
    "set an alarm for {time}",
    "set an alarm at {time}",
    "set alarm for {time}",
    "alarm at {time}",
    "alarm for {time}",
    "alarm {time}",
    "I need an alarm for {time}",
    "put an alarm on for {time}",
    "can you set an alarm for {time}",
    "set an alarm in {duration}",
    "I need to wake up at {time}, alarm please",
    "alarm tomorrow at {time}",
    "set alarm tomorrow {time}",
    "need an alarm set for {time}",
    "please set an alarm for {time}",
    "set multiple alarms starting {time}",
    "alarm at {time} and {time2}",
]

ALARM_WAKEUP = [
    "wake me up at {time}",
    "wake me at {time}",
    "wake me up in {duration}",
    "someone wake me up at {time}",
    "please wake me at {time}",
    "wake me up tomorrow at {time}",
    "wake me tomorrow {time}",
    "gonna need a wake up call at {time}",
    "wake me before {time}",
    "wake me when it's {time}",
    "can someone wake me up at {time}",
]

ALARM_PING_NOTIFY = [
    "ping me at {time}",
    "ping me in {duration}",
    "ping me when it's {time}",
    "notify me at {time}",
    "alert me at {time}",
    "buzz me at {time}",
    "text me at {time}",
    "hit me up at {time}",
    "tell me at {time} to {task}",
    "ping me {time} about {task}",
]

ALARM_MISC = [
    "timer for {duration}",
    "set a timer for {duration}",
    "start a timer {duration}",
    "{duration} timer plz",
    "countdown {duration}",
    "can you start a timer for {duration}",
    "need a {duration} timer",
    "remind me in {duration}",
    "in {duration} remind me",
    "give me a {duration} heads up",
]


# ═════════════════════════════════════════════════════════════════════
#  CONTACT TEMPLATES (US + India phone formats)
# ═════════════════════════════════════════════════════════════════════

CONTACT_SHARE_OTHER = [
    "save {name}'s number: {phone}",
    "this is {name}'s number {phone}",
    "{name}'s number is {phone}",
    "{name}'s cell {phone}",
    "here's {name}'s contact {phone}",
    "here is {name}s number {phone}",
    "{name} is at {phone}",
    "{name}'s new number {phone}",
    "{name} just changed his number to {phone}",
    "you can reach {name} at {phone}",
    "call {name} on {phone}",
    "text {name} at {phone}",
    "ping {name} on {phone}",
    "hey save this, {name}'s cell is {phone}",
    "add {name} to your contacts - {phone}",
    "save as {name}: {phone}",
    "{name} phone: {phone}",
    "number for {name}: {phone}",
    "the plumber's number is {phone}",
    "dentist's office {phone}",
    "electrician {name} {phone}",
]

CONTACT_SHARE_SELF = [
    "my number is {phone}",
    "my number - {phone}",
    "this is my number {phone}",
    "save my number {phone}",
    "hey save my number {phone}",
    "btw my cell {phone}",
    "my new number is {phone}",
    "I just got a new number {phone}",
    "my contact {phone}",
    "text me at {phone}",
    "reach me at {phone}",
    "you can call me on {phone}",
    "my cell {phone}",
    "here's my digits {phone}",
    "feel free to call me at {phone}",
    "save my contact - {phone}",
]

CONTACT_ADDRESSED = [
    "{addressee} save this, {name}'s number {phone}",
    "hey {addressee}, {name}'s cell is {phone}",
    "{addressee} add {name} to contacts {phone}",
    "{addressee} here's the plumber's number {phone}",
    "{addressee} save {phone} as {name}",
    "{addressee} this is {name} from work {phone}",
    "@{addressee} save {name}'s number {phone}",
    "{addressee} take this number down {phone}",
    "yo {addressee}, {name}'s number is {phone}",
]

CONTACT_BUSINESS = [
    "call the restaurant at {phone}",
    "the shop's number is {phone}",
    "reception desk - {phone}",
    "booking hotline {phone}",
    "clinic contact {phone}",
    "the landlord's number {phone}",
    "the electrician's cell {phone}",
    "salon number {phone}",
    "cab driver's number {phone}",
    "uber driver's no {phone}",
    "the guy's number who does the painting is {phone}",
]


# ═════════════════════════════════════════════════════════════════════
#  CALENDAR TEMPLATES
# ═════════════════════════════════════════════════════════════════════

CAL_MEETING = [
    "meeting at {time}",
    "meeting {day} {time}",
    "meeting on {date}",
    "team sync {day} {time}",
    "standup {day} at {time}",
    "1:1 {day} {time}",
    "quick sync at {time}",
    "got a meeting at {time}",
    "call at {time}",
    "zoom call {day} {time}",
    "meeting with {name} at {time}",
    "meeting with {name} {day}",
    "client call {day} at {time}",
    "interview at {time} {day}",
    "performance review {day} at {time}",
    "one on one with {name} {time}",
]

CAL_EVENT = [
    "dinner {day} at {time}",
    "lunch with {name} at {time}",
    "coffee with {name} {day}",
    "drinks {day} at {time}",
    "gym {day} at {time}",
    "yoga class {day} {time}",
    "haircut appointment at {time} {day}",
    "doctor's appointment {day} at {time}",
    "dentist at {time} {day}",
    "therapy {day} at {time}",
    "{event} on {date} at {time}",
    "{event} {day} at {time}",
    "{name}'s birthday {day}",
    "birthday party {day} at {time}",
    "wedding on {date}",
    "{name}'s wedding {date}",
    "graduation on {date}",
    "flight at {time} {day}",
    "train at {time}",
    "concert {day} at {time}",
    "game night {day} {time}",
    "movie night {day} at {time}",
    "class {day} at {time}",
]

CAL_SCHEDULE = [
    "schedule a meeting {day} {time}",
    "let's schedule for {day} at {time}",
    "can we schedule {event} {day}",
    "schedule {event} for {time}",
    "put {event} on the calendar {day}",
    "add {event} to my calendar at {time}",
    "block my calendar {day} {time}",
    "block off {day} from {time} to {time2}",
    "book {event} on {date}",
    "let's book {day} at {time}",
    "pencil me in {day} at {time}",
    "save the date {date}",
    "save the date for {event} - {date}",
]

CAL_INVITE = [
    "let's grab {event} {day}",
    "let's grab {event} at {time}",
    "want to meet {day} at {time}",
    "wanna meet up {day}",
    "free {day} at {time}?",
    "you around {day} at {time}",
    "you free for {event} {day}",
    "up for {event} at {time}",
    "lunch on {day}?",
    "dinner {day}?",
    "meet at {time} on {day}",
    "brunch this {day}",
    "catch up {day} at {time}?",
]

CAL_MULTI_PERSON = [
    "team offsite {date}",
    "everyone in for dinner {day} at {time}",
    "guys lunch at {time}",
    "all hands at {time}",
    "full team meeting {day}",
    "party at my place {day} at {time}",
    "everyone's invited {day}",
    "group dinner {day} at {time}",
    "squad gym session {day} {time}",
    "entire team call {day} at {time}",
]


# ═════════════════════════════════════════════════════════════════════
#  MAPS TEMPLATES
# ═════════════════════════════════════════════════════════════════════

MAPS_MEET_AT = [
    "meet me at {place}",
    "lets meet at {place}",
    "meet at {place}",
    "meet you at {place}",
    "see u at {place}",
    "catch you at {place}",
    "rendezvous at {place}",
    "meet up at {place}",
    "come to {place}",
    "come meet me at {place}",
    "we're meeting at {place}",
    "everyone meet at {place}",
    "group meet at {place}",
]

MAPS_IM_AT = [
    "i'm at {place}",
    "im at {place}",
    "currently at {place}",
    "im parked at {place}",
    "im outside {place}",
    "waiting at {place}",
    "been sitting at {place}",
    "at {place} right now",
    "hanging at {place}",
    "chilling at {place}",
    "im inside {place}",
    "we're at {place}",
]

MAPS_HEADING_TO = [
    "heading to {place}",
    "on my way to {place}",
    "omw to {place}",
    "driving to {place}",
    "pulling up to {place}",
    "pulling into {place}",
    "headed over to {place}",
    "going to {place}",
    "making my way to {place}",
    "about to pull up to {place}",
    "en route to {place}",
]

MAPS_DIRECTIONS = [
    "directions to {place}",
    "how do i get to {place}",
    "what's the route to {place}",
    "shortest way to {place}",
    "navigate to {place}",
    "map me to {place}",
    "can you drop the address for {place}",
    "whats the way to {place}",
    "how far is {place}",
    "distance to {place}",
    "pull up {place} on maps",
    "open {place} in maps",
    "find {place} on maps",
]

MAPS_ADDRESS = [
    "the address is {address}",
    "spot is {address}",
    "come to {address}",
    "{address} is where we are",
    "we're at {address}",
    "send the address - {address}",
    "meet me at {address}",
    "im at {address}",
    "address for the party: {address}",
    "venue is {address}",
    "dropping the address - {address}",
    "here's the address {address}",
    "the airbnb is at {address}",
    "event location: {address}",
]

MAPS_PIN_SHARE = [
    "sharing my pin",
    "dropping a pin at {place}",
    "pin is dropped at {place}",
    "i sent you my location",
    "sharing my live location",
    "check my pin",
    "ill drop a pin",
    "here's the pin {place}",
]


# ═════════════════════════════════════════════════════════════════════
#  CROSS-INTENT MIXED TEMPLATES — messages that fire 2+ intents
# ═════════════════════════════════════════════════════════════════════
#
# Format: (template, {intent: 1 for each fired intent})
# Unmentioned intents default to 0.

MIXED_MULTI = [
    # maps + calendar
    ("meet me at {place} at {time}", {"maps": 1, "calendar": 1}),
    ("dinner at {place} {day} {time}", {"maps": 1, "calendar": 1}),
    ("team offsite at {place} on {date}", {"maps": 1, "calendar": 1}),
    ("lunch with {name} at {place} {time}", {"maps": 1, "calendar": 1}),
    ("let's grab coffee at {place} {day}", {"maps": 1, "calendar": 1}),
    ("meeting at {place} {time}", {"maps": 1, "calendar": 1}),
    ("wedding on {date} at {place}", {"maps": 1, "calendar": 1}),
    ("party at {place} {day} at {time}", {"maps": 1, "calendar": 1}),
    ("birthday at {place} on {date}", {"maps": 1, "calendar": 1}),
    ("interview at {place} {time}", {"maps": 1, "calendar": 1}),

    # maps + alarm
    ("heading to {place}, remind me to leave at {time}", {"maps": 1, "alarm": 1}),
    ("meet at {place}, remind me {time}", {"maps": 1, "alarm": 1}),
    ("i'm at {place}, ping me in {duration}", {"maps": 1, "alarm": 1}),

    # calendar + alarm
    ("meeting at {time}, remind me 10 min before", {"calendar": 1, "alarm": 1}),
    ("{event} at {time}, remind me {duration} before", {"calendar": 1, "alarm": 1}),
    ("flight {day} at {time}, wake me at {time2}", {"calendar": 1, "alarm": 1}),
    ("{event} {day} at {time}, set an alarm", {"calendar": 1, "alarm": 1}),
    ("dentist at {time}, remind me", {"calendar": 1, "alarm": 1}),
    ("gym {day} {time}, remind me 15 min before", {"calendar": 1, "alarm": 1}),
    ("don't let me miss {event} at {time}", {"calendar": 1, "alarm": 1}),

    # contact + alarm
    ("save {name}'s number {phone}, remind me to call them {time}", {"contact": 1, "alarm": 1}),
    ("{name}'s number is {phone}, remind me to follow up {time}", {"contact": 1, "alarm": 1}),

    # contact + calendar
    ("meeting with {name} at {time}, his number is {phone}", {"contact": 1, "calendar": 1}),
    ("{name}'s wedding {date}, his number {phone}", {"contact": 1, "calendar": 1}),

    # contact + maps
    ("{name}'s place is at {address}, his number {phone}", {"contact": 1, "maps": 1}),
    ("pick up from {address}, driver's no {phone}", {"contact": 1, "maps": 1}),

    # money + maps
    ("you owe me {amount}, meet me at {place}", {"money": 1, "maps": 1}),
    ("drinks at {place} - {amount} each", {"money": 1, "maps": 1}),

    # money + calendar
    ("dinner {day} {time} - splitting {amount} each", {"money": 1, "calendar": 1}),
    ("rent due {date} - {amount}", {"money": 1, "calendar": 1}),

    # money + alarm
    ("remind me to venmo you {amount} tomorrow", {"money": 1, "alarm": 1}),
    ("remind me to pay rent {day}", {"money": 1, "alarm": 1}),

    # triple: maps + calendar + alarm
    ("dinner at {place} {time}, remind me {duration} before", {"maps": 1, "calendar": 1, "alarm": 1}),
    ("gym with {name} at {place} {time}, remind me {duration} before", {"maps": 1, "calendar": 1, "alarm": 1}),
    ("{event} at {place} {day} {time}, set a reminder", {"maps": 1, "calendar": 1, "alarm": 1}),

    # triple: money + calendar + maps
    ("dinner {day} at {place}, {amount} each", {"money": 1, "calendar": 1, "maps": 1}),
    ("team offsite at {place} {date} - {amount} per person", {"money": 1, "calendar": 1, "maps": 1}),

    # contact + calendar + alarm
    ("{name}'s interview {day} at {time}, his number {phone}, remind me", {"contact": 1, "calendar": 1, "alarm": 1}),
]


# ═════════════════════════════════════════════════════════════════════
#  NEGATIVE TEMPLATES (shared — boring chat that fires no intent)
# ═════════════════════════════════════════════════════════════════════

NOT_ANYTHING_CASUAL = [
    "what are you up to tonight",
    "did you see that game last night",
    "lmao that's hilarious",
    "no way that actually happened",
    "I'm so tired honestly",
    "this weather is insane",
    "can you send me that link",
    "what time are we meeting",  # ambiguous — let model decide, but bias negative
    "I'll think about it",
    "that movie was so good",
    "happy birthday!!",
    "omg same I was thinking that",
    "this place is packed",
    "he said what?!",
    "that's crazy",
    "I can't believe it",
    "sounds good to me",
    "brb getting food",
    "yo what's good",
    "sup bro",
    "nothing much wbu",
    "chilling at home",
    "just woke up",
    "going to bed soon",
    "good morning",
    "good night everyone",
    "see you tomorrow",
    "miss you guys",
    "that's so funny",
    "this song is fire",
    "check out this meme",
    "haha dead",
    "no cap that's wild",
    "fr fr",
    "I can't rn",
    "lmk",
    "bet",
    "say less",
    "on god",
    "deadass",
    "wild",
    "hell nah",
    "hell yeah",
    "lets go",
    "lets gooo",
    "rip",
    "welp",
    "sheesh",
    "whatever man",
    "figures",
    "ur funny",
    "so true",
    "agreed",
    "ig so",
    "dunno lol",
    "eh maybe",
]


# Past-tense / already-done references — should NOT fire calendar/alarm/maps
NOT_ANYTHING_PAST = [
    "already woke up",
    "already did that",
    "had dinner yesterday",
    "went to {place} last week",
    "was at {place} earlier",
    "met up with {name} yesterday",
    "called {name} last night",
    "my alarm went off already",
    "alarm already rang",
    "saw {name} yesterday",
    "dropped by {place} this morning",
    "was at the meeting earlier",
    "the meeting ended",
    "dinner was at {place} last week",
    "the wedding was last month",
    "appointment was yesterday",
    "got woken up already",
    "drove by {place} earlier",
    "hit up {place} on saturday",
    "the standup is done",
    "meeting just ended",
    "finished my {duration} run",
    "timer went off",
    "already set the alarm thanks",
]

# Phone-number-shaped things that aren't phones
NOT_CONTACT_TRICKY = [
    "my credit card ends in 1234",
    "credit card last 4 is 9876",
    "order # 9876543210",
    "tracking id 9876543210",
    "flight number 9876",
    "confirmation 8765432109",
    "ticket {name} 8765432",
    "passport number 9876543",
    "invoice 9876543210",
    "reference 1234567890",
    "pin is 1234",
    "room 9876",
    "flight AA {num4}",
    "i'm 25 years old",
    "it's been 10 years",
    "3.14 is pi",
    "we need 4 people",
    "score was 9-2",
    "won 10-0 last night",
    "build 9.8.7",
    "version 1.2.3.4",
    "the year 1999",
    "born in 1998",
    "it's 2026 already",
    "zip code 94110",
]

# Place-name-shaped things that aren't navigation intent
NOT_MAPS_TRICKY = [
    "i love paris",
    "paris is beautiful",
    "boston sports are trash",
    "austin powers is funny",
    "paris hilton lol",
    "born in chicago",
    "new york pizza is the best",
    "i'm from la originally",
    "grew up in dallas",
    "miami is too humid",
    "manhattan is crazy",
    "lived in brooklyn for 5 years",
    "sf is wild",
    "la is overrated",
    "vegas was a blur",
    "seattle rain though",
    "new jersey is underrated",
    "the {place} episode of that show",
    "that {place} documentary",
    "texas is huge",
    "california weather",
    "never been to tokyo",
    "wanna go to tokyo someday",  # aspirational, not actionable
    "heard {place} is nice",
    "supposedly {place} has good food",
]

# Time-mentioned but no scheduling intent
NOT_CALENDAR_TRICKY = [
    "what time is it",
    "whats the time",
    "it's 3pm already??",
    "only 10am and im drained",
    "morning already feels long",
    "cant believe its friday",
    "mondays hit different",
    "weekends are the best",
    "time flies",
    "it's been a minute",
    "that was a long time ago",
    "back in the day",
    "some time ago",
    "any time now",
    "one of these days",
    "for the time being",
]

# Money-mentioned but no payment intent (extra)
NOT_MONEY_TRICKY = [
    "prices are insane these days",
    "I'm so broke lately",
    "just got paid finally",
    "rent is going up again",
    "gas prices are ridiculous",
    "money doesn't grow on trees",
    "time is money",
    "penny for your thoughts",
    "worth every penny",
    "you owe me an apology",
    "you owe me an explanation",
    "I owe you one for that favor",
    "that's rich coming from you",
    "he's loaded with work",
]

# Reminder-shaped but not actionable alarm
NOT_ALARM_TRICKY = [
    "I remember that",
    "remember the good old days",
    "remember that time we",
    "do you remember?",
    "never forget",
    "don't remind me lol",
    "don't remind me of that",
    "dont wanna remember",
    "alarming news",
    "that's alarming",
    "alarmingly bad",
    "set the table please",
    "set the record straight",
    "set a good example",
    "timer on the oven already",
    "the alarm in the building went off",
    "fire alarm was a drill",
    "I don't need an alarm",
]


# ═════════════════════════════════════════════════════════════════════
#  GENERATORS — random fillers
# ═════════════════════════════════════════════════════════════════════

_AMOUNTS = [5, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 80, 100, 120, 150, 200]

def random_amount():
    templates = [
        "${:.0f}", "${:.2f}", "{:.0f} bucks", "{:.0f} dollars",
        "like ${:.0f}", "around ${:.0f}", "about ${:.0f}",
    ]
    amt = random.choice(_AMOUNTS)
    t = random.choice(templates)
    if "{:.2f}" in t:
        return t.format(amt + random.choice([0, 0.5, 0.99]))
    return t.format(amt)


def random_time():
    hours = list(range(1, 13))
    mins = ["", ":00", ":15", ":30", ":45"]
    suffixes = ["am", "pm", " am", " pm", "AM", "PM"]
    formats = [
        lambda: f"{random.choice(hours)}{random.choice(mins)}{random.choice(suffixes)}",
        lambda: f"{random.choice(hours)}{random.choice(suffixes)}",
        lambda: f"{random.randint(0,23)}:{random.choice(['00','15','30','45'])}",
        lambda: f"{random.choice(['noon','midnight','morning','afternoon','evening','tonight','first thing'])}",
        lambda: f"{random.choice(hours)} {random.choice(['am','pm'])}",
    ]
    return random.choice(formats)()


def random_duration():
    n = random.choice([5, 10, 15, 20, 30, 45])
    unit = random.choice(["min", "minutes", "mins"])
    if random.random() < 0.3:
        n = random.choice([1, 2, 3, 4])
        unit = random.choice(["hour", "hours", "hr", "hrs"])
    return f"{n} {unit}"


def random_day():
    return random.choice([
        "today", "tomorrow", "tmrw", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday", "mon", "tue", "wed",
        "thu", "fri", "sat", "sun", "next monday", "this friday",
        "next week", "next weekend", "this weekend", "tonight",
    ])


def random_date():
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
              "january", "february", "march", "april", "may", "june", "july", "august"]
    d = random.randint(1, 28)
    m = random.choice(months)
    formats = [
        f"{m} {d}",
        f"{d} {m}",
        f"{d}/{random.randint(1,12)}",
        f"{random.randint(1,12)}/{d}",
        f"{random.randint(1,12)}/{d}/26",
    ]
    return random.choice(formats)


def random_event():
    return random.choice([
        "standup", "team sync", "1:1", "client call", "interview",
        "dinner", "lunch", "coffee", "drinks", "brunch",
        "gym", "yoga", "workout", "run",
        "haircut", "doctor's", "dentist", "therapy",
        "birthday", "wedding", "graduation", "baby shower",
        "flight", "train", "concert", "movie", "game night",
        "class", "lecture", "party", "offsite",
    ])


def random_name():
    return random.choice([
        "akash", "rohit", "priya", "samyak", "nikhil", "aditi", "aunty", "uncle",
        "mom", "dad", "sarah", "mike", "john", "emma", "alex", "chris",
        "jessica", "brian", "kevin", "amanda", "rachel", "dave",
        "the plumber", "the electrician", "the landlord", "the dentist",
        "meera", "kiran", "anjali", "sid", "rohan", "neha",
    ])


def random_addressee():
    return random.choice([
        "akash", "rohit", "priya", "samyak", "sarah", "mike", "john", "alex",
        "chris", "emma", "kevin", "dave",
    ])


def _us_phone():
    area = random.randint(200, 999)
    mid = random.randint(100, 999)
    last = random.randint(1000, 9999)
    formats = [
        f"{area}-{mid}-{last}",
        f"({area}) {mid}-{last}",
        f"+1 {area} {mid} {last}",
        f"+1-{area}-{mid}-{last}",
        f"{area}.{mid}.{last}",
        f"{area}{mid}{last}",
        f"1-{area}-{mid}-{last}",
    ]
    return random.choice(formats)


def _india_phone():
    # India mobile: starts with 6/7/8/9, 10 digits total
    first = random.choice([6, 7, 8, 9])
    rest = "".join(str(random.randint(0, 9)) for _ in range(9))
    num = f"{first}{rest}"
    formats = [
        f"+91 {num[:5]} {num[5:]}",
        f"+91-{num[:5]}-{num[5:]}",
        f"+91{num}",
        f"{num}",
        f"0{num}",
        f"+91 {num}",
    ]
    return random.choice(formats)


def random_phone():
    if random.random() < 0.5:
        return _us_phone()
    return _india_phone()


def random_num4():
    return str(random.randint(1000, 9999))


_PLACES_US = [
    "blue bottle", "blue bottle coffee", "blue bottle on valencia",
    "philz coffee", "starbucks", "starbucks on market",
    "chipotle", "sweetgreen", "chipotle on 4th",
    "the mission", "dolores park", "union square", "central park",
    "times square", "brooklyn bridge", "golden gate park",
    "SFO", "LAX", "JFK", "ORD",
    "moma", "the met", "sfmoma", "academy of sciences",
    "trader joe's", "whole foods", "whole foods on market",
    "target", "costco", "home depot",
    "my place", "your place", "the usual spot", "the rooftop",
    "the corner cafe", "that ramen place", "the sushi spot",
    "coffee shop on 5th", "the gym on mission", "equinox",
]

_PLACES_IN = [
    "cafe coffee day", "starbucks bandra", "blue tokai",
    "bandra bandstand", "marine drive", "hauz khas village",
    "cyber hub", "forum mall", "phoenix mall",
    "koramangala", "indiranagar", "connaught place",
    "the taj", "leela palace", "oberoi",
]


def random_place():
    return random.choice(_PLACES_US + _PLACES_IN)


_STREETS = [
    "main st", "market st", "valencia st", "mission st", "broadway",
    "5th ave", "4th st", "castro st", "polk st", "folsom st",
    "3rd ave", "7th st", "larkin st", "hyde st",
]


def random_address():
    n = random.randint(100, 9999)
    street = random.choice(_STREETS)
    formats = [
        f"{n} {street}",
        f"{n} {street} apt {random.randint(1,20)}",
        f"{n} {street} #{random.randint(1,500)}",
        f"{n} {street}, {random.choice(['sf','ny','la'])}",
    ]
    return random.choice(formats)


def random_task():
    return random.choice([
        "take meds", "take my vitamins", "call mom", "call dad",
        "call the plumber", "check the oven", "turn off the stove",
        "leave for the airport", "pack my bags", "start the laundry",
        "go to the gym", "pick up kids", "pick up groceries",
        "submit the form", "send the email", "call back {}".format(random.choice(['akash','rohit','priya'])),
        "turn off the lights", "water the plants",
        "take out the trash", "book the flight", "pay the bill",
        "reply to sarah", "send the invoice", "follow up with the client",
    ])


# ═════════════════════════════════════════════════════════════════════
#  FILL — replace all {tokens} in a template
# ═════════════════════════════════════════════════════════════════════

_TOKEN_MAP = {
    "amount":    random_amount,
    "time":      random_time,
    "time2":     random_time,
    "duration":  random_duration,
    "day":       random_day,
    "date":      random_date,
    "event":     random_event,
    "name":      random_name,
    "addressee": random_addressee,
    "phone":     random_phone,
    "place":     random_place,
    "address":   random_address,
    "task":      random_task,
    "num4":      random_num4,
}


def fill(template):
    """Replace every {token} in the template with a random filler value."""
    def _sub(match):
        key = match.group(1)
        fn = _TOKEN_MAP.get(key)
        if fn is None:
            return match.group(0)
        return fn()
    return re.sub(r"\{(\w+)\}", _sub, template)


# ═════════════════════════════════════════════════════════════════════
#  AUGMENT — inject casual chat noise
# ═════════════════════════════════════════════════════════════════════

def augment(text):
    variants = [text]

    if random.random() < 0.3:
        variants.append(text.lower())
    if random.random() < 0.1:
        variants.append(text.upper())

    fillers_pre = ["hey ", "yo ", "btw ", "fyi ", "ok so ", "so ", "also ", "wait "]
    fillers_post = [" lol", " haha", " fr", " ngl", " tbh", " tho", " rn", " asap", " plz", " pls"]

    if random.random() < 0.3:
        variants.append(random.choice(fillers_pre) + text)
    if random.random() < 0.25:
        variants.append(text + random.choice(fillers_post))

    # light typo: drop a char from a medium-length word
    if random.random() < 0.1:
        words = text.split()
        if words:
            idx = random.randint(0, len(words) - 1)
            w = words[idx]
            if len(w) > 3:
                i = random.randint(1, len(w) - 2)
                words[idx] = w[:i] + w[i+1:]
            variants.append(" ".join(words))

    return random.choice(variants)


# ═════════════════════════════════════════════════════════════════════
#  BUILD — emit multi-label examples
# ═════════════════════════════════════════════════════════════════════

def _zeros():
    return {k: 0 for k in INTENTS}


def make_example(text, category, labels):
    return {
        "text": text,
        "labels": labels,
        "category": category,
        "split": "train",
    }


def generate_dataset(n_per_intent=600):
    """
    Build the full multi-label dataset.

    Target (roughly):
      money   : ~600 singles + ~30 multi-intent inclusions
      alarm   : ~500 singles + ~60 multi-intent inclusions
      contact : ~500 singles + ~30 multi-intent inclusions
      calendar: ~500 singles + ~60 multi-intent inclusions
      maps    : ~500 singles + ~60 multi-intent inclusions
      negatives: ~1200 (balance against positives)
    Total target: ~4500 examples. Good for a ~25-min colab retrain.
    """
    dataset = []

    # ── Money (keep existing volume) ──
    money_groups = {
        "owing_debt":     OWING_DEBT,
        "bill_splitting": BILL_SPLITTING,
        "direct_amount":  DIRECT_AMOUNTS,
        "venmo_cashapp":  VENMO_CASHAPP,
        "mixed_money":    MIXED_MONEY,
    }
    for cat, templates in money_groups.items():
        count = n_per_intent // 5 if cat != "mixed_money" else n_per_intent // 10
        for _ in range(count):
            text = augment(fill(random.choice(templates)))
            labels = _zeros()
            labels["money"] = 1
            dataset.append(make_example(text, cat, labels))

    # ── Alarm ──
    alarm_groups = {
        "alarm_remind_me": ALARM_REMIND_ME,
        "alarm_set":       ALARM_SET_ALARM,
        "alarm_wakeup":    ALARM_WAKEUP,
        "alarm_ping":      ALARM_PING_NOTIFY,
        "alarm_timer":     ALARM_MISC,
    }
    for cat, templates in alarm_groups.items():
        for _ in range(n_per_intent // 5):
            text = augment(fill(random.choice(templates)))
            labels = _zeros()
            labels["alarm"] = 1
            dataset.append(make_example(text, cat, labels))

    # ── Contact ──
    contact_groups = {
        "contact_other":     CONTACT_SHARE_OTHER,
        "contact_self":      CONTACT_SHARE_SELF,
        "contact_addressed": CONTACT_ADDRESSED,
        "contact_business":  CONTACT_BUSINESS,
    }
    for cat, templates in contact_groups.items():
        for _ in range(n_per_intent // 4):
            text = augment(fill(random.choice(templates)))
            labels = _zeros()
            labels["contact"] = 1
            dataset.append(make_example(text, cat, labels))

    # ── Calendar ──
    calendar_groups = {
        "cal_meeting":      CAL_MEETING,
        "cal_event":        CAL_EVENT,
        "cal_schedule":     CAL_SCHEDULE,
        "cal_invite":       CAL_INVITE,
        "cal_multi_person": CAL_MULTI_PERSON,
    }
    for cat, templates in calendar_groups.items():
        for _ in range(n_per_intent // 5):
            text = augment(fill(random.choice(templates)))
            labels = _zeros()
            labels["calendar"] = 1
            dataset.append(make_example(text, cat, labels))

    # ── Maps ──
    maps_groups = {
        "maps_meet_at":    MAPS_MEET_AT,
        "maps_im_at":      MAPS_IM_AT,
        "maps_heading":    MAPS_HEADING_TO,
        "maps_directions": MAPS_DIRECTIONS,
        "maps_address":    MAPS_ADDRESS,
        "maps_pin":        MAPS_PIN_SHARE,
    }
    for cat, templates in maps_groups.items():
        for _ in range(n_per_intent // 6):
            text = augment(fill(random.choice(templates)))
            labels = _zeros()
            labels["maps"] = 1
            dataset.append(make_example(text, cat, labels))

    # ── Cross-intent mixed ──
    # Each template generates 8 augmented variants -> ~300 multi-intent examples
    for template, intent_flags in MIXED_MULTI:
        for _ in range(8):
            text = augment(fill(template))
            labels = _zeros()
            labels.update(intent_flags)
            dataset.append(make_example(text, "multi_intent", labels))

    # ── Negatives ──
    total_pos = len(dataset)
    neg_templates = (
        NOT_ANYTHING_CASUAL
        + NOT_ANYTHING_PAST
        + NOT_CONTACT_TRICKY
        + NOT_MAPS_TRICKY
        + NOT_CALENDAR_TRICKY
        + NOT_MONEY_TRICKY
        + NOT_ALARM_TRICKY
    )
    # Match 50% of positive count (we have a lot of positives, don't need 1:1)
    n_neg = int(total_pos * 0.5)
    for _ in range(n_neg):
        text = augment(fill(random.choice(neg_templates)))
        labels = _zeros()
        dataset.append(make_example(text, "none", labels))

    # ── Shuffle + split (80/10/10) ──
    random.shuffle(dataset)
    n = len(dataset)
    for i, item in enumerate(dataset):
        ratio = i / n
        if ratio < 0.80:
            item["split"] = "train"
        elif ratio < 0.90:
            item["split"] = "val"
        else:
            item["split"] = "test"

    return dataset


# ═════════════════════════════════════════════════════════════════════
#  SAVE
# ═════════════════════════════════════════════════════════════════════

def save_splits(dataset, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    for item in dataset:
        splits[item["split"]].append(item)

    for split_name, items in splits.items():
        path = out_dir / f"{split_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        print(f"  {split_name}: {len(items)} examples -> {path}")

    with open(out_dir / "full_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    # Per-intent breakdown
    from collections import Counter
    print("\nPer-intent positive counts (one message can count toward multiple intents):")
    for intent in INTENTS:
        n_pos = sum(1 for d in dataset if d["labels"][intent] == 1)
        print(f"  {intent:<10} {n_pos}")

    n_none = sum(1 for d in dataset if all(v == 0 for v in d["labels"].values()))
    n_multi = sum(1 for d in dataset if sum(d["labels"].values()) >= 2)
    print(f"\n  no-intent (negatives): {n_none}")
    print(f"  multi-intent (>=2):    {n_multi}")
    print(f"\nCategory breakdown:")
    cats = Counter(item["category"] for item in dataset)
    for cat, count in cats.most_common():
        print(f"  {cat:<22} {count}")
    print(f"\nTotal: {len(dataset)} examples")


if __name__ == "__main__":
    print("Generating multi-intent training data...")
    dataset = generate_dataset(n_per_intent=600)
    save_splits(dataset, ".")
    print("\nDone. Ready for multi-label training.")
