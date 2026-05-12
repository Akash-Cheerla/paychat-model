"""
v5 multi-turn conversation training banks.

These teach the model to use conversation context when classifying the
CURRENT message. The model sees:

    <s> prior msg 1 | prior msg 2 </s></s> current message </s>

The labels always describe what action the CURRENT message triggers,
given the accumulated context.

Format: list of (prior_messages: list[str], current_msg: str, intent_flags: dict)
  - prior_messages: 1-4 casual or actionable messages that came before
  - current_msg: the message being classified (may have {token} placeholders)
  - intent_flags: which intents should fire (empty dict = nothing fires)

Design:
  Every positive pattern has a contrastive negative — same current message
  but with different context that SHOULD NOT trigger. This teaches the model
  that context is the deciding factor, not just the current message.

  "bro I'm hungry" → "dominos?" = food_order ✅
  "what's your fav app" → "dominos" = nothing ✅ (same word, different context)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONTEXT BUILDUP — casual chat accumulates, then a short message triggers
#
# This is Gowtham's core ask: unrelated messages build up context, and the
# model fires when the conversation naturally shifts toward an action.
# ─────────────────────────────────────────────────────────────────────────────

V5_CONTEXT_BUILDUP = [
    # ── food context ──
    (["bro I'm so hungry"], "dominos?", {"food_order": 1}),
    (["I'm starving", "same, haven't eaten all day"], "dominos?", {"food_order": 1}),
    (["what should we eat tonight"], "pizza?", {"food_order": 1}),
    (["I haven't eaten since morning", "same dude"], "wanna order something?", {"food_order": 1}),
    (["we need food", "yeah I'm dying"], "doordash?", {"food_order": 1}),
    (["I could eat a horse rn"], "chipotle?", {"food_order": 1}),
    (["lunch?", "yeah I'm down"], "sushi or pizza?", {"food_order": 1}),
    (["bro I skipped breakfast", "same lol"], "mcd?", {"food_order": 1}),
    (["craving noodles tbh"], "should we order?", {"food_order": 1}),
    (["man I'm so hungry"], "ubereats?", {"food_order": 1}),
    (["dinner plans?", "not yet"], "let's order in", {"food_order": 1}),
    (["that biryani place was fire last time"], "order again?", {"food_order": 1}),

    # ── ride context ──
    (["I'm running late"], "uber?", {"ride": 1}),
    (["we need to get there by 5"], "lyft?", {"ride": 1}),
    (["how are we getting there", "idk"], "uber?", {"ride": 1}),
    (["the train is delayed again"], "should we cab it?", {"ride": 1}),
    (["I don't wanna drive tonight"], "let's uber", {"ride": 1}),
    (["parking is gonna be a nightmare"], "cab?", {"ride": 1}),
    (["we're gonna be late at this rate"], "book an uber?", {"ride": 1}),
    (["it's raining hard", "ugh I know"], "uber?", {"ride": 1}),
    (["meeting is at 3, it's already 2:30"], "get a cab?", {"ride": 1}),

    # ── maps/navigation context ──
    (["where's the party at"], "downtown", {"maps": 1}),
    (["I'm lost lol"], "where are you", {"maps": 1}),
    (["meeting is somewhere on market st"], "send the address?", {"maps": 1}),
    (["where are we meeting"], "sweetgreen on 5th", {"maps": 1}),
    (["I can't find the place"], "drop a pin?", {"maps": 1}),

    # ── money context ──
    (["dinner was expensive last night"], "how much do I owe you?", {"money": 1}),
    (["thanks for covering me"], "let me venmo you", {"money": 1}),
    (["I forgot my wallet at the restaurant"], "I'll pay you back", {"money": 1}),
    (["that uber was pricey"], "let's split it", {"money": 1}),
    (["groceries came out to $80"], "split?", {"money": 1}),
    (["the bill was insane", "how much was it"], "your share is 40", {"money": 1}),
    (["thanks for getting dinner"], "venmo?", {"money": 1}),

    # ── alarm/reminder context ──
    (["I have a meeting tomorrow at 9"], "set an alarm?", {"alarm": 1}),
    (["don't wanna oversleep tomorrow"], "wake me at 6?", {"alarm": 1}),
    (["the flight is at 6am"], "remind me to pack tonight", {"alarm": 1}),
    (["class starts early tomorrow"], "alarm for 7?", {"alarm": 1}),

    # ── weather context ──
    (["should I bring an umbrella"], "check the weather?", {"weather": 1}),
    (["planning a picnic this weekend"], "weather?", {"weather": 1}),
    (["going hiking tomorrow"], "hope it doesn't rain", {"weather": 1}),

    # ── music context ──
    (["I'm in the mood for some tunes"], "spotify?", {"music": 1}),
    (["this playlist is getting old"], "play something new?", {"music": 1}),
    (["road trip tomorrow"], "make a playlist?", {"music": 1}),

    # ── video context ──
    (["I'm bored", "same"], "wanna watch something?", {"video": 1}),
    (["nothing to do tonight"], "netflix?", {"video": 1}),
    (["did you see the new season dropped"], "let's watch it", {"video": 1}),

    # ── travel context ──
    (["I need a vacation so bad"], "flights to goa?", {"travel": 1}),
    (["thinking about taking a trip"], "let's look at flights", {"travel": 1}),
    (["where should we go for the long weekend"], "book a hotel?", {"travel": 1}),

    # ── health context ──
    (["I've been having headaches all week"], "should I see a doctor?", {"health": 1}),
    (["my tooth has been killing me"], "book a dentist?", {"health": 1}),
    (["feeling sick since yesterday"], "need a doctor", {"health": 1}),

    # ── calendar context ──
    (["when are you free this week"], "thursday at 3?", {"calendar": 1}),
    (["we should catch up"], "lunch on friday?", {"calendar": 1}),
    (["I miss the team"], "team dinner saturday?", {"calendar": 1}),

    # ── bills context ──
    (["rent is due this week right"], "yeah friday", {"bills": 1}),
    (["did you pay the wifi bill", "not yet"], "it's due tomorrow", {"bills": 1}),

    # ── shopping context ──
    (["my headphones just broke"], "buy new ones on amazon?", {"shopping": 1}),
    (["we need a new router"], "let me check amazon", {"shopping": 1}),

    # ── multi-intent context buildup ──
    (["running late for dinner"], "uber to sweetgreen?", {"ride": 1, "maps": 1}),
    (["I'm hungry and we're late"], "uber to chipotle?", {"ride": 1, "food_order": 1}),
    (["dinner bill was huge"], "split it and venmo me?", {"money": 1}),
    (["meeting tomorrow at 9", "and my alarm is broken"], "set one for 8?", {"alarm": 1}),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1b. CONTRASTIVE — same short message, neutral context → nothing fires
#
# Critical: the model must learn that context MATTERS. "dominos?" after
# "I'm hungry" = food_order. "dominos?" after "what's your fav app?" = nothing.
# ─────────────────────────────────────────────────────────────────────────────

V5_CONTEXT_NEGATIVE = [
    # ── food words, non-food context ──
    (["what's your favorite app"], "dominos", {}),
    (["name a brand"], "pizza hut", {}),
    (["which delivery service do you like"], "doordash", {}),
    (["who sponsors that podcast"], "chipotle", {}),
    (["there's a new restaurant on our street"], "pizza place right?", {}),
    (["what was that ad about"], "ubereats I think", {}),
    (["my friend works at"], "mcd", {}),

    # ── ride words, non-ride context ──
    (["their stock went up today"], "uber", {}),
    (["which company IPO'd recently"], "lyft", {}),
    (["what app do you use for work"], "uber", {}),
    (["who's hiring right now"], "lyft I think", {}),

    # ── money words, non-money context ──
    (["what app do you use most"], "venmo", {}),
    (["which payment app is best"], "cashapp probably", {}),
    (["have you tried"], "zelle", {}),

    # ── music words, non-music context ──
    (["which streaming service do you like"], "spotify", {}),
    (["what app do you pay for"], "spotify", {}),

    # ── weather words, non-weather context ──
    (["how's the political climate"], "stormy lol", {}),
    (["how's work going"], "it's been a cold week", {}),

    # ── casual chat that looks like buildup but isn't ──
    (["bro I'm so tired", "same"], "what's up", {}),
    (["I'm bored", "same lol"], "nothing to do", {}),
    (["ugh mondays", "ikr"], "this is the worst", {}),
    (["haven't eaten all day"], "yeah me neither", {}),  # no action trigger!
    (["I'm so hungry"], "lol same", {}),  # just commiserating
    (["dinner was so good last night"], "what did you get", {}),
    (["traffic is insane right now"], "ugh I know", {}),  # just venting
    (["my phone is dying"], "rip", {}),
    (["rent is so expensive here"], "fr fr", {}),  # just venting
    (["the movie was mid"], "agreed", {}),  # just chatting
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIRMATION AFTER ACTIONABLE
#
# User A says something actionable, User B confirms → carry forward the intent.
# The model sees the prior actionable message as context and a bare confirmation
# as the current message, and should fire the same intent.
# ─────────────────────────────────────────────────────────────────────────────

V5_CONFIRMATION = [
    # ── food_order ──
    (["should we order pizza tonight"], "yeah", {"food_order": 1}),
    (["wanna order dominos"], "sure", {"food_order": 1}),
    (["let's get sushi"], "sounds good", {"food_order": 1}),
    (["order chipotle?"], "do it", {"food_order": 1}),
    (["I'm thinking pizza"], "let's do it", {"food_order": 1}),
    (["should we doordash something"], "yeah go ahead", {"food_order": 1}),

    # ── ride ──
    (["should we uber there"], "yeah", {"ride": 1}),
    (["uber to the airport?"], "sure let's go", {"ride": 1}),
    (["want me to book a cab"], "go ahead", {"ride": 1}),
    (["lyft or uber?"], "uber", {"ride": 1}),
    (["should I call a cab"], "yeah do it", {"ride": 1}),

    # ── money ──
    (["venmo me 20 for the pizza"], "ok sending", {"money": 1}),
    (["can you send me 30 for dinner"], "yeah", {"money": 1}),
    (["should I pay you back now"], "yeah whenever", {"money": 1}),
    (["split the bill?"], "sure", {"money": 1}),
    (["cashapp me your share"], "ok", {"money": 1}),

    # ── alarm ──
    (["should I set an alarm for 7"], "yeah", {"alarm": 1}),
    (["want me to remind you"], "please", {"alarm": 1}),
    (["set a reminder?"], "yes pls", {"alarm": 1}),
    (["alarm for 6am?"], "do it", {"alarm": 1}),

    # ── travel ──
    (["should I book the flight"], "yeah go for it", {"travel": 1}),
    (["flights to goa next week?"], "let's do it", {"travel": 1}),

    # ── music ──
    (["play some music?"], "yeah", {"music": 1}),
    (["put on that playlist"], "sure", {"music": 1}),

    # ── video ──
    (["wanna watch a movie"], "yeah pick something", {"video": 1}),
    (["netflix tonight?"], "sure", {"video": 1}),

    # ── maps ──
    (["should I send the address"], "yeah please", {"maps": 1}),
    (["directions to the restaurant?"], "yeah send em", {"maps": 1}),

    # ── calendar ──
    (["lunch friday at 1?"], "sounds good", {"calendar": 1}),
    (["schedule the meeting for 3pm?"], "yeah works for me", {"calendar": 1}),

    # ── health ──
    (["should I book a doctor appointment"], "yeah you should", {"health": 1}),

    # ── shopping ──
    (["should I order it on amazon"], "yeah", {"shopping": 1}),

    # ── reservation ──
    (["book a table for tonight?"], "yes please", {"reservation": 1}),

    # ── bills ──
    (["should I pay the electricity bill"], "yeah it's due", {"bills": 1}),

    # ── contact ──
    (["should I call mom"], "yeah give her a call", {"contact": 1}),
    (["text dad about dinner?"], "sure", {"contact": 1}),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2b. CONFIRMATION AFTER NON-ACTIONABLE — must NOT fire
#
# "yeah" after casual chat has no intent to carry forward.
# ─────────────────────────────────────────────────────────────────────────────

V5_CONFIRM_NEGATIVE = [
    (["that movie was wild"], "yeah", {}),
    (["I'm so tired today"], "same", {}),
    (["the game was insane last night"], "for real", {}),
    (["bro that meme was hilarious"], "lol yeah", {}),
    (["monday vibes fr"], "ikr", {}),
    (["do you like coffee"], "yeah", {}),
    (["that party was fun"], "sure was", {}),
    (["it's getting cold out"], "yeah fr", {}),
    (["I can't believe it's already may"], "right", {}),
    (["she's so dramatic"], "ok true", {}),
    (["this show is mid"], "agreed", {}),
    (["gym was brutal today"], "sounds like it", {}),
    (["the traffic was terrible"], "ugh yeah", {}),
    (["that test was so hard"], "for real", {}),
    (["he said what??"], "yeah lol", {}),
    (["her cooking is amazing"], "100%", {}),
    (["bro failed the test lol"], "oof", {}),
    (["I miss summer"], "same tbh", {}),
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. CORRECTION AFTER ACTIONABLE
#
# User says something actionable, then corrects a detail → same intent,
# updated slot.
# ─────────────────────────────────────────────────────────────────────────────

V5_CORRECTION = [
    # ── amount correction ──
    (["venmo me 20 for dinner"], "actually make it 30", {"money": 1}),
    (["send me 50"], "wait no, 40", {"money": 1}),
    (["your share is 25"], "nah it's 30", {"money": 1}),
    (["cashapp me 15"], "actually 20", {"money": 1}),

    # ── service/app correction ──
    (["uber to the airport"], "actually make it lyft", {"ride": 1}),
    (["book a lyft"], "wait get uber instead", {"ride": 1}),
    (["order from doordash"], "actually use ubereats", {"food_order": 1}),
    (["let's order on swiggy"], "nah zomato is better", {"food_order": 1}),

    # ── destination correction ──
    (["uber to JFK"], "wait no, LaGuardia", {"ride": 1, "maps": 1}),
    (["heading to starbucks"], "actually let's go to blue bottle", {"maps": 1}),
    (["directions to sweetgreen"], "nah go to chipotle instead", {"maps": 1}),

    # ── time correction ──
    (["set alarm for 6am"], "actually make it 6:30", {"alarm": 1}),
    (["meeting at 3pm"], "wait, change it to 4", {"calendar": 1}),
    (["reminder at 5"], "no, 5:30", {"alarm": 1}),

    # ── food correction ──
    (["order pizza"], "actually let's get sushi", {"food_order": 1}),
    (["get burgers tonight"], "nah pizza sounds better", {"food_order": 1}),

    # ── person correction ──
    (["call mom"], "wait, call dad first", {"contact": 1}),
    (["venmo sarah"], "actually send it to mike", {"money": 1}),

    # ── general corrections ──
    (["book flights to goa"], "actually let's do delhi", {"travel": 1}),
    (["play that taylor swift song"], "no the weeknd one", {"music": 1}),
    (["watch stranger things"], "nah let's watch the office", {"video": 1}),
]


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADDITION AFTER ACTIONABLE
#
# User adds a NEW intent on top of the prior one.
# "order pizza" → "and remind me to take meds" = new alarm intent
# ─────────────────────────────────────────────────────────────────────────────

V5_ADDITION = [
    (["order pizza from dominos"], "and remind me to take meds at 9", {"alarm": 1}),
    (["uber to the airport"], "and remind me to pack", {"alarm": 1}),
    (["venmo me 20 for dinner"], "also call mom after", {"contact": 1}),
    (["book a flight to goa"], "and reserve a hotel too", {"travel": 1, "reservation": 1}),
    (["set an alarm for 7am"], "and order breakfast on doordash", {"food_order": 1}),
    (["uber to sweetgreen"], "oh and also remind me about the 3pm meeting", {"alarm": 1, "calendar": 1}),
    (["order sushi tonight"], "and also get drinks", {"food_order": 1}),
    (["venmo sarah 20"], "and text her about dinner", {"contact": 1}),
    (["play some music"], "and set a timer for 30 mins", {"alarm": 1}),
    (["call dad"], "and remind me to call uncle too", {"alarm": 1, "contact": 1}),
]


# ─────────────────────────────────────────────────────────────────────────────
# 5. CASUAL AFTER ACTIONABLE — no carry-forward
#
# Critical negative: action fired, then casual chat. The model must NOT
# carry forward the intent to unrelated messages.
# ─────────────────────────────────────────────────────────────────────────────

V5_NO_CARRYFORWARD = [
    (["order pizza from dominos"], "that was a good game last night", {}),
    (["venmo me 20"], "haha bro that meme was hilarious", {}),
    (["uber to JFK"], "did you see what she posted", {}),
    (["set alarm for 7am"], "this weather is crazy", {}),
    (["remind me to call mom"], "what time does the game start", {}),
    (["book flights to goa"], "lol that's so true", {}),
    (["play anti-hero"], "I'm so tired rn", {}),
    (["watch stranger things"], "can't believe it's already friday", {}),
    (["call dad"], "he said what??", {}),
    (["pay rent tomorrow"], "bruh mondays hit different", {}),
    (["split the bill"], "anyway how was your weekend", {}),
    (["uber to sweetgreen"], "that test was brutal", {}),
    (["venmo me 30"], "nice pic btw", {}),
    (["remind me at 5pm"], "who even watches that show", {}),
    (["order chipotle"], "this song goes hard", {}),
]


# ─────────────────────────────────────────────────────────────────────────────
# 6. SINGLE-TURN WITH CASUAL CONTEXT — still classifies correctly
#
# Even with prior casual messages, a clear actionable message should fire
# normally. This prevents the model from ONLY relying on context.
# ─────────────────────────────────────────────────────────────────────────────

V5_SINGLE_WITH_CONTEXT = [
    # ── clear actionable after casual ──
    (["what's up", "nothing much"], "venmo me 20 for pizza", {"money": 1}),
    (["how was your day", "pretty good"], "order pizza from dominos", {"food_order": 1}),
    (["lol that meme was fire"], "uber to JFK at 5am", {"ride": 1, "maps": 1}),
    (["bro that game was insane"], "remind me to call mom tomorrow", {"alarm": 1, "contact": 1}),
    (["can't believe it's friday"], "play some music on spotify", {"music": 1}),
    (["haha nice"], "set alarm for 7am", {"alarm": 1}),
    (["yeah for sure"], "book flights to goa next month", {"travel": 1}),
    (["this is so boring"], "watch stranger things on netflix", {"video": 1}),
    (["sup", "nm wbu"], "call dad real quick", {"contact": 1}),
    (["how's the weather there"], "pay the electricity bill", {"bills": 1}),
    (["bruh moment"], "reserve a table for 4 at olive garden", {"reservation": 1}),
    (["good morning everyone"], "navigate to starbucks", {"maps": 1}),
    (["what a week"], "buy headphones on amazon", {"shopping": 1}),
    (["ugh mondays"], "book a doctor appointment", {"health": 1, "calendar": 1}),
    (["nice pic"], "2 tickets for the concert saturday", {"tickets": 1}),
]


# ─────────────────────────────────────────────────────────────────────────────
# 7. DISAMBIGUATION — same message, different context → different outcomes
#
# This is the most powerful pattern. The model learns that identical words
# can mean different things depending on what came before.
# ─────────────────────────────────────────────────────────────────────────────

V5_DISAMBIGUATION = [
    # "sure" — confirmation vs casual
    (["should we order pizza"], "sure", {"food_order": 1}),
    (["that movie was really good"], "sure", {}),

    # "do it" — confirmation vs casual
    (["uber to the airport?"], "do it", {"ride": 1}),
    (["you think I should dye my hair"], "do it", {}),

    # "let's go" — confirmation vs casual
    (["book a cab to downtown?"], "let's go", {"ride": 1}),
    (["the warriors won again"], "let's go", {}),

    # "send it" — money vs casual
    (["venmo me 20"], "send it", {"money": 1}),
    (["check out this meme"], "send it", {}),

    # "book it" — actionable vs casual
    (["should I reserve the table"], "book it", {"reservation": 1}),
    (["you better run"], "book it", {}),

    # "yes" — many contexts
    (["order from dominos?"], "yes", {"food_order": 1}),
    (["was it a good movie?"], "yes", {}),

    # "sounds good" — calendar vs casual
    (["lunch friday at 1?"], "sounds good", {"calendar": 1}),
    (["I'm learning guitar"], "sounds good", {}),

    # numbers — money vs casual
    (["what's your share of dinner"], "30", {"money": 1}),
    (["how old are you"], "30", {}),

    # "ok" — confirmation vs casual
    (["should I set the alarm"], "ok", {"alarm": 1}),
    (["the meeting got cancelled"], "ok", {}),

    # time mentions — calendar vs casual
    (["when should we meet"], "3pm", {"calendar": 1}),
    (["what time did you wake up"], "3pm", {}),

    # "bet" — confirmation vs casual
    (["wanna order pizza"], "bet", {"food_order": 1}),
    (["that was a crazy play"], "bet", {}),

    # place names — maps vs casual
    (["where should we eat"], "sweetgreen", {"maps": 1, "food_order": 1}),
    (["what's your favorite restaurant"], "sweetgreen", {}),
]


# ─────────────────────────────────────────────────────────────────────────────
# 8. MULTI-TURN CHAINS — 3-4 messages of context
#
# Deeper conversations where the model needs several messages to understand.
# ─────────────────────────────────────────────────────────────────────────────

V5_LONG_CONTEXT = [
    # Food ordering chain
    (["what should we do for dinner", "idk, I'm too tired to cook", "same"],
     "dominos?", {"food_order": 1}),

    (["anyone hungry?", "starving", "me too, skipped lunch"],
     "let's order", {"food_order": 1}),

    # Ride chain
    (["the event starts at 8", "we should leave by 7", "who's driving?"],
     "let's just uber", {"ride": 1}),

    (["where's the party", "downtown near the bridge", "that's far"],
     "uber?", {"ride": 1}),

    # Money chain
    (["dinner was 120 total", "that's expensive", "yeah but it was worth it"],
     "split 4 ways?", {"money": 1}),

    (["how much was the uber", "35 bucks", "oof"],
     "I'll venmo you my half", {"money": 1}),

    # Planning chain → multiple intents
    (["we should plan something this weekend", "yeah it's been a while", "what about dinner?"],
     "sweetgreen friday at 7?", {"calendar": 1, "reservation": 1}),

    (["flight is at 6am saturday", "ugh so early", "I know"],
     "set alarm for 4am and book an uber?", {"alarm": 1, "ride": 1}),

    # Chain that stays casual (negative)
    (["this week was rough", "tell me about it", "can't wait for the weekend"],
     "fr fr", {}),

    (["did you watch the game", "nah I missed it", "it was insane"],
     "really? who won", {}),

    (["I'm so tired of cooking", "same honestly", "we need a break"],
     "true", {}),  # just agreeing, no action trigger
]


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate all banks for easy import
# ─────────────────────────────────────────────────────────────────────────────

ALL_V5_BANKS = {
    "v5_context_buildup":     V5_CONTEXT_BUILDUP,
    "v5_context_negative":    V5_CONTEXT_NEGATIVE,
    "v5_confirmation":        V5_CONFIRMATION,
    "v5_confirm_negative":    V5_CONFIRM_NEGATIVE,
    "v5_correction":          V5_CORRECTION,
    "v5_addition":            V5_ADDITION,
    "v5_no_carryforward":     V5_NO_CARRYFORWARD,
    "v5_single_with_context": V5_SINGLE_WITH_CONTEXT,
    "v5_disambiguation":      V5_DISAMBIGUATION,
    "v5_long_context":        V5_LONG_CONTEXT,
}

# Template counts for diagnostics
if __name__ == "__main__":
    total = 0
    for name, bank in ALL_V5_BANKS.items():
        positives = sum(1 for _, _, flags in bank if flags)
        negatives = sum(1 for _, _, flags in bank if not flags)
        print(f"  {name:<30} {len(bank):>3} templates ({positives} pos, {negatives} neg)")
        total += len(bank)
    print(f"\n  Total: {total} multi-turn templates")
