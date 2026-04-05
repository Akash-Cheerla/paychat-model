"""
Training Data Generator for PayChat Money Detector
Generates high-quality labeled examples across all 4 detection categories:
  1. Owing/Debt
  2. Bill Splitting
  3. Direct Amounts
  4. Venmo/CashApp mentions
Also generates hard negatives (messages that mention money-adjacent topics but are NOT requests)
"""

import json
import random
import re
from pathlib import Path

random.seed(42)

# ─────────────────────────────────────────────
# POSITIVE EXAMPLES — money detection required
# ─────────────────────────────────────────────

OWING_DEBT = [
    # Direct debt
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
    "I kept the receipt — you owe {amount}",
    "quick reminder: you owe me {amount} for the Lyft",
    "hey I covered the tip, that's {amount} from you",
    "you said you'd pay me back ages ago",
    "your share was {amount} btw",
    "just a heads up you still owe me",
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
    "divide the bill — your share is {amount}",
    "let's do {amount} each for the Uber",
    "splitting the groceries {amount} ways tonight",
    "we split the utility bill evenly right?",
    "your half comes out to {amount}",
    "I calculated your share — it's {amount}",
    "everyone owes {amount} for the group gift",
    "splitting the hotel room 3 ways so {amount} each",
    "let's divide this evenly, about {amount} a person",
    "we're splitting the cable bill right?",
    "can you chip in {amount} for the supplies?",
    "splitting the dinner check — you're {amount}",
    "let's cut costs and split the subscription",
    "your portion of the bill is {amount}",
    "we're splitting everything down the middle",
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
]

# Mixed/compound (include multiple signals)
MIXED_MONEY = [
    "you owe me {amount}, just venmo me",
    "split it with me? should be {amount} each, cashapp me",
    "we can split the bill — your half is {amount}, send on venmo",
    "you owe me {amount} for the Uber, venmo: @handle",
    "hey I covered dinner, everyone venmo me {amount}",
    "can you zelle me {amount}? we're splitting the Airbnb",
    "I paid {amount} for parking, split it with me?",
    "you owe me {amount}, just cashapp me when you can",
    "let's split the subscription — {amount} each, I'll venmo request",
    "still owe me {amount} from the grocery run, my venmo is @name",
]

# ─────────────────────────────────────────────
# NEGATIVE EXAMPLES — NOT money-related
# ─────────────────────────────────────────────

NOT_MONEY_CASUAL = [
    "what are you up to tonight",
    "did you see that game last night",
    "lmao that's hilarious",
    "no way that actually happened",
    "I'm so tired honestly",
    "this weather is insane",
    "can you send me that link",
    "what time are we meeting",
    "you around later?",
    "I'm on my way",
    "that movie was so good",
    "happy birthday!! 🎉",
    "ok I'll be there in 10",
    "omg same I was thinking that",
    "are you coming to the party?",
    "I just got here",
    "this place is packed",
    "where are you parked",
    "I'll meet you outside",
    "he said what?!",
]

NOT_MONEY_TRANSACTIONAL = [
    "can you send me the doc",
    "what's the address again",
    "I'll send you the photos later",
    "forward me that email",
    "can you share the spreadsheet",
    "what time does it start",
    "I'll send over the details soon",
    "just text me when you're ready",
    "can you share your location",
    "send me the invite link",
    "what's the WiFi password",
    "I'll forward you the email",
    "you got the address I sent?",
    "just dm me on instagram",
    "drop me a pin when you get there",
]

NOT_MONEY_TRICKY = [
    # Mentions numbers but not money
    "I'll be there at 7",
    "my number is 555-1234",
    "it's on floor 3",
    "we need like 5 people",
    "the score was 3-1",
    "meet at gate 12",
    "text me at 4pm",
    "I'm at table 8",
    # Mentions financial topics but not requesting payment
    "prices are insane these days",
    "I'm so broke lately",
    "just got paid finally",
    "rent is going up again",
    "gas prices are ridiculous",
    "I need to save more money",
    "I'm saving up for a trip",
    "money doesn't grow on trees",
    "I spent too much this month",
    "I'm on a budget right now",
]

# ─────────────────────────────────────────────
# Amount generators
# ─────────────────────────────────────────────

def random_amount():
    templates = [
        "${:.0f}",
        "${:.2f}",
        "{:.0f} bucks",
        "{:.0f} dollars",
        "like ${:.0f}",
        "around ${:.0f}",
        "about ${:.0f}",
    ]
    amounts = [5, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 80, 100, 120, 150, 200]
    amt = random.choice(amounts)
    template = random.choice(templates)
    if "{:.2f}" in template:
        return template.format(amt + random.choice([0, 0.5, 0.99]))
    return template.format(amt)

def fill(template):
    return template.replace("{amount}", random_amount())

# ─────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────

def augment(text):
    """Apply random light augmentations to simulate real chat variety."""
    variants = [text]

    # Lowercase/uppercase variation
    if random.random() < 0.3:
        variants.append(text.lower())
    if random.random() < 0.15:
        variants.append(text.upper())

    # Add filler words
    fillers_pre = ["hey ", "yo ", "btw ", "fyi ", "ok so ", "so ", "also "]
    fillers_post = [" lol", " haha", " fr", " ngl", " tbh", " tho", " rn", " asap"]
    if random.random() < 0.3:
        variants.append(random.choice(fillers_pre) + text)
    if random.random() < 0.25:
        variants.append(text + random.choice(fillers_post))

    # Typo simulation (very light)
    if random.random() < 0.1:
        words = text.split()
        if words:
            idx = random.randint(0, len(words) - 1)
            w = words[idx]
            if len(w) > 3:
                i = random.randint(1, len(w) - 2)
                words[idx] = w[:i] + w[i+1:]  # drop a char
            variants.append(" ".join(words))

    return random.choice(variants)

# ─────────────────────────────────────────────
# Build dataset
# ─────────────────────────────────────────────

def generate_dataset(n_per_category=600):
    dataset = []

    categories = {
        "owing_debt": OWING_DEBT,
        "bill_splitting": BILL_SPLITTING,
        "direct_amount": DIRECT_AMOUNTS,
        "venmo_cashapp": VENMO_CASHAPP,
        "mixed": MIXED_MONEY,
    }

    for cat_name, templates in categories.items():
        count = n_per_category if cat_name != "mixed" else n_per_category // 2
        for _ in range(count):
            template = random.choice(templates)
            text = fill(template)
            text = augment(text)
            dataset.append({
                "text": text,
                "label": 1,
                "category": cat_name,
                "split": "train"
            })

    # Negatives (match total positive count for balance)
    total_pos = len(dataset)
    neg_templates = NOT_MONEY_CASUAL + NOT_MONEY_TRANSACTIONAL + NOT_MONEY_TRICKY
    for _ in range(total_pos):
        text = augment(random.choice(neg_templates))
        dataset.append({
            "text": text,
            "label": 0,
            "category": "not_money",
            "split": "train"
        })

    # Shuffle
    random.shuffle(dataset)

    # Train/val/test split: 80/10/10
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


def save_splits(dataset, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    for item in dataset:
        splits[item["split"]].append(item)

    for split_name, items in splits.items():
        path = out_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  {split_name}: {len(items)} examples -> {path}")

    # Also save full dataset
    with open(out_dir / "full_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    # Category breakdown
    from collections import Counter
    cats = Counter(item["category"] for item in dataset if item["label"] == 1)
    print(f"\nPositive example breakdown:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    print(f"\nTotal: {len(dataset)} examples ({sum(1 for d in dataset if d['label']==1)} positive, {sum(1 for d in dataset if d['label']==0)} negative)")


if __name__ == "__main__":
    print("Generating training data...")
    dataset = generate_dataset(n_per_category=600)
    save_splits(dataset, ".")
    print("\nDone! Ready for training.")
