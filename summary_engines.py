"""Summary engines behind one interface, so they can be scored against each other.

Akash, 2026-10-08: set up a bake-off between the rules and a model. The point is to find
out whether a model is better at the two things rules are worst at - writing a title, and
telling an example apart from an instruction - before any of this goes to Brahma.

Three engines, and WHICH DATA EACH MAY SEE is the first thing to understand:

  RulesEngine        the summarizer in summarizer.py. Runs anywhere, sees anything.
  LocalModelEngine   a model loaded with transformers, on this machine. Nothing leaves
                     it, so it may see dogfood.
  ApiProxyEngine     DeepSeek, and it may ONLY see synthetic text. It exists because this
                     laptop has ~2GB of RAM free and a 2017 CPU: the largest model that
                     fits here is about 0.5B, which would lose to the rules and tell us
                     nothing except that the laptop is small. DeepSeek stands in for "a
                     competent instruct model" to measure the CEILING a model would buy
                     us. It is NOT a deployment proposal - production would self-host.

The guard below is not advisory. ApiProxyEngine refuses any text that did not come from
the generated set, because the dogfood logs are other people's private conversations and
the 2026-09-10 approval for DeepSeek covers measurement on synthetic data only.
"""
import json
import os
import re
from pathlib import Path
from typing import List, Optional

import summarizer

ROOT = Path(__file__).resolve().parent

PROMPT = """You summarize one chat message for the person who received it, for a
"Summary" tab they scroll back through later.

Keep ONLY what they need to remember or do: deadlines, who owns what, amounts and counts,
decisions taken, anything broken or blocking, explicit instructions. Drop greetings,
opinions, hedging, thanks and small talk.

If the message contains nothing worth remembering - it is only chatter - return
{"title": null, "points": []}.

Quoted text inside the message is usually an EXAMPLE the sender is talking about, not an
instruction to the reader. Do not turn it into a point.

Return ONLY JSON: {"title": "<two or three words, Title Case>", "points": ["<a complete
phrase of three to fifteen words>", ...]}

Use as many points as the message needs - two if it says two things, eight if it says
eight. Never a bare list item or a single word. Use the message's own wording. Never
invent anything that is not in it."""


# A small model cannot hold five instructions at once, but it can copy a pattern. The
# zero-shot PROMPT above was written for a large model and handed unchanged to a 1.5B,
# which scored 13% recall - that is a prompt result as much as a model result (2026-10-09).
FEWSHOT = [
    ("Hi team, hope you're all well. The payment service is throwing 500s on checkout "
     "when a promo code is applied - Marco can you take it since you wrote that module? "
     "Also we need the migration script in before Friday noon or the release slips. "
     "Nothing else major, thanks everyone!",
     '{"title": "Checkout Bug", "points": ["Payment service returns 500s on checkout with '
     'a promo code", "Marco to take the checkout bug", "Migration script due before Friday '
     'noon or the release slips"]}'),
    ("morning all! that storm last night was wild, lost power for three hours. anyway I "
     "tried the new ramen place, honestly overrated. anyone around for lunch later?",
     '{"title": null, "points": []}'),
]

SHORT_PROMPT = """Summarize one chat message into a title and bullet points.

Keep only what the reader must remember or do: deadlines, owners, amounts, decisions,
anything broken. Drop greetings, opinions and small talk. If the message has nothing
worth remembering, return {"title": null, "points": []}.

Reply with JSON only."""


class Engine:
    name = "base"

    def summarize(self, text: str) -> Optional[dict]:
        raise NotImplementedError


class RulesEngine(Engine):
    name = "rules"

    def summarize(self, text):
        d = summarizer.summarize(text)
        return d.as_dict() if d else None


class LocalModelEngine(Engine):
    """A transformers model on this machine. Safe for dogfood - nothing leaves."""

    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=380,
                 fewshot=True):
        self.name = f"local:{model_id.split('/')[-1]}" + ("+fs" if fewshot else "")
        self.model_id = model_id
        # 220 was too tight: a truncated JSON fails to parse and the whole card is lost,
        # which scores as total recall failure rather than as the formatting problem it is.
        self.max_new_tokens = max_new_tokens
        self.fewshot = fewshot
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model.eval()
        self._pipe = (tok, model)

    def summarize(self, text):
        self._load()
        import torch
        tok, model = self._pipe
        if self.fewshot:
            messages = [{"role": "system", "content": SHORT_PROMPT}]
            for src, want in FEWSHOT:
                messages.append({"role": "user", "content": src})
                messages.append({"role": "assistant", "content": want})
            messages.append({"role": "user", "content": text})
        else:
            messages = [{"role": "system", "content": PROMPT},
                        {"role": "user", "content": f"Summarize this message:\n\n{text}"}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse(reply)


class Seq2SeqEngine(Engine):
    """A purpose-built abstractive summarizer - BART/T5 family. Local, safe for dogfood.

    The category I skipped when I jumped from rules to LLMs, and Akash caught it
    (2026-10-09). These are an order of magnitude smaller and faster than even a 0.5B
    instruct model - DistilBART-CNN-12-6 is 306M parameters, ~400MB, ~307ms - because
    they do one thing instead of everything. Crucially they are ABSTRACTIVE: they write
    new text rather than selecting sentences, which is the whole reason rules fail here.

    The catch is the output shape. A seq2seq model emits a prose paragraph and cannot be
    instructed, so it gives us no title and no bullets. We split its paragraph into
    bullets and take the title from the rules, which already get that to 9% generic on
    real traffic. The model does the half it is good at and the rules do the other.
    """

    def __init__(self, model_id="sshleifer/distilbart-cnn-12-6",
                 max_len=400, min_len=25):
        self.name = f"seq2seq:{model_id.split('/')[-1]}"
        self.model_id = model_id
        self.max_len, self.min_len = max_len, min_len
        self._pipe = None

    def _load(self):
        # transformers 5.x removed the "summarization" pipeline task, so the model is
        # driven directly. Same thing, one layer down.
        if self._pipe is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            model.eval()
            self._pipe = (tok, model)

    def summarize(self, text):
        if not summarizer.is_worth_summarizing(text):
            return None
        self._load()
        import torch
        tok, model = self._pipe
        ids = tok(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = model.generate(**ids, max_length=self.max_len, min_length=self.min_len,
                                 num_beams=4, do_sample=False, early_stopping=True)
        prose = tok.decode(out[0], skip_special_tokens=True).strip()
        if not prose:
            return None
        points = [summarizer._tidy_point(s) for s in summarizer.sentences(prose)]
        points = [p for p in points if p and summarizer._is_substantive(p)]
        if not points:
            return None
        return {"title": summarizer.title_for(prose), "points": points}


def production_summary(text, student=None):
    """What /message-summary serves. Rules decide IF, the student writes WHAT.

    The split is not a compromise, it is each half doing what it measurably does best.
    Rules drop 90.5% of pure chatter and 82 of 151 qualifying dogfood messages; the
    student rewrites instead of copying (5.2% verbatim against the rules' 60.7%, half the
    source length). Neither does the other's job well: the student alone cards every piece
    of small talk it is given, and the rules alone reprint sentences.

    Falls back to the rules card when the student is absent or returns nothing, so a
    missing model file degrades the feature instead of breaking the endpoint.
    """
    gate = summarizer.summarize(text)
    if gate is None:
        return None                      # nothing here worth a card
    if student is None:
        return gate.as_dict()
    try:
        card = student.summarize(text)
    except Exception:                    # noqa: BLE001 - a model fault must not 500
        return gate.as_dict()
    if not card or not card.get("points"):
        return gate.as_dict()
    # The rules write the better title: "Build 52" keeps the identifier where the model
    # gives "App Build Testing" and loses it.
    return {"title": gate.title, "points": card["points"]}


def default_student(model_dir=None):
    """The student if its weights are on disk, else None. Never raises."""
    path = Path(model_dir or os.environ.get("PAYCHAT_SUMMARY_MODEL")
                or (ROOT / "models" / "student_flan-t5-base"))
    if not (path / "config.json").exists():
        return None
    try:
        return StudentEngine(str(path))
    except Exception:                    # noqa: BLE001
        return None


class StudentEngine(Engine):
    """The distilled student - a flan-t5-base fine-tuned on the teacher's 1,796 pairs.

    This is the candidate that would actually ship: 250M parameters, ~1GB, CPU-only, no
    GPU on the box. Trained 2026-10-10 and chosen over a DistilBART-SAMSum student that
    scored HIGHER on recall (69.4% vs 58.6%) while hallucinating - "End Call Post-Bogan",
    "Recallation issues", an invented email address - and padding its cards with the same
    fact written four ways. The padding is what lifted its score: more points means more
    chances for a word-overlap check to hit. The cards decided it, not the number.

    Targets are flat text, not JSON: "Title | point | point". A 250M model would spend
    real capacity on brackets, and one bad character would lose the whole card. NONE means
    no card, which 20% of its training set taught it to say.
    """

    def __init__(self, model_dir="models/student_flan-t5-base",
                 prefix="summarize the chat message: "):
        self.name = "student"
        self.model_dir = str(ROOT / model_dir) if not Path(model_dir).is_absolute() else model_dir
        self.prefix = prefix
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            # Two threads, not "all of them". Measured 2026-10-11 on this model: 4 threads
            # 4.0s/card, 2 threads 2.8s. On a model this small the sync cost of more
            # threads exceeds what they buy, and the default of every logical core made
            # the dogfood run slower than it needed to be. It also leaves the other vCPU
            # for /classify, which is the path that has to stay at 69ms.
            torch.set_num_threads(int(os.environ.get("PAYCHAT_SUMMARY_THREADS", "2")))
            tok = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
            model.eval()
            self._pipe = (tok, model, torch)

    def summarize(self, text):
        self._load()
        tok, model, torch = self._pipe
        x = tok(self.prefix + (text or ""), max_length=512, truncation=True,
                return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**x, max_length=256, num_beams=4, early_stopping=True)
        return _parse_flat(tok.decode(out[0], skip_special_tokens=True))


_DEDUPE_OVERLAP = 0.7


def _same_point(a, b):
    """Two bullets saying the same thing, in different words.

    Exact repeats are only half of it. A seq2seq under beam search restates: the student
    wrote "Ensure 0 issues in positive flow" and "Ensure no issues before handover" on one
    card, and 7 of 42 real cards repeated something (dogfood, 2026-10-10). Compared on
    content words so a reworded repeat is caught too.
    """
    wa = {w for w in re.findall(r"[a-z0-9']+", a.lower()) if len(w) > 2}
    wb = {w for w in re.findall(r"[a-z0-9']+", b.lower()) if len(w) > 2}
    if not wa or not wb:
        return a.strip().lower() == b.strip().lower()
    return len(wa & wb) / len(wa | wb) >= _DEDUPE_OVERLAP


def _parse_flat(text):
    """"Title | point | point" -> a card. Anything malformed means no card, never a crash."""
    text = (text or "").strip()
    if not text or text.upper().startswith("NONE"):
        return None
    bits = [b.strip() for b in text.split("|") if b.strip()]
    if len(bits) < 2:
        return None
    title, points = bits[0], bits[1:]

    kept = []
    for p in points:
        # Structural guards the rules engine already has: a bullet that stops mid-phrase,
        # is only names, or is two words is not a point whatever wrote it.
        if len(p.split()) < 3 or summarizer._is_dangling(p) or summarizer._is_only_mentions(p):
            continue
        if any(_same_point(p, q) for q in kept):
            continue
        kept.append(p)
    if not kept:
        return None
    return {"title": title, "points": kept}


class HybridEngine(Engine):
    """Rules choose what matters; the model writes it. Local, safe for dogfood.

    The two engines fail in opposite directions, measured on 2026-10-09:

        rules    76.7% recall, 87.5% of chatter correctly dropped, 52% verbatim
        seq2seq  49.2% recall,  0.0% of chatter dropped,             6% verbatim

    Rules select well and write badly. The model writes well and selects badly - it
    self-limits to ~50 tokens whatever max_length says, because BART-CNN was trained on
    fixed-length news highlights, so a dense message loses content no matter what.

    So: rules pick the points and decide whether a card exists at all, then the model
    rewrites those points instead of the raw message. The model never sees the chatter,
    and never has to choose.
    """

    def __init__(self, model_id="philschmid/distilbart-cnn-12-6-samsum"):
        self.name = "hybrid"
        self._seq = Seq2SeqEngine(model_id, max_len=160, min_len=12)

    def summarize(self, text):
        card = summarizer.summarize(text)
        if card is None:
            return None                       # rules decided there is nothing here
        selected = ". ".join(p.rstrip(".") for p in card.points) + "."
        self._seq._load()
        import torch
        tok, model = self._seq._pipe
        ids = tok(selected, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = model.generate(**ids, max_length=self._seq.max_len,
                                 min_length=self._seq.min_len, num_beams=4,
                                 do_sample=False, early_stopping=True)
        prose = tok.decode(out[0], skip_special_tokens=True).strip()
        points = [summarizer._tidy_point(s) for s in summarizer.sentences(prose)]
        points = [p for p in points if p and summarizer._is_substantive(p)]
        if not points:
            return card.as_dict()             # model added nothing - keep the rules card
        return {"title": card.title, "points": points}


class ApiProxyEngine(Engine):
    """DeepSeek, as a stand-in for a competent instruct model. SYNTHETIC TEXT ONLY.

    Every message it is asked to summarize must be present in the generated corpus. A
    dogfood line is a real private conversation and is refused, loudly - this is the one
    place in the bake-off where a mistake would send someone's chat to a third party.
    """

    name = "api-proxy"

    def __init__(self, allowed_corpus: List[str]):
        self._allowed = {" ".join((t or "").split()) for t in allowed_corpus}
        self._client = None

    def _key(self):
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key and (ROOT / ".env").exists():
            for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
                if line.startswith("DEEPSEEK_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip("\"'")
        return key

    def summarize(self, text):
        if " ".join((text or "").split()) not in self._allowed:
            raise PermissionError(
                "ApiProxyEngine was given text that is not in the generated corpus. "
                "Real conversations never go to an API. Refusing.")
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._key(), base_url="https://api.deepseek.com/v1")
        try:
            r = self._client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": PROMPT},
                          {"role": "user", "content": f"Summarize this message:\n\n{text}"}],
                temperature=0, max_tokens=600,
                response_format={"type": "json_object"})
            return _parse(r.choices[0].message.content)
        except Exception as e:                      # noqa: BLE001
            print(f"    api call failed: {type(e).__name__}: {e}")
            return None


def _parse(reply: str) -> Optional[dict]:
    """The model's JSON -> a card, or None when it says there is nothing."""
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    points = [str(p).strip() for p in (data.get("points") or []) if str(p).strip()]
    title = data.get("title")
    if not points or not title:
        return None
    return {"title": str(title).strip(), "points": points}
