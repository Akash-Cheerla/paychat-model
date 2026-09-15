"""Replay one dogfood log through TWO live servers and diff them per room and per message.

replay_rooms.py compares a server against the prompts recorded IN the log. That is the
right check for "did the room stop working", but the recorded prompts come from whatever
model happened to be live when the log was written - which over a three-week log is more
than one model. For "what does v14 fix relative to v5" the only clean comparison is to
put the identical traffic through both and read the differences.

Rooms are replayed whole and in order, with fresh room ids per arm, so neither server
sees state the other does not.

    python replay_ab.py --log <path> --a name=url --b name=url
"""
import argparse, collections, io, json, re, sys, time
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
MR = ("money", "ride")
TESTP = ("nw_", "rw_", "g15_", "df_", "si_", "rv_", "pre_", "gate_", "echo_", "rq_",
         "tri_", "cy_", "cz_", "rc_", "dbg", "rep_", "rr_", "idchk", "swda_", "val_",
         "inv_", "trace_", "basic_", "bm_", "ov_", "oa_", "cov_", "cx_", "bv_", "tr_",
         "dm_verify", "group_verify", "dm_fix", "dm_selfack", "cg_", "dst_", "vfy_",
         "spec_", "padchk_", "mp_", "snd_", "doc_", "dc_", "nat_", "po_", "pa_", "vc_",
         "t85_", "of_", "L")


def load(path):
    rooms = collections.OrderedDict()
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        rid = str(r.get("room") or "")
        if not rid or rid.startswith(TESTP):
            continue
        if not (r.get("text") or "").strip():
            continue
        rooms.setdefault(rid, []).append(r)
    return rooms


def play(url, rid, msgs, tag):
    """Return the list of (index, sender, text, intents) that produced a prompt."""
    room = f"ab_{tag}_{abs(hash(rid)) % 10**8}"
    out = []
    for i, m in enumerate(msgs):
        body = {"text": m.get("text") or "", "room_id": room,
                "sender": str(m.get("sender")), "message_id": f"{tag}_{i}"}
        if m.get("participants"):
            body["participants"] = m["participants"]
        try:
            d = requests.post(url, json=body, timeout=90).json()
        except Exception:
            d = {}
        g = [x for x in (d.get("intents") or []) if x in MR]
        if g:
            out.append((i, str(m.get("sender")), m.get("text") or "", g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--a", required=True, help="name=url  (the baseline)")
    ap.add_argument("--b", required=True, help="name=url  (the candidate)")
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    an, au = a.a.split("=", 1)
    bn, bu = a.b.split("=", 1)

    rooms = load(a.log)
    n = sum(len(v) for v in rooms.values())
    print(f"  {len(rooms)} real rooms, {n:,} messages")
    print(f"  A = {an} ({au})")
    print(f"  B = {bn} ({bu})\n")

    tag = int(time.time())
    res = {}
    for k, (rid, msgs) in enumerate(rooms.items()):
        res[rid] = (play(au, rid, msgs, tag + k),
                    play(bu, rid, msgs, tag + 100000 + k))
        if (k + 1) % 10 == 0:
            print(f"    {k+1}/{len(rooms)} rooms")

    ta = sum(len(v[0]) for v in res.values())
    tb = sum(len(v[1]) for v in res.values())
    zero_a = sum(1 for v in res.values() if v[0] and not v[1])
    zero_b = sum(1 for v in res.values() if v[1] and not v[0])
    print(f"\n  total prompts   {an}: {ta}    {bn}: {tb}")
    print(f"  rooms that had prompts in {an} and none in {bn}: {zero_a}")
    print(f"  rooms that had prompts in {bn} and none in {an}: {zero_b}")

    # per-message diff, keyed on the message index within the room
    only_a, only_b, both = [], [], 0
    for rid, (A, B) in res.items():
        ia = {(i, tuple(g)) for i, _, _, g in A}
        ib = {(i, tuple(g)) for i, _, _, g in B}
        ta_ = {i for i, _ in ia}
        tb_ = {i for i, _ in ib}
        for i, s, t, g in A:
            if i not in tb_:
                only_a.append((rid, i, s, t, g))
        for i, s, t, g in B:
            if i not in ta_:
                only_b.append((rid, i, s, t, g))
        both += len(ta_ & tb_)
    print(f"\n  fired in both                 : {both}")
    print(f"  only {an:<6} (B dropped it)      : {len(only_a)}")
    print(f"  only {bn:<6} (B added it)        : {len(only_b)}")

    ctx = {rid: msgs for rid, msgs in rooms.items()}

    def dump(title, rows):
        print(f"\n  === {title} ({len(rows)}) ===")
        for rid, i, s, t, g in rows[:a.show]:
            prev = ctx[rid][i - 1].get("text", "") if i else ""
            print(f"\n    [{rid} #{i}] {g}")
            if prev:
                print(f"        prev  {str(prev)[:66]!r}")
            print(f"        -->   {s}: {str(t)[:66]!r}")

    dump(f"prompts {an} shows and {bn} does NOT", only_a)
    dump(f"prompts {bn} shows and {an} does NOT", only_b)

    if a.save:
        Path(a.save).write_text(json.dumps(
            {"only_a": only_a, "only_b": only_b,
             "totals": {an: ta, bn: tb, "both": both}}, indent=1,
            ensure_ascii=False), encoding="utf-8")
        print(f"\n  wrote {a.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
