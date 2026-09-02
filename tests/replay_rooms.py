"""Replay whole rooms from a dogfood log and compare prompts PER ROOM, not per message.

tools/replay_log_rooms.py compares message by message, which reads a fix as a regression.
On 2026-08-31 it reported dm_10_4 losing two prompts and called it a regression; the room
had actually gone from two prompts on the wrong messages to one on the right one, which
is the correct outcome. A user does not experience messages, they experience "did the
payment sheet appear in this conversation" - so that is the unit.

What matters, in order:

  rooms that went to ZERO    a conversation that used to work and now does not. The only
                             unambiguous regression, and the criterion to hold hardest.
  rooms with fewer prompts    could be a fix collapsing duplicates. Read them.
  total prompts               context, not a verdict.

Test rooms are excluded by prefix - an earlier audit read 9,721 of 12,112 messages from
replay rooms this repo's own tooling had created.

    python tests/replay_rooms.py --log data/eval/dogfood_2026-08-31.jsonl \
                                 --url http://127.0.0.1:8900/detect
"""
import argparse, collections, io, json, sys, time
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEST_PREFIXES = ("g15_", "df_", "si_", "rv_", "pre_", "gate_", "echo_", "rq_", "tri_",
                 "cy_", "cz_", "rc_", "dbg", "rep_", "rr_", "idchk", "ov_", "oa_",
                 "of_", "basic_", "bm_", "mp_", "snd_", "doc_", "dc_", "nat_", "po_",
                 "pa_", "bv_", "vc_", "t85_", "tc", "L", "padchk_")
MR = ("money", "ride")


def load_rooms(path):
    rooms = collections.OrderedDict()
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        rid = str(r.get("room") or r.get("room_id") or "")
        if rid and not rid.startswith(TEST_PREFIXES):
            rooms.setdefault(rid, []).append(r)
    return rooms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()

    rooms = load_rooms(a.log)
    tag = int(time.time())
    then_c, now_c = {}, {}
    for rid, msgs in rooms.items():
        then_c[rid] = sum(1 for m in msgs
                          if any(x in MR for x in (m.get("fired") or [])))
        rm = f"rpl{abs(hash(a.log + rid)) % 999999}"
        n = 0
        for i, m in enumerate(msgs):
            try:
                d = requests.post(a.url, timeout=60, json={
                    "text": m.get("text") or "", "room_id": rm,
                    "sender": str(m.get("sender")),
                    "message_id": f"{tag}_{abs(hash(rid)) % 99999}_{i}"}).json()
            except Exception:
                continue
            if [x for x in (d.get("intents") or []) if x in MR]:
                n += 1
        now_c[rid] = n

    lost = [(r, then_c[r], now_c[r]) for r in rooms if now_c[r] < then_c[r]]
    gained = [(r, then_c[r], now_c[r]) for r in rooms if now_c[r] > then_c[r]]
    zeroed = [(r, t, n) for r, t, n in lost if n == 0 and t > 0]

    name = Path(a.log).name
    print(f"\n  === {name} ===")
    print(f"  rooms {len(rooms)}   prompts then {sum(then_c.values())} "
          f"-> now {sum(now_c.values())}")
    print(f"  rooms with more prompts  : {len(gained)}")
    print(f"  rooms with fewer prompts : {len(lost)}")
    for r, t, n in sorted(lost, key=lambda x: x[2] - x[1])[:a.show]:
        print(f"      {r}: {t} -> {n}")
    print(f"  rooms that went to ZERO  : {len(zeroed)}   <- the only hard failure")
    for r, t, n in zeroed:
        print(f"      REGRESSION {r}: {t} -> 0")
    return 1 if zeroed else 0


if __name__ == "__main__":
    raise SystemExit(main())
