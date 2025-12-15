#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser  # type: ignore

TICKRATE = 64.0          # good default for CS2 demos
MAX_GAP_TICKS = int(0.12 * TICKRATE)   # 120ms between shots
MIN_BULLETS = 6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", default="out_sprays")
    ap.add_argument("--player", default="", help="Filter by player name (optional)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parser = DemoParser(args.demo)
    wf = parser.parse_events(["weapon_fire"])
    wf = wf[0][1]

    # Basic cleanup
    wf = wf[["tick", "user_name", "user_steamid", "weapon"]].copy()
    wf = wf.sort_values(["user_name", "tick"])

    if args.player:
        wf = wf[wf["user_name"].str.contains(args.player, case=False, na=False)]

    bursts = []

    for (player, steamid), shots in wf.groupby(["user_name", "user_steamid"]):
        ticks = shots["tick"].values
        weapons = shots["weapon"].values

        start = ticks[0]
        last = ticks[0]
        bullets = 1
        weapon_counts = {weapons[0]: 1}

        for t, w in zip(ticks[1:], weapons[1:]):
            if t - last <= MAX_GAP_TICKS:
                bullets += 1
                weapon_counts[w] = weapon_counts.get(w, 0) + 1
            else:
                if bullets >= MIN_BULLETS:
                    weapon = max(weapon_counts, key=weapon_counts.get)
                    bursts.append({
                        "player": player,
                        "steamid": int(steamid),
                        "weapon": weapon,
                        "start_tick": int(start),
                        "end_tick": int(last),
                        "bullets": bullets,
                        "duration_s": (last - start) / TICKRATE
                    })
                start = t
                bullets = 1
                weapon_counts = {w: 1}
            last = t

        # flush last
        if bullets >= MIN_BULLETS:
            weapon = max(weapon_counts, key=weapon_counts.get)
            bursts.append({
                "player": player,
                "steamid": int(steamid),
                "weapon": weapon,
                "start_tick": int(start),
                "end_tick": int(last),
                "bullets": bullets,
                "duration_s": (last - start) / TICKRATE
            })

    bursts_df = pd.DataFrame(bursts)
    bursts_df.to_parquet(out / "bursts.parquet")

    print(f"Detected {len(bursts_df)} bursts")
    print(bursts_df.head(10))

if __name__ == "__main__":
    main()

