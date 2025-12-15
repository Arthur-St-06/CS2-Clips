#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from demoparser2 import DemoParser  # type: ignore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", default="out_sprays")
    ap.add_argument("--tickrate", type=float, default=64.0)
    ap.add_argument("--die-within-s", type=float, default=3.0)
    ap.add_argument("--min-bullets", type=int, default=6)
    ap.add_argument("--min-duration-s", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=20, help="how many rows to print")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bursts_path = out / "bursts.parquet"
    if not bursts_path.exists():
        raise FileNotFoundError(f"Missing {bursts_path}. Run step1_bursts.py first.")

    bursts = pd.read_parquet(bursts_path).copy()

    # Filter to spray-like bursts
    bursts = bursts[(bursts["bullets"] >= args.min_bullets) & (bursts["duration_s"] >= args.min_duration_s)].copy()
    bursts = bursts.sort_values(["player", "end_tick"]).reset_index(drop=True)

    parser = DemoParser(args.demo)
    deaths = parser.parse_events(["player_death"])
    deaths = deaths[0][1]

    # victim info: user_name/user_steamid at tick
    deaths_v = deaths[["tick", "user_name", "user_steamid"]].copy()
    deaths_v = deaths_v.rename(columns={"tick": "death_tick", "user_name": "player", "user_steamid": "steamid"})
    deaths_v["death_tick"] = deaths_v["death_tick"].astype(int)

    # kill info: attacker_name/attacker_steamid at tick (same player_death event)
    kills = deaths[["tick", "attacker_name", "attacker_steamid"]].copy()
    kills = kills.rename(columns={"tick": "kill_tick", "attacker_name": "player", "attacker_steamid": "steamid"})
    # attacker can be None (world), drop those
    kills = kills.dropna(subset=["player"])
    kills["kill_tick"] = kills["kill_tick"].astype(int)

    # Build quick per-player sorted lists for fast scanning
    deaths_by_player = {}
    for p, g in deaths_v.groupby("player"):
        deaths_by_player[p] = g["death_tick"].sort_values().tolist()

    kills_by_player = {}
    for p, g in kills.groupby("player"):
        kills_by_player[p] = g["kill_tick"].sort_values().tolist()

    die_within_ticks = int(args.die_within_s * args.tickrate)

    def first_death_after(player: str, tick: int, within_ticks: int):
        arr = deaths_by_player.get(player, [])
        for dt in arr:
            if dt >= tick and dt <= tick + within_ticks:
                return dt
            if dt > tick + within_ticks:
                break
        return None

    def has_kill_between(player: str, start_tick: int, end_tick: int) -> bool:
        arr = kills_by_player.get(player, [])
        for kt in arr:
            if kt < start_tick:
                continue
            if kt <= end_tick:
                return True
            break
        return False

    oversprays = []
    for row in bursts.itertuples(index=False):
        player = row.player
        start_tick = int(row.start_tick)
        end_tick = int(row.end_tick)

        death_tick = first_death_after(player, end_tick, die_within_ticks)
        if death_tick is None:
            continue

        # no kill from burst start until death
        if has_kill_between(player, start_tick, death_tick):
            continue

        oversprays.append({
            "player": player,
            "steamid": int(row.steamid),
            "weapon": row.weapon,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "bullets": int(row.bullets),
            "duration_s": float(row.duration_s),
            "death_tick": int(death_tick),
            "start_s": start_tick / args.tickrate,
            "end_s": end_tick / args.tickrate,
            "death_s": death_tick / args.tickrate,
            "time_to_death_after_burst_s": (death_tick - end_tick) / args.tickrate
        })

    df = pd.DataFrame(oversprays).sort_values(
        ["player", "bullets", "duration_s"],
        ascending=[True, False, False]
    )

    df.to_parquet(out / "overspray_candidates.parquet")

    print(f"Spray-like bursts: {len(bursts)}")
    print(f"Overspray candidates (die soon, no kill): {len(df)}")
    print(df.head(args.limit).to_string(index=False))

if __name__ == "__main__":
    main()

