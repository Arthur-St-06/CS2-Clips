#!/usr/bin/env python3
"""
orchestrator_batch.py - Batch orchestrator for CS2 clip generation

Analyzes ALL demos, collects clips for a specific player, and sends
a single batch request to Windows. CS2 launches only ONCE and processes
all clips from all demos sequentially.

Usage:
    # Analyze existing demos
    python orchestrator_batch.py --demo-dir ./demos --player Remag

    # Download demos first, then analyze
    python orchestrator_batch.py --demo-dir ./demos --player Remag --download --start-code CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx

    # Dry run (analyze only, don't send to Windows)
    python orchestrator_batch.py --demo-dir ./demos --player Remag --dry-run

    # Full pipeline with download
    python orchestrator_batch.py --demo-dir ./demos --player Remag --download --gc-version 2000696 --start-code CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx --max-matches 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from demoparser2 import DemoParser
except ImportError:
    print("ERROR: demoparser2 not installed. Run: pip install demoparser2")
    sys.exit(1)


# =========================
# Configuration
# =========================

@dataclass
class Config:
    demo_dir: Path = field(default_factory=lambda: Path("./demos"))
    out_dir: Path = field(default_factory=lambda: Path("./output"))
    inbox_dir: Path = field(default_factory=lambda: Path("./inbox"))
    
    # Analysis
    tickrate: float = 64.0
    max_gap_ticks: int = 8
    min_bullets: int = 6
    min_duration_s: float = 0.4
    die_within_s: float = 3.0
    
    # Clips
    clip_pre_s: float = 3.0
    clip_post_s: float = 2.0
    top_clips_per_demo: int = 10  # Max clips per demo
    
    # Network
    windows_base_url: str = "http://10.0.0.108:8788"
    ubuntu_base_url: str = "http://10.0.0.196:8787"
    clip_token: str = "token"
    
    # Demo download
    download_demos: bool = False
    gc_version: int = 2000696
    start_code: Optional[str] = None
    max_matches: int = 50
    
    # Required
    player_name: str = ""
    
    # Flags
    dry_run: bool = False


def load_config_from_env(cfg: Config) -> Config:
    cfg.windows_base_url = os.environ.get("WINDOWS_BASE_URL", cfg.windows_base_url).rstrip("/")
    cfg.ubuntu_base_url = os.environ.get("UBUNTU_BASE_URL", cfg.ubuntu_base_url).rstrip("/")
    cfg.clip_token = os.environ.get("CLIP_TOKEN", cfg.clip_token)
    
    if os.environ.get("GC_VERSION"):
        cfg.gc_version = int(os.environ["GC_VERSION"])
    if os.environ.get("TARGET_KNOWN_CODE"):
        cfg.start_code = os.environ["TARGET_KNOWN_CODE"]
    
    return cfg


# =========================
# Demo Download
# =========================

def download_demos(cfg: Config, log: logging.Logger) -> bool:
    """Download demos using demo_download.py / demo_download.py"""
    
    if not cfg.start_code:
        log.error("No start code provided. Set --start-code or TARGET_KNOWN_CODE in .env")
        return False
    
    log.info(f"Using start code: {cfg.start_code}")
    
    cmd = [
        "python", "demo_download.py",
        str(cfg.demo_dir),
        "--gc-version", str(cfg.gc_version),
        "--max-matches", str(cfg.max_matches),
        "--start-code", cfg.start_code,
    ]
    
    log.info(f"Downloading demos: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except FileNotFoundError:
        log.warning("steam_login.py not found, trying demo_download.py...")
        cmd[1] = "demo_download.py"
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode == 0
        except Exception as e:
            log.error(f"Demo download failed: {e}")
            return False
    except subprocess.CalledProcessError as e:
        log.error(f"Demo download failed with exit code {e.returncode}")
        return False


# =========================
# Logging
# =========================

def setup_logging(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger("orchestrator_batch")
    if not debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    return log


# =========================
# Demo Analysis
# =========================

def detect_bursts(
    demo_path: Path,
    tickrate: float,
    max_gap_ticks: int,
    min_bullets: int,
    player_filter: str,
) -> pd.DataFrame:
    parser = DemoParser(str(demo_path))
    
    try:
        wf = parser.parse_events(["weapon_fire"])
        wf = wf[0][1]
    except Exception as e:
        raise RuntimeError(f"Failed to parse weapon_fire: {e}")
    
    if wf.empty:
        return pd.DataFrame()
    
    wf = wf[["tick", "user_name", "user_steamid", "weapon"]].copy()
    wf = wf.sort_values(["user_name", "tick"])
    
    if player_filter:
        wf = wf[wf["user_name"].str.contains(player_filter, case=False, na=False)]
    
    if wf.empty:
        return pd.DataFrame()
    
    bursts = []
    for (player, steamid), shots in wf.groupby(["user_name", "user_steamid"]):
        ticks = shots["tick"].values
        weapons = shots["weapon"].values
        
        if len(ticks) == 0:
            continue
            
        start = ticks[0]
        last = ticks[0]
        bullets = 1
        weapon_counts = {weapons[0]: 1}
        
        for t, w in zip(ticks[1:], weapons[1:]):
            if t - last <= max_gap_ticks:
                bullets += 1
                weapon_counts[w] = weapon_counts.get(w, 0) + 1
            else:
                if bullets >= min_bullets:
                    weapon = max(weapon_counts, key=weapon_counts.get)
                    bursts.append({
                        "player": player,
                        "steamid": int(steamid),
                        "weapon": weapon,
                        "start_tick": int(start),
                        "end_tick": int(last),
                        "bullets": bullets,
                        "duration_s": (last - start) / tickrate
                    })
                start = t
                bullets = 1
                weapon_counts = {w: 1}
            last = t
        
        if bullets >= min_bullets:
            weapon = max(weapon_counts, key=weapon_counts.get)
            bursts.append({
                "player": player,
                "steamid": int(steamid),
                "weapon": weapon,
                "start_tick": int(start),
                "end_tick": int(last),
                "bullets": bullets,
                "duration_s": (last - start) / tickrate
            })
    
    return pd.DataFrame(bursts)


def detect_oversprays(
    demo_path: Path,
    bursts_df: pd.DataFrame,
    tickrate: float,
    die_within_s: float,
    min_bullets: int,
    min_duration_s: float,
) -> pd.DataFrame:
    if bursts_df.empty:
        return pd.DataFrame()
    
    bursts = bursts_df[
        (bursts_df["bullets"] >= min_bullets) & 
        (bursts_df["duration_s"] >= min_duration_s)
    ].copy()
    
    if bursts.empty:
        return pd.DataFrame()
    
    bursts = bursts.sort_values(["player", "end_tick"]).reset_index(drop=True)
    
    parser = DemoParser(str(demo_path))
    
    try:
        deaths = parser.parse_events(["player_death"])
        deaths = deaths[0][1]
    except Exception as e:
        raise RuntimeError(f"Failed to parse player_death: {e}")
    
    deaths_v = deaths[["tick", "user_name", "user_steamid"]].copy()
    deaths_v = deaths_v.rename(columns={"tick": "death_tick", "user_name": "player", "user_steamid": "steamid"})
    deaths_v["death_tick"] = deaths_v["death_tick"].astype(int)
    
    kills = deaths[["tick", "attacker_name", "attacker_steamid"]].copy()
    kills = kills.rename(columns={"tick": "kill_tick", "attacker_name": "player", "attacker_steamid": "steamid"})
    kills = kills.dropna(subset=["player"])
    kills["kill_tick"] = kills["kill_tick"].astype(int)
    
    deaths_by_player = {p: g["death_tick"].sort_values().tolist() for p, g in deaths_v.groupby("player")}
    kills_by_player = {p: g["kill_tick"].sort_values().tolist() for p, g in kills.groupby("player")}
    
    die_within_ticks = int(die_within_s * tickrate)
    
    def first_death_after(player, tick, within_ticks):
        for dt in deaths_by_player.get(player, []):
            if dt >= tick and dt <= tick + within_ticks:
                return dt
            if dt > tick + within_ticks:
                break
        return None
    
    def has_kill_between(player, start_tick, end_tick):
        for kt in kills_by_player.get(player, []):
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
            "start_s": start_tick / tickrate,
            "end_s": end_tick / tickrate,
            "death_s": death_tick / tickrate,
            "time_to_death_after_burst_s": (death_tick - end_tick) / tickrate
        })
    
    df = pd.DataFrame(oversprays)
    if not df.empty:
        df = df.sort_values(["bullets", "duration_s", "time_to_death_after_burst_s"], ascending=[False, False, True])
    return df


def analyze_demo(demo_path: Path, cfg: Config, log: logging.Logger) -> pd.DataFrame:
    """Analyze single demo, return overspray candidates for the target player"""
    log.info(f"  Analyzing: {demo_path.name}")
    
    bursts = detect_bursts(
        demo_path,
        tickrate=cfg.tickrate,
        max_gap_ticks=cfg.max_gap_ticks,
        min_bullets=cfg.min_bullets,
        player_filter=cfg.player_name,
    )
    
    if bursts.empty:
        log.info(f"    No bursts found for player '{cfg.player_name}'")
        return pd.DataFrame()
    
    oversprays = detect_oversprays(
        demo_path,
        bursts,
        tickrate=cfg.tickrate,
        die_within_s=cfg.die_within_s,
        min_bullets=cfg.min_bullets,
        min_duration_s=cfg.min_duration_s,
    )
    
    log.info(f"    Found {len(oversprays)} overspray candidates")
    return oversprays


# =========================
# Batch Request
# =========================

def send_batch_request(
    demos_data: list[dict],
    player_name: str,
    cfg: Config,
    log: logging.Logger,
) -> dict:
    """
    Send batch request to Windows with all demos and clips.
    
    demos_data format:
    [
        {
            "demo_id": "match123",
            "clips": [
                {"start_s": 10.0, "duration_s": 8.0},
                {"start_s": 50.5, "duration_s": 6.0},
            ]
        },
        ...
    ]
    """
    upload_url = cfg.ubuntu_base_url.rstrip("/") + "/upload"
    
    payload = {
        "username": player_name,
        "ubuntu_upload_url": upload_url,
        "demos": demos_data,
    }
    
    total_clips = sum(len(d["clips"]) for d in demos_data)
    log.info(f"Sending batch request: {len(demos_data)} demos, {total_clips} clips")
    
    if cfg.dry_run:
        log.info("[DRY RUN] Would send to Windows:")
        log.info(json.dumps(payload, indent=2))
        return {"dry_run": True, "demos": len(demos_data), "clips": total_clips}
    
    try:
        r = requests.post(
            cfg.windows_base_url + "/batch_clips",
            json=payload,
            headers={"X-Token": cfg.clip_token},
            timeout=3600,  # 1 hour timeout for large batches
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        log.error(f"Batch request failed: {e}")
        raise


def check_windows_server(cfg: Config, log: logging.Logger) -> bool:
    try:
        r = requests.get(
            cfg.windows_base_url + "/",
            headers={"X-Token": cfg.clip_token},
            timeout=10,
        )
        r.raise_for_status()
        log.info(f"Windows server OK: {cfg.windows_base_url}")
        return True
    except Exception as e:
        log.error(f"Cannot reach Windows server: {e}")
        return False


# =========================
# Main
# =========================

def run_batch_orchestrator(cfg: Config, log: logging.Logger) -> dict:
    log.info("=" * 60)
    log.info("CS2 Batch Clip Orchestrator")
    log.info("=" * 60)
    log.info(f"Player: {cfg.player_name}")
    log.info(f"Demo dir: {cfg.demo_dir}")
    log.info(f"Output dir: {cfg.out_dir}")
    log.info(f"Windows: {cfg.windows_base_url}")
    log.info(f"Ubuntu: {cfg.ubuntu_base_url}")
    log.info(f"Dry run: {cfg.dry_run}")
    log.info("=" * 60)
    
    # Ensure directories
    cfg.demo_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.inbox_dir.mkdir(parents=True, exist_ok=True)
    
    # Download demos if requested
    if cfg.download_demos:
        log.info("")
        log.info("Phase 0: Downloading demos from Steam...")
        if not download_demos(cfg, log):
            log.warning("Demo download failed or incomplete, continuing with existing demos...")
        log.info("")
    
    # Check Windows server
    if not cfg.dry_run:
        if not check_windows_server(cfg, log):
            return {"error": "Windows server unreachable"}
    
    # Find all demos
    demos = sorted(cfg.demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not demos:
        log.warning(f"No .dem files found in {cfg.demo_dir}")
        return {"error": "No demos found"}
    
    log.info(f"Found {len(demos)} demo files")
    log.info("")
    
    # Analyze all demos
    log.info("Phase 1: Analyzing all demos...")
    demos_data = []
    total_candidates = 0
    
    for demo_path in demos:
        try:
            oversprays = analyze_demo(demo_path, cfg, log)
        except Exception as e:
            log.error(f"    Failed to analyze {demo_path.name}: {e}")
            continue
        
        if oversprays.empty:
            continue
        
        # Save analysis
        demo_out = cfg.out_dir / demo_path.stem
        demo_out.mkdir(parents=True, exist_ok=True)
        oversprays.to_parquet(demo_out / "overspray_candidates.parquet")
        
        # Take top N clips
        top = oversprays.head(cfg.top_clips_per_demo)
        
        clips = []
        for _, row in top.iterrows():
            start_s = max(0.0, row["start_s"] - cfg.clip_pre_s)
            end_s = row["death_s"] + cfg.clip_post_s
            duration_s = end_s - start_s
            clips.append({
                "start_s": round(start_s, 3),
                "duration_s": round(duration_s, 3),
            })
        
        if clips:
            demos_data.append({
                "demo_id": demo_path.stem,
                "clips": clips,
            })
            total_candidates += len(clips)
    
    log.info("")
    log.info(f"Analysis complete: {len(demos_data)} demos with clips, {total_candidates} total clips")
    
    if not demos_data:
        log.warning("No clips to generate")
        return {"demos_analyzed": len(demos), "demos_with_clips": 0, "total_clips": 0}
    
    # Send batch request
    log.info("")
    log.info("Phase 2: Sending batch request to Windows...")
    
    try:
        result = send_batch_request(demos_data, cfg.player_name, cfg, log)
    except Exception as e:
        return {"error": str(e)}
    
    log.info("")
    log.info("=" * 60)
    log.info("COMPLETE")
    log.info("=" * 60)
    log.info(f"Demos analyzed: {len(demos)}")
    log.info(f"Demos with clips: {len(demos_data)}")
    log.info(f"Total clips: {total_candidates}")
    log.info(f"Clips will arrive in: {cfg.inbox_dir}")
    
    return {
        "demos_analyzed": len(demos),
        "demos_with_clips": len(demos_data),
        "total_clips": total_candidates,
        "windows_response": result,
    }


def main():
    load_dotenv()
    
    ap = argparse.ArgumentParser(description="Batch CS2 clip orchestrator - single CS2 launch for all demos")
    
    ap.add_argument("--demo-dir", type=Path, default=Path("./demos"), help="Directory with .dem files")
    ap.add_argument("--out", type=Path, default=Path("./output"), help="Output directory")
    ap.add_argument("--inbox", type=Path, default=Path("./inbox"), help="Where clips arrive")
    
    ap.add_argument("--player", required=True, help="Player name to filter (e.g., Remag)")
    
    ap.add_argument("--tickrate", type=float, default=64.0)
    ap.add_argument("--min-bullets", type=int, default=6)
    ap.add_argument("--die-within", type=float, default=3.0)
    ap.add_argument("--top-per-demo", type=int, default=10, help="Max clips per demo")
    ap.add_argument("--pre", type=float, default=3.0, help="Seconds before burst")
    ap.add_argument("--post", type=float, default=2.0, help="Seconds after death")
    
    ap.add_argument("--windows-url", default="http://10.0.0.108:8788")
    ap.add_argument("--ubuntu-url", default="http://10.0.0.196:8787")
    
    # Demo download options
    ap.add_argument("--download", action="store_true", help="Download demos from Steam first")
    ap.add_argument("--gc-version", type=int, default=2000696, help="GC version for Steam")
    ap.add_argument("--start-code", help="Starting sharecode (CSGO-xxxxx-xxxxx-...)")
    ap.add_argument("--max-matches", type=int, default=50, help="Max matches to download")
    
    ap.add_argument("--dry-run", action="store_true", help="Analyze only, don't send request")
    ap.add_argument("--debug", action="store_true")
    
    args = ap.parse_args()
    
    cfg = Config()
    cfg = load_config_from_env(cfg)
    
    cfg.demo_dir = args.demo_dir.expanduser().resolve()
    cfg.out_dir = args.out.expanduser().resolve()
    cfg.inbox_dir = args.inbox.expanduser().resolve()
    cfg.player_name = args.player
    cfg.tickrate = args.tickrate
    cfg.min_bullets = args.min_bullets
    cfg.die_within_s = args.die_within
    cfg.top_clips_per_demo = args.top_per_demo
    cfg.clip_pre_s = args.pre
    cfg.clip_post_s = args.post
    cfg.windows_base_url = args.windows_url.rstrip("/")
    cfg.ubuntu_base_url = args.ubuntu_url.rstrip("/")
    cfg.download_demos = args.download
    cfg.gc_version = args.gc_version
    cfg.start_code = args.start_code or cfg.start_code  # Don't overwrite env with None
    cfg.max_matches = args.max_matches
    cfg.dry_run = args.dry_run
    
    log = setup_logging(args.debug)
    
    result = run_batch_orchestrator(cfg, log)
    
    if "error" in result:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
