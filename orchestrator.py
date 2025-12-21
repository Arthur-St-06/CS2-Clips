#!/usr/bin/env python3
"""
orchestrator.py - Combines demo downloading, analysis, and clip generation

Flow:
  1. [Optional] Download demos from Steam
  2. Analyze demos on Ubuntu (detect overspray candidates)
  3. Send clip requests to Windows
  4. Windows creates clips and uploads them back to Ubuntu

Usage:
  # Analyze existing demos and create clips for all players
  python orchestrator.py --demo-dir ./demos --out ./output

  # Analyze for a specific player
  python orchestrator.py --demo-dir ./demos --out ./output --player "YourName"

  # Analyze a single demo
  python orchestrator.py --demo ./demos/match123.dem --out ./output

  # Download new demos first, then analyze
  python orchestrator.py --download --gc-version 2000696 --demo-dir ./demos --out ./output

  # Dry run (analyze but don't send clip requests)
  python orchestrator.py --demo-dir ./demos --out ./output --dry-run

Requirements:
  - Ubuntu: demoparser2, pandas, requests, python-dotenv
  - Windows clip server running (clip_server.py)
  - Ubuntu receiver running (ubuntu_node.py serve)
  - Demos must exist on BOTH Ubuntu (for analysis) and Windows (for clip generation)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# Try to import demoparser2 - required for analysis
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
    # Directories
    demo_dir: Path = field(default_factory=lambda: Path("./demos"))
    out_dir: Path = field(default_factory=lambda: Path("./output"))
    inbox_dir: Path = field(default_factory=lambda: Path("./inbox"))
    
    # Analysis parameters
    tickrate: float = 64.0
    max_gap_ticks: int = 8  # ~120ms at 64 tick
    min_bullets: int = 6
    min_duration_s: float = 0.4
    die_within_s: float = 3.0
    
    # Clip parameters
    clip_pre_s: float = 3.0   # seconds before burst start
    clip_post_s: float = 2.0  # seconds after death
    top_clips: int = 5        # max clips per demo
    
    # Network
    windows_base_url: str = "http://10.0.0.108:8788"
    ubuntu_base_url: str = "http://10.0.0.196:8787"
    clip_token: str = "token"
    
    # Demo download (optional)
    gc_version: int = 2000696
    start_code: Optional[str] = None
    max_matches: int = 50
    
    # Player filter
    player_filter: str = ""
    
    # Flags
    dry_run: bool = False
    skip_analyzed: bool = True
    download_demos: bool = False


def load_config_from_env(cfg: Config) -> Config:
    """Override config from environment variables"""
    cfg.windows_base_url = os.environ.get("WINDOWS_BASE_URL", cfg.windows_base_url).rstrip("/")
    cfg.ubuntu_base_url = os.environ.get("UBUNTU_BASE_URL", cfg.ubuntu_base_url).rstrip("/")
    cfg.clip_token = os.environ.get("CLIP_TOKEN", cfg.clip_token)
    
    if os.environ.get("GC_VERSION"):
        cfg.gc_version = int(os.environ["GC_VERSION"])
    if os.environ.get("TARGET_KNOWN_CODE"):
        cfg.start_code = os.environ["TARGET_KNOWN_CODE"]
    
    return cfg


# =========================
# Logging
# =========================

def setup_logging(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("orchestrator")
    if not debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    return log


# =========================
# Demo Analysis (from step1 + step2)
# =========================

def detect_bursts(
    demo_path: Path,
    tickrate: float = 64.0,
    max_gap_ticks: int = 8,
    min_bullets: int = 6,
    player_filter: str = "",
) -> pd.DataFrame:
    """Extract shooting bursts from demo (step1_bursts logic)"""
    parser = DemoParser(str(demo_path))
    
    try:
        wf = parser.parse_events(["weapon_fire"])
        wf = wf[0][1]
    except Exception as e:
        raise RuntimeError(f"Failed to parse weapon_fire events: {e}")
    
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
        
        # Flush last burst
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
    tickrate: float = 64.0,
    die_within_s: float = 3.0,
    min_bullets: int = 6,
    min_duration_s: float = 0.4,
) -> pd.DataFrame:
    """Find overspray candidates (step2_overspray logic)"""
    if bursts_df.empty:
        return pd.DataFrame()
    
    # Filter to spray-like bursts
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
        raise RuntimeError(f"Failed to parse player_death events: {e}")
    
    # Victim deaths
    deaths_v = deaths[["tick", "user_name", "user_steamid"]].copy()
    deaths_v = deaths_v.rename(columns={
        "tick": "death_tick", 
        "user_name": "player", 
        "user_steamid": "steamid"
    })
    deaths_v["death_tick"] = deaths_v["death_tick"].astype(int)
    
    # Kills (attacker info)
    kills = deaths[["tick", "attacker_name", "attacker_steamid"]].copy()
    kills = kills.rename(columns={
        "tick": "kill_tick", 
        "attacker_name": "player", 
        "attacker_steamid": "steamid"
    })
    kills = kills.dropna(subset=["player"])
    kills["kill_tick"] = kills["kill_tick"].astype(int)
    
    # Build lookup structures
    deaths_by_player = {}
    for p, g in deaths_v.groupby("player"):
        deaths_by_player[p] = g["death_tick"].sort_values().tolist()
    
    kills_by_player = {}
    for p, g in kills.groupby("player"):
        kills_by_player[p] = g["kill_tick"].sort_values().tolist()
    
    die_within_ticks = int(die_within_s * tickrate)
    
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
        
        # No kill from burst start until death
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
        df = df.sort_values(
            ["bullets", "duration_s", "time_to_death_after_burst_s"],
            ascending=[False, False, True]
        )
    
    return df


def analyze_demo(
    demo_path: Path,
    cfg: Config,
    log: logging.Logger,
) -> pd.DataFrame:
    """Full analysis pipeline for a single demo"""
    log.info(f"Analyzing: {demo_path.name}")
    
    # Step 1: Detect bursts
    bursts = detect_bursts(
        demo_path,
        tickrate=cfg.tickrate,
        max_gap_ticks=cfg.max_gap_ticks,
        min_bullets=cfg.min_bullets,
        player_filter=cfg.player_filter,
    )
    log.info(f"  Detected {len(bursts)} bursts")
    
    if bursts.empty:
        return pd.DataFrame()
    
    # Step 2: Find oversprays
    oversprays = detect_oversprays(
        demo_path,
        bursts,
        tickrate=cfg.tickrate,
        die_within_s=cfg.die_within_s,
        min_bullets=cfg.min_bullets,
        min_duration_s=cfg.min_duration_s,
    )
    log.info(f"  Found {len(oversprays)} overspray candidates")
    
    return oversprays


# =========================
# Clip Request (from ubuntu_node.py)
# =========================

def request_clips(
    demo_id: str,
    username: str,
    clips: list[dict],
    cfg: Config,
    log: logging.Logger,
) -> dict:
    """Send clip request to Windows server"""
    upload_url = cfg.ubuntu_base_url.rstrip("/") + "/upload"
    
    payload = {
        "demo_id": demo_id,
        "username": username,
        "ubuntu_upload_url": upload_url,
        "clips": [{"start_s": float(c["start_s"]), "duration_s": float(c["duration_s"])} for c in clips],
    }
    
    log.info(f"  Requesting {len(clips)} clips for {username} from {demo_id}")
    log.debug(f"  Payload: {json.dumps(payload, indent=2)}")
    
    if cfg.dry_run:
        log.info("  [DRY RUN] Would send request to Windows")
        return {"dry_run": True, "clips_count": len(clips)}
    
    try:
        r = requests.post(
            cfg.windows_base_url + "/clip",
            json=payload,
            headers={"X-Token": cfg.clip_token},
            timeout=600,  # 10 min timeout for multiple clips
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        log.error(f"  Failed to request clips: {e}")
        raise


def check_windows_server(cfg: Config, log: logging.Logger) -> bool:
    """Check if Windows clip server is reachable"""
    try:
        r = requests.get(
            cfg.windows_base_url + "/demos",
            headers={"X-Token": cfg.clip_token},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        log.info(f"Windows server OK. Available demos: {data.get('demo_ids', [])}")
        return True
    except Exception as e:
        log.error(f"Cannot reach Windows server at {cfg.windows_base_url}: {e}")
        return False


def get_available_demo_ids(cfg: Config) -> set[str]:
    """Get list of demo IDs available on Windows server"""
    try:
        r = requests.get(
            cfg.windows_base_url + "/demos",
            headers={"X-Token": cfg.clip_token},
            timeout=10,
        )
        r.raise_for_status()
        return set(r.json().get("demo_ids", []))
    except Exception:
        return set()


# =========================
# Demo ID mapping
# =========================

def demo_path_to_id(demo_path: Path) -> str:
    """
    Convert demo path to demo_id that Windows server understands.
    
    The Windows server uses demo stem (filename without extension) as ID,
    or custom mappings in DEMO_MAP.
    """
    return demo_path.stem


def find_matching_demo_id(demo_path: Path, available_ids: set[str]) -> Optional[str]:
    """Try to find a matching demo_id on Windows for this demo"""
    stem = demo_path.stem
    
    # Direct match
    if stem in available_ids:
        return stem
    
    # Try partial matches (e.g., "demo392" might match longer filename)
    for aid in available_ids:
        if aid in stem or stem in aid:
            return aid
    
    return None


# =========================
# Demo Download (simplified from demo_download.py)
# =========================

def download_demos(cfg: Config, log: logging.Logger) -> list[Path]:
    """
    Download demos using demo_download.py subprocess.
    Returns list of downloaded demo paths.
    """
    cmd = [
        "python3", "demo_download.py",
        str(cfg.demo_dir),
        "--gc-version", str(cfg.gc_version),
        "--max-matches", str(cfg.max_matches),
    ]
    
    if cfg.start_code:
        cmd += ["--start-code", cfg.start_code]
    
    log.info(f"Running demo download: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log.error(f"Demo download failed with exit code {e.returncode}")
        raise
    
    # Return all .dem files in demo_dir
    demos = sorted(cfg.demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
    return demos


# =========================
# Main Orchestrator
# =========================

def process_demo(
    demo_path: Path,
    cfg: Config,
    log: logging.Logger,
    available_demo_ids: set[str],
) -> dict:
    """Process a single demo: analyze and request clips"""
    result = {
        "demo": demo_path.name,
        "status": "unknown",
        "candidates": 0,
        "clips_requested": 0,
        "error": None,
    }
    
    # Find matching demo_id on Windows
    demo_id = find_matching_demo_id(demo_path, available_demo_ids)
    if demo_id is None:
        log.warning(f"  No matching demo_id on Windows for {demo_path.name}")
        log.warning(f"  Available IDs: {sorted(available_demo_ids)[:10]}...")
        result["status"] = "skip_no_windows_demo"
        return result
    
    log.info(f"  Matched to Windows demo_id: {demo_id}")
    
    # Output directory for this demo
    demo_out = cfg.out_dir / demo_path.stem
    demo_out.mkdir(parents=True, exist_ok=True)
    
    # Check if already analyzed
    candidates_path = demo_out / "overspray_candidates.parquet"
    if cfg.skip_analyzed and candidates_path.exists():
        try:
            existing = pd.read_parquet(candidates_path)
            if not existing.empty:
                log.info(f"  Already analyzed ({len(existing)} candidates). Use --no-skip to re-analyze.")
                result["status"] = "skip_already_analyzed"
                result["candidates"] = len(existing)
                return result
        except Exception:
            pass
    
    # Analyze
    try:
        oversprays = analyze_demo(demo_path, cfg, log)
    except Exception as e:
        log.error(f"  Analysis failed: {e}")
        result["status"] = "error_analysis"
        result["error"] = str(e)
        return result
    
    # Save analysis results
    oversprays.to_parquet(candidates_path)
    result["candidates"] = len(oversprays)
    
    if oversprays.empty:
        log.info("  No overspray candidates found")
        result["status"] = "ok_no_candidates"
        return result
    
    # Prepare clips (top N by severity)
    top = oversprays.head(cfg.top_clips)
    
    # Group by player and request clips
    clips_requested = 0
    for player, player_clips in top.groupby("player"):
        clips = []
        for _, row in player_clips.iterrows():
            start_s = max(0.0, row["start_s"] - cfg.clip_pre_s)
            end_s = row["death_s"] + cfg.clip_post_s
            duration_s = end_s - start_s
            clips.append({
                "start_s": start_s,
                "duration_s": duration_s,
            })
        
        try:
            resp = request_clips(demo_id, player, clips, cfg, log)
            clips_requested += len(clips)
            log.info(f"  Clips requested for {player}: {resp}")
        except Exception as e:
            log.error(f"  Failed to request clips for {player}: {e}")
    
    result["clips_requested"] = clips_requested
    result["status"] = "ok"
    return result


def run_orchestrator(cfg: Config, log: logging.Logger) -> list[dict]:
    """Main orchestrator loop"""
    log.info("=" * 60)
    log.info("CS2 Clip Orchestrator")
    log.info("=" * 60)
    log.info(f"Demo dir: {cfg.demo_dir}")
    log.info(f"Output dir: {cfg.out_dir}")
    log.info(f"Windows server: {cfg.windows_base_url}")
    log.info(f"Ubuntu receiver: {cfg.ubuntu_base_url}")
    log.info(f"Player filter: {cfg.player_filter or '(all players)'}")
    log.info(f"Dry run: {cfg.dry_run}")
    log.info("=" * 60)
    
    # Ensure directories exist
    cfg.demo_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.inbox_dir.mkdir(parents=True, exist_ok=True)
    
    # Download demos if requested
    if cfg.download_demos:
        log.info("Downloading demos...")
        try:
            download_demos(cfg, log)
        except Exception as e:
            log.error(f"Demo download failed: {e}")
            if not any(cfg.demo_dir.glob("*.dem")):
                return []
    
    # Check Windows server
    if not cfg.dry_run:
        if not check_windows_server(cfg, log):
            log.error("Cannot proceed without Windows server connection")
            return []
    
    # Get available demo IDs from Windows
    available_demo_ids = get_available_demo_ids(cfg)
    if not cfg.dry_run and not available_demo_ids:
        log.warning("No demos available on Windows server!")
    
    # Find demos to process
    demos = sorted(cfg.demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not demos:
        log.warning(f"No .dem files found in {cfg.demo_dir}")
        return []
    
    log.info(f"Found {len(demos)} demo files")
    
    # Process each demo
    results = []
    for i, demo_path in enumerate(demos, 1):
        log.info(f"\n[{i}/{len(demos)}] Processing: {demo_path.name}")
        result = process_demo(demo_path, cfg, log, available_demo_ids)
        results.append(result)
    
    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    
    total_candidates = sum(r["candidates"] for r in results)
    total_clips = sum(r["clips_requested"] for r in results)
    ok_count = sum(1 for r in results if r["status"].startswith("ok"))
    skip_count = sum(1 for r in results if r["status"].startswith("skip"))
    error_count = sum(1 for r in results if r["status"].startswith("error"))
    
    log.info(f"Demos processed: {len(results)}")
    log.info(f"  OK: {ok_count}")
    log.info(f"  Skipped: {skip_count}")
    log.info(f"  Errors: {error_count}")
    log.info(f"Total overspray candidates: {total_candidates}")
    log.info(f"Total clips requested: {total_clips}")
    log.info(f"Clips will arrive in: {cfg.inbox_dir}")
    
    return results


# =========================
# CLI
# =========================

def main():
    load_dotenv()
    
    ap = argparse.ArgumentParser(
        description="CS2 Clip Orchestrator - Analyze demos and create coaching clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input sources
    ap.add_argument("--demo", type=Path, help="Single demo file to process")
    ap.add_argument("--demo-dir", type=Path, default=Path("./demos"), 
                    help="Directory containing .dem files (default: ./demos)")
    
    # Output
    ap.add_argument("--out", type=Path, default=Path("./output"),
                    help="Output directory for analysis results (default: ./output)")
    ap.add_argument("--inbox", type=Path, default=Path("./inbox"),
                    help="Directory where clips will be received (default: ./inbox)")
    
    # Analysis parameters
    ap.add_argument("--player", default="", help="Filter to specific player name")
    ap.add_argument("--tickrate", type=float, default=64.0, help="Demo tickrate (default: 64)")
    ap.add_argument("--min-bullets", type=int, default=6, help="Min bullets for burst (default: 6)")
    ap.add_argument("--die-within", type=float, default=3.0, 
                    help="Max seconds after burst to count as overspray death (default: 3.0)")
    
    # Clip parameters
    ap.add_argument("--top", type=int, default=5, help="Max clips per demo (default: 5)")
    ap.add_argument("--pre", type=float, default=3.0, help="Seconds before burst for clip (default: 3.0)")
    ap.add_argument("--post", type=float, default=2.0, help="Seconds after death for clip (default: 2.0)")
    
    # Network
    ap.add_argument("--windows-url", default="http://10.0.0.108:8788",
                    help="Windows clip server URL")
    ap.add_argument("--ubuntu-url", default="http://10.0.0.196:8787",
                    help="Ubuntu receiver URL")
    
    # Download options
    ap.add_argument("--download", action="store_true", help="Download demos first")
    ap.add_argument("--gc-version", type=int, default=2000696, help="GC version for demo download")
    ap.add_argument("--start-code", help="Starting sharecode for demo download")
    ap.add_argument("--max-matches", type=int, default=50, help="Max matches to download")
    
    # Flags
    ap.add_argument("--dry-run", action="store_true", help="Analyze but don't send clip requests")
    ap.add_argument("--no-skip", action="store_true", help="Re-analyze already processed demos")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = ap.parse_args()
    
    # Build config
    cfg = Config()
    cfg = load_config_from_env(cfg)
    
    # Override from args
    if args.demo:
        # Single demo mode: use demo's directory as demo_dir
        cfg.demo_dir = args.demo.parent
    else:
        cfg.demo_dir = args.demo_dir.expanduser().resolve()
    
    cfg.out_dir = args.out.expanduser().resolve()
    cfg.inbox_dir = args.inbox.expanduser().resolve()
    cfg.player_filter = args.player
    cfg.tickrate = args.tickrate
    cfg.min_bullets = args.min_bullets
    cfg.die_within_s = args.die_within
    cfg.top_clips = args.top
    cfg.clip_pre_s = args.pre
    cfg.clip_post_s = args.post
    cfg.windows_base_url = args.windows_url.rstrip("/")
    cfg.ubuntu_base_url = args.ubuntu_url.rstrip("/")
    cfg.download_demos = args.download
    cfg.gc_version = args.gc_version
    cfg.start_code = args.start_code
    cfg.max_matches = args.max_matches
    cfg.dry_run = args.dry_run
    cfg.skip_analyzed = not args.no_skip
    
    log = setup_logging(args.debug)
    
    # If single demo specified, only process that one
    if args.demo:
        demo_path = args.demo.expanduser().resolve()
        if not demo_path.exists():
            log.error(f"Demo file not found: {demo_path}")
            return 1
        
        # Temporarily override demo_dir to just that file's directory
        # But we'll only process the specified file
        available_ids = get_available_demo_ids(cfg) if not cfg.dry_run else set()
        
        log.info(f"Processing single demo: {demo_path}")
        result = process_demo(demo_path, cfg, log, available_ids)
        log.info(f"Result: {result}")
        return 0 if result["status"].startswith("ok") else 1
    
    # Full orchestration
    results = run_orchestrator(cfg, log)
    
    # Exit code based on results
    errors = sum(1 for r in results if r["status"].startswith("error"))
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
