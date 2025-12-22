#!/usr/bin/env python3
"""
pipeline_runner.py - Background pipeline runner with status tracking

Runs the full pipeline (download -> analyze -> request clips) and saves
status to disk so the web UI can display progress.

Usage:
    # Run as background process
    python pipeline_runner.py --player Remag --demo-dir ./demos --out ./output

    # The web app reads status from ./output/pipeline_status.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from demoparser2 import DemoParser
except ImportError:
    print("ERROR: demoparser2 not installed")
    sys.exit(1)


# =========================
# Status tracking
# =========================

@dataclass
class PipelineStatus:
    """Pipeline status that gets saved to disk"""
    phase: str = "idle"  # idle, downloading, indexing, analyzing, requesting, receiving, complete, error
    message: str = ""
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    # Download phase
    demos_downloaded: int = 0
    
    # Index phase
    demos_indexed: int = 0
    demos_info: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis phase
    demos_analyzed: int = 0
    total_candidates: int = 0
    
    # Clip request phase
    clips_requested: int = 0
    
    # Receiving phase
    clips_received: int = 0
    clips_info: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error info
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "PipelineStatus":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StatusManager:
    """Manages pipeline status file"""
    
    def __init__(self, status_path: Path):
        self.status_path = status_path
        self.status = PipelineStatus()
        self._lock = threading.Lock()
    
    def load(self) -> PipelineStatus:
        """Load status from disk"""
        if self.status_path.exists():
            try:
                data = json.loads(self.status_path.read_text())
                self.status = PipelineStatus.from_dict(data)
            except Exception:
                self.status = PipelineStatus()
        return self.status
    
    def save(self) -> None:
        """Save status to disk"""
        with self._lock:
            self.status.updated_at = datetime.now().isoformat()
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.write_text(json.dumps(self.status.to_dict(), indent=2))
    
    def update(self, **kwargs) -> None:
        """Update status fields and save"""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.status, k):
                    setattr(self.status, k, v)
        self.save()
    
    def set_phase(self, phase: str, message: str = "") -> None:
        """Set current phase"""
        self.update(phase=phase, message=message)
    
    def set_error(self, error: str) -> None:
        """Set error state"""
        self.update(phase="error", error=error, message=f"Error: {error}")


# =========================
# Demo operations
# =========================

# Note: Demo downloading is now handled by demo_daemon.py
# This module only handles indexing, analysis, and clip generation

def index_demos_with_status(
    demo_dir: Path,
    status: StatusManager,
    tickrate: float,
    log: logging.Logger,
) -> pd.DataFrame:
    """Index demos and update status with info"""
    status.set_phase("indexing", "Indexing demo files...")
    
    demos = sorted(demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    demos_info = []
    for i, demo_path in enumerate(demos):
        try:
            info = summarize_demo_minimal(demo_path, tickrate)
            demos_info.append(info)
            status.update(
                demos_indexed=i + 1,
                demos_info=demos_info,
                message=f"Indexed {i + 1}/{len(demos)} demos"
            )
        except Exception as e:
            log.warning(f"Failed to index {demo_path.name}: {e}")
    
    return pd.DataFrame(demos_info)


def summarize_demo_minimal(demo_path: Path, tickrate: float = 64.0) -> dict:
    """Quick demo summary for display"""
    demo_path = demo_path.resolve()
    parser = DemoParser(str(demo_path))
    
    map_name = "Unknown"
    try:
        hdr = parser.parse_header()
        if isinstance(hdr, dict):
            map_name = hdr.get("map_name") or hdr.get("map") or map_name
    except Exception:
        pass
    
    # Get played_at from .dem.info
    played_at = None
    info_path = Path(str(demo_path) + ".info")
    if info_path.exists():
        try:
            import subprocess
            import re
            from datetime import timezone
            p = subprocess.run(
                ["protoc", "--decode_raw"],
                input=info_path.read_bytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if p.returncode == 0:
                txt = p.stdout.decode("utf-8", errors="replace")
                m = re.search(r"(?m)^2:\s*(\d+)\s*$", txt)
                if m:
                    epoch = int(m.group(1))
                    played_at = datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone().isoformat()
        except Exception:
            pass
    
    if played_at is None:
        played_at = datetime.fromtimestamp(demo_path.stat().st_mtime).isoformat()
    
    return {
        "demo_id": demo_path.stem,
        "filename": demo_path.name,
        "path": str(demo_path),
        "map": map_name,
        "played_at": played_at,
        "size_mb": round(demo_path.stat().st_size / (1024 * 1024), 1),
    }


# =========================
# Analysis (from orchestrator_batch)
# =========================

def detect_bursts(demo_path: Path, tickrate: float, player_filter: str) -> pd.DataFrame:
    parser = DemoParser(str(demo_path))
    max_gap_ticks = 8
    min_bullets = 6
    
    try:
        wf = parser.parse_events(["weapon_fire"])
        wf = wf[0][1]
    except Exception:
        return pd.DataFrame()
    
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


def detect_oversprays(demo_path: Path, bursts_df: pd.DataFrame, tickrate: float) -> pd.DataFrame:
    if bursts_df.empty:
        return pd.DataFrame()
    
    min_bullets = 6
    min_duration_s = 0.4
    die_within_s = 3.0
    
    bursts = bursts_df[
        (bursts_df["bullets"] >= min_bullets) &
        (bursts_df["duration_s"] >= min_duration_s)
    ].copy()
    
    if bursts.empty:
        return pd.DataFrame()
    
    parser = DemoParser(str(demo_path))
    
    try:
        deaths = parser.parse_events(["player_death"])
        deaths = deaths[0][1]
    except Exception:
        return pd.DataFrame()
    
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
    
    oversprays = []
    for row in bursts.itertuples(index=False):
        player = row.player
        start_tick = int(row.start_tick)
        end_tick = int(row.end_tick)
        
        # Find death after burst
        death_tick = None
        for dt in deaths_by_player.get(player, []):
            if dt >= end_tick and dt <= end_tick + die_within_ticks:
                death_tick = dt
                break
            if dt > end_tick + die_within_ticks:
                break
        
        if death_tick is None:
            continue
        
        # Check for kill
        has_kill = False
        for kt in kills_by_player.get(player, []):
            if kt >= start_tick and kt <= death_tick:
                has_kill = True
                break
        
        if has_kill:
            continue
        
        oversprays.append({
            "player": player,
            "weapon": row.weapon,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "bullets": int(row.bullets),
            "duration_s": float(row.duration_s),
            "death_tick": int(death_tick),
            "start_s": start_tick / tickrate,
            "death_s": death_tick / tickrate,
        })
    
    df = pd.DataFrame(oversprays)
    if not df.empty:
        df = df.sort_values(["bullets", "duration_s"], ascending=[False, False])
    return df


def analyze_demos_with_status(
    demo_dir: Path,
    out_dir: Path,
    player_name: str,
    status: StatusManager,
    tickrate: float,
    top_per_demo: int,
    clip_pre_s: float,
    clip_post_s: float,
    log: logging.Logger,
) -> List[dict]:
    """Analyze all demos and return batch request data"""
    status.set_phase("analyzing", "Analyzing demos for mistakes...")
    
    demos = sorted(demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    demos_data = []
    total_candidates = 0
    
    for i, demo_path in enumerate(demos):
        log.info(f"Analyzing {demo_path.name}...")
        status.update(
            demos_analyzed=i,
            message=f"Analyzing {i + 1}/{len(demos)}: {demo_path.name}"
        )
        
        try:
            bursts = detect_bursts(demo_path, tickrate, player_name)
            if bursts.empty:
                continue
            
            oversprays = detect_oversprays(demo_path, bursts, tickrate)
            if oversprays.empty:
                continue
            
            # Save analysis
            demo_out = out_dir / demo_path.stem
            demo_out.mkdir(parents=True, exist_ok=True)
            oversprays.to_parquet(demo_out / "overspray_candidates.parquet")
            
            # Build clips
            top = oversprays.head(top_per_demo)
            clips = []
            for _, row in top.iterrows():
                start_s = max(0.0, row["start_s"] - clip_pre_s)
                end_s = row["death_s"] + clip_post_s
                clips.append({
                    "start_s": round(start_s, 3),
                    "duration_s": round(end_s - start_s, 3),
                })
            
            if clips:
                demos_data.append({
                    "demo_id": demo_path.stem,
                    "clips": clips,
                })
                total_candidates += len(clips)
        
        except Exception as e:
            log.warning(f"Failed to analyze {demo_path.name}: {e}")
    
    status.update(
        demos_analyzed=len(demos),
        total_candidates=total_candidates,
        message=f"Analyzed {len(demos)} demos, found {total_candidates} clips"
    )
    
    return demos_data


# =========================
# Clip request and monitoring
# =========================

def send_batch_request(
    demos_data: List[dict],
    player_name: str,
    windows_url: str,
    ubuntu_url: str,
    token: str,
    status: StatusManager,
    log: logging.Logger,
) -> dict:
    """Send batch request to Windows"""
    upload_url = ubuntu_url.rstrip("/") + "/upload"
    
    payload = {
        "username": player_name,
        "ubuntu_upload_url": upload_url,
        "demos": demos_data,
    }
    
    total_clips = sum(len(d["clips"]) for d in demos_data)
    status.set_phase("requesting", f"Requesting {total_clips} clips from Windows...")
    status.update(clips_requested=total_clips)
    
    log.info(f"Sending batch request: {len(demos_data)} demos, {total_clips} clips")
    
    try:
        r = requests.post(
            windows_url.rstrip("/") + "/batch_clips",
            json=payload,
            headers={"X-Token": token},
            timeout=3600,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Batch request failed: {e}")
        raise


def monitor_inbox(
    inbox_dir: Path,
    expected_clips: int,
    status: StatusManager,
    timeout_s: float = 3600,
    log: logging.Logger = None,
) -> None:
    """Monitor inbox for incoming clips"""
    status.set_phase("receiving", "Waiting for clips from Windows...")
    
    inbox_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    seen_clips = set()
    
    while time.time() - start_time < timeout_s:
        clips = list(inbox_dir.glob("*.mp4"))
        new_clips = [c for c in clips if c.name not in seen_clips]
        
        if new_clips:
            for c in new_clips:
                seen_clips.add(c.name)
                if log:
                    log.info(f"Received clip: {c.name}")
            
            clips_info = [
                {
                    "filename": c.name,
                    "path": str(c),
                    "size_mb": round(c.stat().st_size / (1024 * 1024), 2),
                    "received_at": datetime.now().isoformat(),
                }
                for c in clips
            ]
            
            status.update(
                clips_received=len(clips),
                clips_info=clips_info,
                message=f"Received {len(clips)}/{expected_clips} clips"
            )
        
        if len(clips) >= expected_clips:
            status.set_phase("complete", f"All {len(clips)} clips received!")
            return
        
        time.sleep(2)
    
    status.update(message=f"Timeout: received {len(seen_clips)}/{expected_clips} clips")


# =========================
# Main pipeline
# =========================

@dataclass
class PipelineConfig:
    demo_dir: Path
    out_dir: Path
    inbox_dir: Path
    player_name: str
    
    # Analysis
    tickrate: float = 64.0
    top_per_demo: int = 10
    clip_pre_s: float = 3.0
    clip_post_s: float = 2.0
    
    # Network
    windows_url: str = "http://10.0.0.108:8788"
    ubuntu_url: str = "http://10.0.0.196:8787"
    token: str = "token"


def run_pipeline(cfg: PipelineConfig, status: StatusManager, log: logging.Logger) -> None:
    """Run the full pipeline (analysis + clip generation)"""
    try:
        status.update(
            started_at=datetime.now().isoformat(),
            phase="starting",
            message="Pipeline starting..."
        )
        
        # Ensure directories
        cfg.demo_dir.mkdir(parents=True, exist_ok=True)
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        cfg.inbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Note: Demo downloading is handled by demo_daemon.py
        # This pipeline just analyzes existing demos
        
        # Phase 1: Index demos
        index_demos_with_status(cfg.demo_dir, status, cfg.tickrate, log)
        
        # Check if we have demos
        demos = list(cfg.demo_dir.glob("*.dem"))
        if not demos:
            status.set_phase("complete", "No demos found. Make sure demo_daemon.py is running.")
            return
        
        # Phase 2: Analyze demos
        demos_data = analyze_demos_with_status(
            cfg.demo_dir, cfg.out_dir, cfg.player_name, status,
            cfg.tickrate, cfg.top_per_demo, cfg.clip_pre_s, cfg.clip_post_s, log
        )
        
        if not demos_data:
            status.set_phase("complete", f"No mistakes found for player '{cfg.player_name}'")
            return
        
        # Phase 3: Send batch request
        total_clips = sum(len(d["clips"]) for d in demos_data)
        
        try:
            result = send_batch_request(
                demos_data, cfg.player_name,
                cfg.windows_url, cfg.ubuntu_url, cfg.token,
                status, log
            )
            log.info(f"Batch request result: {result}")
        except Exception as e:
            status.set_error(f"Failed to send clip request: {e}")
            return
        
        # Phase 4: Monitor inbox for clips
        monitor_inbox(cfg.inbox_dir, total_clips, status, timeout_s=3600, log=log)
        
    except Exception as e:
        log.exception("Pipeline failed")
        status.set_error(str(e))


def main():
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger("pipeline")
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True, help="Player name (e.g., Remag)")
    ap.add_argument("--demo-dir", type=Path, default=Path("./demos"))
    ap.add_argument("--out", type=Path, default=Path("./output"))
    ap.add_argument("--inbox", type=Path, default=Path("./inbox"))
    
    ap.add_argument("--windows-url", default="http://10.0.0.108:8788")
    ap.add_argument("--ubuntu-url", default="http://10.0.0.196:8787")
    
    args = ap.parse_args()
    
    cfg = PipelineConfig(
        demo_dir=args.demo_dir.expanduser().resolve(),
        out_dir=args.out.expanduser().resolve(),
        inbox_dir=args.inbox.expanduser().resolve(),
        player_name=args.player,
        windows_url=args.windows_url,
        ubuntu_url=args.ubuntu_url,
        token=os.environ.get("CLIP_TOKEN", "token"),
    )
    
    status_path = cfg.out_dir / "pipeline_status.json"
    status = StatusManager(status_path)
    
    log.info(f"Starting pipeline for player: {cfg.player_name}")
    log.info(f"Status file: {status_path}")
    
    run_pipeline(cfg, status, log)


if __name__ == "__main__":
    main()
