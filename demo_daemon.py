#!/usr/bin/env python3
"""
demo_daemon.py - Persistent Steam demo download + analysis service

Stays logged into Steam, downloads new matches automatically, and immediately
triggers clip generation when new demos arrive.

Usage:
    # Start the daemon (will prompt for 2FA interactively if no shared_secret)
    python demo_daemon.py --demo-dir ./demos --player Remag

    # Run in background
    nohup python demo_daemon.py --demo-dir ./demos --player Remag > daemon.log 2>&1 &

2FA Options (in order of preference):
    1. STEAM_SHARED_SECRET env var - Auto-generates 2FA codes (best for production)
    2. Interactive prompt - Enter code manually at startup

To get your shared_secret:
    - Use Steam Desktop Authenticator (Windows): Look in maFiles/*.maFile
    - Use steamguard-cli (Linux): ~/.config/steamguard-cli/maFiles/*.maFile
    - Extract from Android: adb shell run-as com.valvesoftware.android.steam.community

Features:
    - Stays logged into Steam (2FA only at startup, or auto with shared_secret)
    - Polls for new matches every N minutes
    - Immediately analyzes new demos for player mistakes
    - Sends clip requests to Windows
    - Updates status file for web UI
    - HTTP API for status and manual triggers
"""

from gevent import monkey
monkey.patch_all()

import argparse
import base64
import bz2
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import requests as http_requests
from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict

from steam.client import SteamClient
from steam.enums.common import EResult

from csgo.client import CSGOClient
from csgo import sharecode
from csgo.enums import EGCBaseClientMsg, ECsgoGCMsg

try:
    from demoparser2 import DemoParser
    HAS_PARSER = True
except ImportError:
    HAS_PARSER = False
    print("WARNING: demoparser2 not installed - analysis disabled")

try:
    from flask import Flask, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


# =========================
# Steam TOTP Generator
# =========================

STEAM_ALPHABET = "23456789BCDFGHJKMNPQRTVWXY"

def generate_steam_totp(shared_secret: str) -> str:
    """
    Generate a Steam-style 2FA code from the shared_secret.
    
    This allows fully automated logins without manual 2FA entry.
    """
    # Decode the base64 shared_secret
    key = base64.b64decode(shared_secret)
    
    # Get current time interval (30 second periods)
    timestamp = int(time.time()) // 30
    
    # Create HMAC-SHA1
    msg = struct.pack(">Q", timestamp)
    auth = hmac.new(key, msg, hashlib.sha1)
    digest = auth.digest()
    
    # Dynamic truncation
    start = digest[19] & 0x0F
    code_int = struct.unpack(">I", digest[start:start + 4])[0] & 0x7FFFFFFF
    
    # Convert to Steam's alphabet
    code_chars = []
    for _ in range(5):
        code_chars.append(STEAM_ALPHABET[code_int % len(STEAM_ALPHABET)])
        code_int //= len(STEAM_ALPHABET)
    
    return "".join(code_chars)


# =========================
# Status tracking
# =========================

@dataclass
class DaemonStatus:
    """Status saved to disk for web UI"""
    running: bool = False
    phase: str = "idle"  # idle, downloading, analyzing, requesting, complete
    message: str = ""
    
    # Connection
    steam_connected: bool = False
    gc_ready: bool = False
    
    # Demos
    last_check: str = ""
    total_downloaded: int = 0
    demos_available: int = 0
    
    # Current job
    current_demo: str = ""
    demos_analyzed: int = 0
    clips_requested: int = 0
    clips_received: int = 0
    
    # Batch state
    pending_demos: int = 0  # Demos waiting to be sent in batch
    pending_clips: int = 0  # Clips waiting to be sent in batch
    batch_sent: bool = False
    
    # Recent activity
    recent_demos: List[Dict[str, Any]] = field(default_factory=list)
    recent_clips: List[Dict[str, Any]] = field(default_factory=list)
    
    # Errors
    last_error: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "DaemonStatus":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StatusManager:
    def __init__(self, path: Path):
        self.path = path
        self.status = DaemonStatus()
        self._lock = threading.Lock()
    
    def load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.status = DaemonStatus.from_dict(data)
            except Exception:
                self.status = DaemonStatus()
    
    def save(self):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.status.to_dict(), indent=2, default=str))
    
    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.status, k):
                    setattr(self.status, k, v)
        self.save()
    
    def add_recent_demo(self, demo_info: dict):
        with self._lock:
            self.status.recent_demos.insert(0, demo_info)
            self.status.recent_demos = self.status.recent_demos[:20]  # Keep last 20
        self.save()
    
    def add_recent_clip(self, clip_info: dict):
        with self._lock:
            self.status.recent_clips.insert(0, clip_info)
            self.status.recent_clips = self.status.recent_clips[:50]  # Keep last 50
        self.save()


# =========================
# Logging
# =========================

def setup_logging(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger("daemon")
    if not debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("steam").setLevel(logging.WARNING)
    return log


# =========================
# Demo Analysis
# =========================

def detect_bursts(demo_path: Path, tickrate: float, player_filter: str) -> pd.DataFrame:
    if not HAS_PARSER:
        return pd.DataFrame()
    
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
    if not HAS_PARSER or bursts_df.empty:
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
    
    deaths_v = deaths[["tick", "user_name"]].copy()
    deaths_v = deaths_v.rename(columns={"tick": "death_tick", "user_name": "player"})
    deaths_v["death_tick"] = deaths_v["death_tick"].astype(int)
    
    kills = deaths[["tick", "attacker_name"]].copy()
    kills = kills.rename(columns={"tick": "kill_tick", "attacker_name": "player"})
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
        
        death_tick = None
        for dt in deaths_by_player.get(player, []):
            if dt >= end_tick and dt <= end_tick + die_within_ticks:
                death_tick = dt
                break
        
        if death_tick is None:
            continue
        
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


def detect_knife_deaths(demo_path: Path, tickrate: float, player_filter: str) -> pd.DataFrame:
    """Detect deaths where player was holding a knife
    
    Note: At the exact death tick, active_weapon_name is None (player is dead).
    We need to check 1-2 ticks BEFORE death to get the weapon they were holding.
    
    Excludes:
    - Bomb deaths (planted_c4) - not a mistake to have knife when bomb explodes
    - Fire/grenade deaths - environmental
    - Suicides/world deaths
    """
    if not HAS_PARSER:
        return pd.DataFrame()
    
    parser = DemoParser(str(demo_path))
    
    try:
        deaths = parser.parse_events(["player_death"])
        deaths = deaths[0][1]
    except Exception:
        return pd.DataFrame()
    
    if deaths.empty:
        return pd.DataFrame()
    
    # Filter by player
    if player_filter:
        player_deaths = deaths[deaths["user_name"].str.contains(player_filter, case=False, na=False)]
    else:
        player_deaths = deaths
    
    if player_deaths.empty:
        return pd.DataFrame()
    
    # Get ticks BEFORE death (not at death - weapon is None when dead)
    # Check 1-2 ticks before death to get the weapon they were holding
    pre_death_ticks = []
    death_tick_map = {}  # pre_tick -> original death tick
    
    for _, death in player_deaths.iterrows():
        death_tick = int(death["tick"])
        # Check 1 tick before death
        pre_tick = death_tick - 1
        pre_death_ticks.append(pre_tick)
        death_tick_map[pre_tick] = death_tick
    
    try:
        # Parse player state at PRE-death ticks
        player_data = parser.parse_ticks(["active_weapon_name"], ticks=pre_death_ticks)
    except Exception:
        player_data = pd.DataFrame()
    
    if player_data.empty:
        return pd.DataFrame()
    
    knife_deaths = []
    
    # Weapons that indicate bomb/environment kills (not player fault for having knife)
    bomb_weapons = {"planted_c4", "c4_planted", "inferno", "hegrenade", "world"}
    
    for _, death in player_deaths.iterrows():
        victim = death["user_name"]
        death_tick = int(death["tick"])
        pre_tick = death_tick - 1
        attacker = death.get("attacker_name", "Unknown") or "Unknown"
        attacker_weapon = death.get("weapon", "Unknown") or "Unknown"
        
        # Skip bomb deaths - not a mistake to have knife out when bomb explodes
        if attacker_weapon and str(attacker_weapon).lower() in bomb_weapons:
            continue
        
        # Skip suicide/world deaths
        if not attacker or attacker == "Unknown" or attacker == victim:
            continue
        
        # Find victim's weapon at pre-death tick
        victim_state = player_data[
            (player_data["tick"] == pre_tick) & 
            (player_data["name"] == victim)
        ]
        
        if victim_state.empty:
            # Try exact death tick as fallback (might work for some demos)
            victim_state = player_data[
                (player_data["tick"] == death_tick) & 
                (player_data["name"] == victim)
            ]
        
        if victim_state.empty:
            continue
        
        active_weapon = victim_state.iloc[0].get("active_weapon_name", "")
        
        # Skip if no weapon data
        if not active_weapon:
            continue
        
        # Check if holding knife (includes all knife variants)
        weapon_lower = str(active_weapon).lower()
        is_knife = "knife" in weapon_lower or "bayonet" in weapon_lower or "karambit" in weapon_lower
        
        if is_knife:
            knife_deaths.append({
                "player": victim,
                "tick": death_tick,
                "time_s": round(death_tick / tickrate, 1),
                "active_weapon": active_weapon,
                "killed_by": attacker,
                "killer_weapon": attacker_weapon,
            })
    
    df = pd.DataFrame(knife_deaths)
    if not df.empty:
        df = df.sort_values("time_s")
    return df


def get_demo_info(demo_path: Path, tickrate: float = 64.0) -> dict:
    """Get basic demo info"""
    if not HAS_PARSER:
        return {"demo_id": demo_path.stem, "map": "Unknown"}
    
    parser = DemoParser(str(demo_path))
    map_name = "Unknown"
    
    try:
        hdr = parser.parse_header()
        if isinstance(hdr, dict):
            map_name = hdr.get("map_name") or hdr.get("map") or map_name
    except Exception:
        pass
    
    return {
        "demo_id": demo_path.stem,
        "filename": demo_path.name,
        "map": map_name,
        "size_mb": round(demo_path.stat().st_size / 1024 / 1024, 1),
        "downloaded_at": datetime.now().isoformat(),
    }


# =========================
# Steam/CSGO helpers
# =========================

def find_demo_url(obj) -> Optional[str]:
    exts = (".dem", ".dem.bz2", ".bz2")
    if isinstance(obj, dict):
        for _, v in obj.items():
            u = find_demo_url(v)
            if u:
                return u
    elif isinstance(obj, list):
        for v in obj:
            u = find_demo_url(v)
            if u:
                return u
    elif isinstance(obj, str):
        low = obj.lower()
        if obj.startswith("http") and any(ext in low for ext in exts):
            return obj
    return None


def decode_share_code(code: str) -> tuple:
    ids = sharecode.decode(code)
    return int(ids["matchid"]), int(ids["outcomeid"]), int(ids["token"])


def download_file(url: str, out_path: Path, timeout_s: int = 120, log: logging.Logger = None, max_retries: int = 3) -> bool:
    """Download file with retry logic for rate limiting"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if log:
        log.info(f"Downloading: {url}")
    
    for attempt in range(max_retries):
        try:
            with http_requests.get(url, stream=True, timeout=timeout_s) as r:
                # Check for rate limiting
                if r.status_code == 429:
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    if log:
                        log.warning(f"Rate limited (429), waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                # Check for Akamai CDN errors (usually HTML with Reference #)
                content_type = r.headers.get("content-type", "")
                if "text/html" in content_type and r.status_code >= 400:
                    wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                    if log:
                        log.warning(f"CDN rate limit detected, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                r.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            f.write(chunk)
            return True
        except http_requests.exceptions.HTTPError as e:
            if "429" in str(e) or "too many" in str(e).lower():
                wait_time = 60 * (attempt + 1)
                if log:
                    log.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            if log:
                log.error(f"Download failed: {e}")
            return False
        except Exception as e:
            if log:
                log.error(f"Download failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            return False
    
    if log:
        log.error(f"Download failed after {max_retries} attempts")
    return False


def get_next_sharecode_webapi(
    web_api_key: str,
    steamid64: str,
    auth_code: str,
    known_code: str,
    log: logging.Logger,
    max_retries: int = 3,
) -> Optional[str]:
    """Get next sharecode with retry logic for rate limiting"""
    url = "https://api.steampowered.com/ICSGOPlayers_730/GetNextMatchSharingCode/v1/"
    params = {
        "key": web_api_key,
        "steamid": steamid64,
        "steamidkey": auth_code,
        "knowncode": known_code,
    }
    
    for attempt in range(max_retries):
        try:
            r = http_requests.get(url, params=params, timeout=20)
            
            # Handle rate limiting
            if r.status_code == 429:
                wait_time = 30 * (attempt + 1)
                log.warning(f"Steam API rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            r.raise_for_status()
            data = r.json()
            
            result = data.get("result", {})
            nextcode = result.get("nextcode", "")
            
            if isinstance(nextcode, str) and nextcode.startswith("CSGO-"):
                return nextcode
            return None
        except http_requests.exceptions.HTTPError as e:
            if "429" in str(e):
                wait_time = 30 * (attempt + 1)
                log.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            log.warning(f"WebAPI error: {e}")
            return None
        except Exception as e:
            log.warning(f"WebAPI error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None
    
    return None


# =========================
# Daemon State (persistent across restarts)
# =========================

@dataclass
class DaemonState:
    last_sharecode: str = ""
    processed_demos: List[str] = field(default_factory=list)
    
    # Pending batch - demos analyzed but not yet sent to Windows
    pending_batch: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "DaemonState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StateManager:
    def __init__(self, path: Path):
        self.path = path
        self.state = DaemonState()
        self.load()
    
    def load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.state = DaemonState.from_dict(data)
            except Exception:
                self.state = DaemonState()
    
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state.to_dict(), indent=2))
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.state, k):
                setattr(self.state, k, v)
        self.save()
    
    def mark_processed(self, demo_id: str):
        if demo_id not in self.state.processed_demos:
            self.state.processed_demos.append(demo_id)
            self.state.processed_demos = self.state.processed_demos[-200:]  # Keep last 200
            self.save()
    
    def is_processed(self, demo_id: str) -> bool:
        return demo_id in self.state.processed_demos
    
    def add_to_batch(self, demo_id: str, demo_path: str, clips: List[dict]):
        """Add a demo's clips to the pending batch"""
        # Check if already in batch
        for item in self.state.pending_batch:
            if item["demo_id"] == demo_id:
                return
        
        self.state.pending_batch.append({
            "demo_id": demo_id,
            "demo_path": demo_path,
            "clips": clips,
        })
        self.save()
    
    def get_pending_batch(self) -> List[Dict[str, Any]]:
        """Get pending batch for sending to Windows"""
        return self.state.pending_batch
    
    def clear_batch(self):
        """Clear pending batch after successful send"""
        self.state.pending_batch = []
        self.save()
    
    def get_batch_demo_ids(self) -> List[str]:
        """Get list of demo IDs in pending batch"""
        return [item["demo_id"] for item in self.state.pending_batch]


# =========================
# Demo Daemon
# =========================

class DemoDaemon:
    def __init__(
        self,
        demo_dir: Path,
        output_dir: Path,
        inbox_dir: Path,
        player_name: str,
        gc_version: int,
        poll_interval_s: int = 300,
        windows_url: str = "http://10.0.0.108:8788",
        ubuntu_url: str = "http://10.0.0.196:8787",
        clip_token: str = "token",
        tickrate: float = 64.0,
        top_clips: int = 10,
        clip_pre_s: float = 3.0,
        clip_post_s: float = 2.0,
        download_delay_s: int = 30,  # Delay between demo downloads (avoid rate limit)
        rate_limit_delay_s: int = 120,  # Delay when rate limited
        log: logging.Logger = None,
    ):
        self.demo_dir = demo_dir
        self.output_dir = output_dir
        self.inbox_dir = inbox_dir
        self.player_name = player_name
        self.gc_version = gc_version
        self.poll_interval_s = poll_interval_s
        self.windows_url = windows_url.rstrip("/")
        self.ubuntu_url = ubuntu_url.rstrip("/")
        self.clip_token = clip_token
        self.tickrate = tickrate
        self.top_clips = top_clips
        self.clip_pre_s = clip_pre_s
        self.clip_post_s = clip_post_s
        self.download_delay_s = download_delay_s
        self.rate_limit_delay_s = rate_limit_delay_s
        self.log = log or logging.getLogger("daemon")
        
        # Paths
        self.state_path = demo_dir / ".daemon_state.json"
        self.status_path = output_dir / "daemon_status.json"
        
        # Managers
        self.state = StateManager(self.state_path)
        self.status = StatusManager(self.status_path)
        
        # Steam
        self.client = SteamClient()
        self.cs = CSGOClient(self.client)
        
        # State
        self.gc_ready = False
        self.running = False
        
        # Config from env
        self.steam_web_api_key = os.environ.get("STEAM_WEB_API_KEY", "").strip()
        self.target_steamid64 = os.environ.get("TARGET_STEAMID64", "").strip()
        self.target_auth_code = os.environ.get("TARGET_AUTH_CODE", "").strip()
        self.start_code = os.environ.get("TARGET_KNOWN_CODE", "").strip()
        self.shared_secret = os.environ.get("STEAM_SHARED_SECRET", "").strip()
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self.client.on("disconnected")
        def on_disconnect():
            self.log.warning("Steam disconnected")
            self.gc_ready = False
            self.status.update(steam_connected=False, gc_ready=False)
            
            # Auto-reconnect handled in main loop, not here
            # (to avoid issues with greenlet state)
        
        @self.client.on("reconnect")
        def on_reconnect(delay):
            self.log.info(f"Steam reconnecting in {delay}s...")
        
        @self.cs.on(EGCBaseClientMsg.EMsgGCClientWelcome)
        def on_gc_welcome(msg):
            self.gc_ready = True
            self.log.info("GC ready")
            self.status.update(gc_ready=True)
        
        @self.cs.on(ECsgoGCMsg.EMsgGCCStrike15_v2_ClientLogonFatalError)
        def on_gc_fatal(msg):
            self.log.error("GC fatal error")
            self.gc_ready = False
            self.status.update(gc_ready=False, last_error="GC fatal error")
    
    def _ensure_connected(self) -> bool:
        """Ensure we're connected to Steam and GC. Reconnect if needed."""
        # Check if already connected
        if self.client.connected and self.client.logged_on:
            if not self.gc_ready:
                return self.connect_gc()
            return True
        
        # Need to reconnect
        self.log.info("Connection lost, reconnecting...")
        
        for attempt in range(3):
            try:
                # Disconnect cleanly first if in bad state
                if self.client.connected:
                    try:
                        self.client.disconnect()
                    except Exception:
                        pass
                    time.sleep(2)
                
                if self.login():
                    if self.connect_gc():
                        self.log.info("Reconnect successful")
                        return True
            except Exception as e:
                self.log.warning(f"Reconnect attempt {attempt + 1} failed: {e}")
            
            time.sleep(30)
        
        self.log.error("Failed to reconnect after 3 attempts")
        return False
    
    def login(self) -> bool:
        self.log.info("Logging into Steam...")
        self.status.update(phase="connecting", message="Logging into Steam...")
        
        user = os.environ.get("BOT_STEAM_USER", "").strip()
        pw = os.environ.get("BOT_STEAM_PASS", "").strip()
        
        if not user or not pw:
            self.log.error("BOT_STEAM_USER and BOT_STEAM_PASS required")
            return False
        
        # Check if already logged on
        if self.client.logged_on:
            self.log.info("Already logged on")
            self.status.update(steam_connected=True)
            return True
        
        # Try relogin first (uses saved session)
        try:
            result = self.client.relogin()
            if result == EResult.OK:
                self.log.info("Relogin successful (using saved session)")
                self.status.update(steam_connected=True)
                return True
        except Exception as e:
            self.log.debug(f"Relogin failed: {e}")
        
        # Fresh login required
        if self.shared_secret:
            # Automatic 2FA using shared_secret
            self.log.info("Logging in with automatic 2FA (shared_secret)...")
            
            try:
                result = self.client.login(user, pw)
            except RuntimeError as e:
                if "Already logged on" in str(e):
                    self.log.info("Already logged on")
                    self.status.update(steam_connected=True)
                    return True
                raise
            
            if result == EResult.AccountLoginDeniedNeedTwoFactor:
                # Generate and submit 2FA code
                code = generate_steam_totp(self.shared_secret)
                self.log.info(f"Generated 2FA code: {code}")
                result = self.client.login(user, pw, two_factor_code=code)
            
            if result == EResult.TwoFactorCodeMismatch:
                # Wait and retry (clock sync issue)
                self.log.warning("2FA code mismatch, retrying in 5s...")
                time.sleep(5)
                code = generate_steam_totp(self.shared_secret)
                result = self.client.login(user, pw, two_factor_code=code)
        else:
            # Interactive login (will prompt for 2FA)
            self.log.info("Interactive login (enter 2FA when prompted)...")
            self.log.info("TIP: Set STEAM_SHARED_SECRET for automatic 2FA")
            result = self.client.cli_login(user, pw)
        
        if result != EResult.OK:
            self.log.error(f"Login failed: {result}")
            self.status.update(last_error=f"Login failed: {result}")
            return False
        
        self.log.info("Steam login successful")
        self.status.update(steam_connected=True)
        return True
    
    def connect_gc(self) -> bool:
        self.log.info("Connecting to CS2 GC...")
        self.status.update(phase="connecting", message="Connecting to CS2...")
        
        self.client.games_played([730])
        
        start = time.time()
        timeout = 60
        
        while time.time() - start < timeout:
            if self.gc_ready:
                self.log.info("GC connected")
                return True
            
            self.cs.send(EGCBaseClientMsg.EMsgGCClientHello, {"version": self.gc_version})
            self.client.sleep(3)
        
        self.log.error("GC connection timeout")
        self.status.update(last_error="GC connection timeout")
        return False
    
    def download_demo(self, matchid: int, outcomeid: int, token: int) -> Optional[Path]:
        """Download a single demo. Returns path if successful."""
        dem_path = self.demo_dir / f"{matchid}.dem"
        bz2_path = self.demo_dir / f"{matchid}.dem.bz2"
        
        if dem_path.exists():
            self.log.info(f"Already have {dem_path.name}")
            return dem_path
        
        self.log.info(f"Requesting match info for {matchid}...")
        self.cs.request_full_match_info(matchid, outcomeid, token)
        
        ev = self.cs.wait_event("full_match_info", timeout=60)
        if not ev:
            self.log.error(f"Timeout getting match info")
            return None
        
        msg = ev[0]
        d = MessageToDict(msg, preserving_proto_field_name=True)
        
        # Save match info
        info_path = self.demo_dir / f"{matchid}_info.json"
        info_path.write_text(json.dumps(d, indent=2))
        
        url = find_demo_url(d)
        if not url:
            self.log.error(f"No demo URL found")
            return None
        
        if not download_file(url, bz2_path, log=self.log):
            return None
        
        # Decompress
        try:
            dem_path.write_bytes(bz2.decompress(bz2_path.read_bytes()))
            bz2_path.unlink()
            self.log.info(f"Downloaded: {dem_path.name}")
        except Exception:
            self.log.info(f"Downloaded (not bz2): {bz2_path.name}")
            return bz2_path
        
        return dem_path
    
    def analyze_demo(self, demo_path: Path) -> List[dict]:
        """Analyze demo and return clip specifications for all mistake types"""
        self.log.info(f"Analyzing {demo_path.name} for player '{self.player_name}'...")
        self.status.update(
            phase="analyzing",
            current_demo=demo_path.name,
            message=f"Analyzing {demo_path.name}..."
        )
        
        clips = []
        demo_out = self.output_dir / demo_path.stem
        demo_out.mkdir(parents=True, exist_ok=True)
        
        # === Overspray Detection ===
        bursts = detect_bursts(demo_path, self.tickrate, self.player_name)
        if not bursts.empty:
            oversprays = detect_oversprays(demo_path, bursts, self.tickrate)
            if not oversprays.empty:
                oversprays.to_parquet(demo_out / "overspray_candidates.parquet")
                self.log.info(f"Found {len(oversprays)} oversprays")
                
                # Build overspray clips
                top_oversprays = oversprays.head(self.top_clips)
                for _, row in top_oversprays.iterrows():
                    start_s = max(0.0, row["start_s"] - self.clip_pre_s)
                    end_s = row["death_s"] + self.clip_post_s
                    clips.append({
                        "start_s": round(start_s, 3),
                        "duration_s": round(end_s - start_s, 3),
                        "type": "overspray",
                    })
        
        # === Knife Death Detection ===
        knife_deaths = detect_knife_deaths(demo_path, self.tickrate, self.player_name)
        if not knife_deaths.empty:
            knife_deaths.to_parquet(demo_out / "knife_deaths.parquet")
            self.log.info(f"Found {len(knife_deaths)} knife deaths")
            
            # Build knife death clips (limit to top 3)
            top_knife = knife_deaths.head(3)
            for _, row in top_knife.iterrows():
                start_s = max(0.0, row["time_s"] - 5.0)  # 5 seconds before death
                end_s = row["time_s"] + 2.0  # 2 seconds after death
                clips.append({
                    "start_s": round(start_s, 3),
                    "duration_s": round(end_s - start_s, 3),
                    "type": "knife_death",
                })
        
        self.log.info(f"Found {len(clips)} total clip candidates")
        return clips
    
    def process_new_demo(self, demo_path: Path) -> bool:
        """Analyze demo and add clips to pending batch (don't send yet)"""
        demo_id = demo_path.stem
        
        if self.state.is_processed(demo_id):
            self.log.debug(f"Already processed: {demo_id}")
            return False
        
        # Get demo info
        demo_info = get_demo_info(demo_path, self.tickrate)
        self.status.add_recent_demo(demo_info)
        
        # Analyze
        clips = self.analyze_demo(demo_path)
        
        if clips:
            # Add to pending batch (don't send yet)
            self.state.add_to_batch(demo_id, str(demo_path), clips)
            self.log.info(f"Added {len(clips)} clips from {demo_id} to pending batch")
            
            # Update status
            pending = self.state.get_pending_batch()
            total_clips = sum(len(d["clips"]) for d in pending)
            self.status.update(
                pending_demos=len(pending),
                pending_clips=total_clips,
            )
        
        # Mark as processed
        self.state.mark_processed(demo_id)
        self.status.update(demos_analyzed=self.status.status.demos_analyzed + 1)
        
        return len(clips) > 0
    
    def send_pending_batch(self) -> bool:
        """Send all pending clips to Windows in a single batch request"""
        pending = self.state.get_pending_batch()
        
        if not pending:
            self.log.info("No pending clips to send")
            return True
        
        total_clips = sum(len(d["clips"]) for d in pending)
        self.log.info(f"Sending batch: {len(pending)} demos, {total_clips} clips")
        
        self.status.update(
            phase="requesting",
            clips_requested=total_clips,
            message=f"Sending {total_clips} clips from {len(pending)} demos to Windows..."
        )
        
        # Build payload
        demos_payload = []
        for item in pending:
            demos_payload.append({
                "demo_id": item["demo_id"],
                "clips": item["clips"],
            })
        
        payload = {
            "username": self.player_name,
            "ubuntu_upload_url": f"{self.ubuntu_url}/upload",
            "demos": demos_payload,
            "delete_after": True,  # Tell Windows to delete demos after processing
        }
        
        try:
            r = http_requests.post(
                f"{self.windows_url}/batch_clips",
                json=payload,
                headers={"X-Token": self.clip_token},
                timeout=7200,  # 2 hours for large batches
            )
            r.raise_for_status()
            result = r.json()
            self.log.info(f"Batch request result: {result.get('clips_successful', 0)} successful")
            
            # Clear pending batch
            self.state.clear_batch()
            self.status.update(
                pending_demos=0,
                pending_clips=0,
                batch_sent=True,
                phase="receiving",
                message=f"Waiting for {total_clips} clips from Windows..."
            )
            
            return True
        except Exception as e:
            self.log.error(f"Batch request failed: {e}")
            self.status.update(last_error=f"Batch request failed: {e}")
            return False
    
    def delete_demo(self, demo_path: Path) -> bool:
        """Delete a demo file and its associated files (keeps analysis output for summary)"""
        demo_id = demo_path.stem
        
        try:
            # Delete main demo file
            if demo_path.exists():
                demo_path.unlink()
                self.log.info(f"Deleted demo: {demo_path.name}")
            
            # Delete associated files
            for suffix in ["_info.json", ".dem.bz2"]:
                assoc = demo_path.parent / f"{demo_id}{suffix}"
                if assoc.exists():
                    assoc.unlink()
            
            # NOTE: Keep analysis output (parquet files) for summary display in UI
            # They are small and needed to show stats after cleanup
            
            return True
        except Exception as e:
            self.log.warning(f"Failed to delete demo {demo_id}: {e}")
            return False
    
    def cleanup_processed_demos(self):
        """Delete all demos that have been successfully processed"""
        pending_ids = set(self.state.get_batch_demo_ids())
        
        for demo_path in list(self.demo_dir.glob("*.dem")):
            demo_id = demo_path.stem
            
            # Don't delete if in pending batch
            if demo_id in pending_ids:
                continue
            
            # Delete if processed
            if self.state.is_processed(demo_id):
                self.delete_demo(demo_path)
        
        self.status.update(demos_available=len(list(self.demo_dir.glob("*.dem"))))
    
    def check_for_new_matches(self) -> int:
        """Check for and download ALL new matches, then send batch when done"""
        if not self._ensure_connected():
            self.log.error("Cannot check for matches - not connected")
            return 0
        
        self.status.update(phase="checking", message="Checking for new matches...")
        
        known_code = self.state.state.last_sharecode or self.start_code
        if not known_code:
            self.log.warning("No starting sharecode")
            return 0
        
        if not all([self.steam_web_api_key, self.target_steamid64, self.target_auth_code]):
            # Just try the start code
            try:
                matchid, outcomeid, token = decode_share_code(known_code)
                demo_path = self.download_demo(matchid, outcomeid, token)
                if demo_path and demo_path.exists():
                    self.status.update(total_downloaded=self.status.status.total_downloaded + 1)
                    self.process_new_demo(demo_path)
                    
                    # Send batch if we have pending clips
                    if self.state.get_pending_batch():
                        self.send_pending_batch()
                    
                    return 1
            except Exception as e:
                self.log.error(f"Failed: {e}")
            return 0
        
        # Walk forward through ALL available matches
        downloaded = 0
        current = known_code
        max_iterations = 100  # Safety limit
        
        self.log.info(f"Checking for new matches (will download ALL before sending clips)...")
        
        for iteration in range(max_iterations):
            next_code = get_next_sharecode_webapi(
                self.steam_web_api_key,
                self.target_steamid64,
                self.target_auth_code,
                current,
                self.log,
            )
            
            if not next_code:
                # No more matches - we've caught up!
                self.log.info(f"No more new matches. Downloaded {downloaded} demos total.")
                break
            
            self.log.info(f"New match found: {next_code[:20]}...")
            self.status.update(
                phase="downloading",
                message=f"Downloading match {downloaded + 1}..."
            )
            
            try:
                matchid, outcomeid, token = decode_share_code(next_code)
                demo_path = self.download_demo(matchid, outcomeid, token)
                
                if demo_path and demo_path.exists():
                    downloaded += 1
                    self.status.update(total_downloaded=self.status.status.total_downloaded + 1)
                    self.state.update(last_sharecode=next_code)
                    
                    # Analyze and add to batch (don't send yet)
                    self.process_new_demo(demo_path)
            except Exception as e:
                self.log.error(f"Failed to download {next_code}: {e}")
                # If rate limited, wait longer before retry
                if "429" in str(e) or "too many" in str(e).lower() or "Reference #" in str(e):
                    self.log.warning(f"Rate limited! Waiting {self.rate_limit_delay_s} seconds...")
                    time.sleep(self.rate_limit_delay_s)
            
            current = next_code
            # Wait between downloads to avoid Akamai CDN rate limit
            # Demo files are large (~100MB), need to be conservative
            self.log.debug(f"Waiting {self.download_delay_s}s before next download...")
            time.sleep(self.download_delay_s)
        
        # Update status
        self.status.update(
            last_check=datetime.now().isoformat(),
        )
        
        # Now send batch if we have pending clips
        pending = self.state.get_pending_batch()
        if pending:
            total_clips = sum(len(d["clips"]) for d in pending)
            self.log.info(f"All demos downloaded. Sending batch: {len(pending)} demos, {total_clips} clips")
            self.send_pending_batch()
        else:
            self.status.update(
                phase="idle",
                message=f"Downloaded {downloaded} demos, no clips to send" if downloaded else "No new demos"
            )
        
        return downloaded
    
    def check_inbox_for_clips(self):
        """Check inbox for new clips and update status. Cleanup when complete."""
        clips = list(self.inbox_dir.glob("*.mp4"))
        prev_count = self.status.status.clips_received
        
        # Update recent clips
        for clip in sorted(clips, key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
            clip_info = {
                "filename": clip.name,
                "path": str(clip),
                "size_mb": round(clip.stat().st_size / 1024 / 1024, 2),
                "received_at": datetime.fromtimestamp(clip.stat().st_mtime).isoformat(),
            }
            
            # Check if already in recent
            existing = [c["filename"] for c in self.status.status.recent_clips]
            if clip.name not in existing:
                self.status.add_recent_clip(clip_info)
        
        self.status.update(clips_received=len(clips))
        
        # Check if batch is complete
        if self.status.status.batch_sent and self.status.status.clips_requested > 0:
            if len(clips) >= self.status.status.clips_requested:
                self.log.info(f"All {len(clips)} clips received! Cleaning up demos...")
                
                # Cleanup demos from Ubuntu
                self.cleanup_processed_demos()
                
                self.status.update(
                    phase="complete",
                    batch_sent=False,
                    message=f"✅ Complete! Received {len(clips)} clips."
                )
    
    def scan_existing_demos(self):
        """Process any unprocessed demos already in demo_dir, then send batch"""
        demos = sorted(self.demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime)
        
        processed_any = False
        for demo_path in demos:
            if not self.state.is_processed(demo_path.stem):
                self.log.info(f"Found unprocessed demo: {demo_path.name}")
                self.process_new_demo(demo_path)
                processed_any = True
        
        # Send batch if we processed any demos
        if processed_any and self.state.get_pending_batch():
            self.log.info("Finished processing existing demos, sending batch...")
            self.send_pending_batch()
    
    def run(self):
        self.running = True
        self.status.update(
            running=True,
            phase="starting",
            message="Starting daemon..."
        )
        
        # Ensure directories
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Login
        if not self.login():
            self.status.update(running=False, phase="error")
            return
        
        # Connect GC
        if not self.connect_gc():
            self.status.update(running=False, phase="error")
            return
        
        self.log.info(f"Daemon running. Player: {self.player_name}")
        self.log.info(f"Demo dir: {self.demo_dir}")
        self.log.info(f"Polling every {self.poll_interval_s}s")
        self.log.info(f"Rate limiting: {self.download_delay_s}s between downloads, {self.rate_limit_delay_s}s on rate limit")
        
        self.status.update(
            phase="idle",
            message="Ready",
            demos_available=len(list(self.demo_dir.glob("*.dem")))
        )
        
        # Process existing unprocessed demos
        self.scan_existing_demos()
        
        # Initial check
        self.check_for_new_matches()
        
        # Main loop
        last_inbox_check = 0
        
        while self.running:
            try:
                # Check inbox frequently
                if time.time() - last_inbox_check > 5:
                    self.check_inbox_for_clips()
                    last_inbox_check = time.time()
                
                # Sleep in intervals
                for i in range(self.poll_interval_s):
                    if not self.running:
                        break
                    
                    # Check inbox every 5 seconds during wait
                    if i % 5 == 0:
                        self.check_inbox_for_clips()
                    
                    self.client.sleep(1)
                
                if self.running:
                    self.check_for_new_matches()
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log.exception(f"Error: {e}")
                self.status.update(last_error=str(e))
                time.sleep(30)
        
        self.log.info("Daemon stopping...")
        self.status.update(running=False, phase="stopped")
        
        try:
            self.client.disconnect()
        except Exception:
            pass
    
    def run_local(self, reprocess: bool = False):
        """Process existing demos without Steam login (local-only mode)
        
        Use this for testing or when you already have demo files.
        
        Args:
            reprocess: If True, re-analyze demos even if already processed
        """
        self.running = True
        self.log.info("=" * 60)
        self.log.info("LOCAL-ONLY MODE - Processing existing demos")
        self.log.info("=" * 60)
        self.log.info(f"Player: {self.player_name}")
        self.log.info(f"Demo dir: {self.demo_dir}")
        if reprocess:
            self.log.info("Reprocess mode: will re-analyze all demos")
        
        self.status.update(
            running=True,
            phase="analyzing",
            message="Local mode: processing existing demos..."
        )
        
        # Ensure directories
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all demo files
        demos = list(self.demo_dir.glob("*.dem"))
        self.log.info(f"Found {len(demos)} demo files")
        
        if not demos:
            self.log.warning("No demo files found in demo directory")
            self.status.update(running=False, phase="complete", message="No demos to process")
            return
        
        # Process each demo
        total_clips = 0
        total_oversprays = 0
        total_knife_deaths = 0
        processed = 0
        skipped = 0
        
        for demo_path in sorted(demos, key=lambda p: p.stat().st_mtime):
            demo_id = demo_path.stem
            
            # Skip if already processed (unless reprocess flag)
            if not reprocess and self.state.is_processed(demo_id):
                self.log.info(f"Skipping already processed: {demo_path.name}")
                skipped += 1
                continue
            
            self.log.info(f"\nAnalyzing: {demo_path.name}")
            self.log.info("-" * 40)
            self.status.update(
                current_demo=demo_path.name,
                message=f"Analyzing {demo_path.name}..."
            )
            
            try:
                clips = self.analyze_demo(demo_path)
                
                # Count by type
                overspray_clips = [c for c in clips if c.get("type") == "overspray"]
                knife_clips = [c for c in clips if c.get("type") == "knife_death"]
                
                if overspray_clips:
                    self.log.info(f"  🔫 Oversprays: {len(overspray_clips)}")
                    total_oversprays += len(overspray_clips)
                if knife_clips:
                    self.log.info(f"  🔪 Knife deaths: {len(knife_clips)}")
                    total_knife_deaths += len(knife_clips)
                
                if clips:
                    self.state.add_to_batch(demo_id, str(demo_path), clips)
                    total_clips += len(clips)
                else:
                    self.log.info(f"  ✅ No mistakes found")
                
                self.state.mark_processed(demo_id)
                processed += 1
                
            except Exception as e:
                self.log.error(f"  Error analyzing {demo_path.name}: {e}")
                continue
        
        # Print summary
        self.log.info("")
        self.log.info("=" * 60)
        self.log.info("ANALYSIS COMPLETE")
        self.log.info("=" * 60)
        self.log.info(f"Demos processed: {processed}")
        if skipped > 0:
            self.log.info(f"Demos skipped (already processed): {skipped}")
        self.log.info(f"Total mistakes found: {total_clips}")
        self.log.info(f"  🔫 Oversprays: {total_oversprays}")
        self.log.info(f"  🔪 Knife deaths: {total_knife_deaths}")
        
        # Show output location
        self.log.info("")
        self.log.info(f"Analysis results saved to: {self.output_dir}")
        
        # Send batch to Windows if we have clips and Windows is configured
        pending = self.state.get_pending_batch()
        if pending and self.windows_url:
            self.log.info("")
            self.log.info(f"Sending {total_clips} clips to Windows...")
            try:
                self.send_pending_batch()
                self.log.info("Batch sent successfully")
            except Exception as e:
                self.log.error(f"Failed to send batch: {e}")
        elif pending:
            self.log.info("")
            self.log.info("ℹ️  No Windows URL configured - clips not sent")
            self.log.info("   Configure WINDOWS_BASE_URL to enable clip recording")
            self.log.info("   Or use the Web UI to view analysis results")
        
        self.status.update(
            running=False,
            phase="complete",
            message=f"Processed {processed} demos: {total_oversprays} oversprays, {total_knife_deaths} knife deaths"
        )
        
        self.log.info("Local processing complete")
    
    def stop(self):
        self.running = False
    
    def get_status(self) -> dict:
        return self.status.status.to_dict()


# =========================
# HTTP API
# =========================

def create_api(daemon: DemoDaemon):
    if not HAS_FLASK:
        return None
    
    app = Flask(__name__)
    app.logger.setLevel(logging.WARNING)
    
    @app.route("/status")
    def status():
        return jsonify(daemon.get_status())
    
    @app.route("/check", methods=["POST"])
    def trigger_check():
        count = daemon.check_for_new_matches()
        return jsonify({"ok": True, "new_demos": count})
    
    @app.route("/process", methods=["POST"])
    def process_existing():
        """Process all unprocessed demos"""
        daemon.scan_existing_demos()
        return jsonify({"ok": True})
    
    @app.route("/send_batch", methods=["POST"])
    def send_batch():
        """Force send pending batch now"""
        pending = daemon.state.get_pending_batch()
        if not pending:
            return jsonify({"ok": False, "error": "No pending batch"})
        
        success = daemon.send_pending_batch()
        return jsonify({"ok": success})
    
    @app.route("/demos")
    def list_demos():
        demos = sorted(daemon.demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
        return jsonify({
            "count": len(demos),
            "demos": [{"name": d.name, "size_mb": round(d.stat().st_size / 1024 / 1024, 1)} for d in demos[:50]]
        })
    
    @app.route("/clips")
    def list_clips():
        clips = sorted(daemon.inbox_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        return jsonify({
            "count": len(clips),
            "clips": [{"name": c.name, "size_mb": round(c.stat().st_size / 1024 / 1024, 1)} for c in clips[:50]]
        })
    
    @app.route("/2fa")
    def get_2fa_code():
        """Generate current 2FA code (for testing shared_secret)"""
        if not daemon.shared_secret:
            return jsonify({"error": "STEAM_SHARED_SECRET not configured"}), 400
        
        code = generate_steam_totp(daemon.shared_secret)
        return jsonify({"code": code, "valid_for": 30 - (int(time.time()) % 30)})
    
    return app


# =========================
# Main
# =========================

def main():
    load_dotenv()
    
    ap = argparse.ArgumentParser(description="CS2 Demo Daemon - Download + Analyze + Clip")
    ap.add_argument("--player", required=True, help="Player name (e.g., Remag)")
    ap.add_argument("--demo-dir", type=Path, default=Path("./demos"))
    ap.add_argument("--output-dir", type=Path, default=Path("./output"))
    ap.add_argument("--inbox-dir", type=Path, default=Path("./inbox"))
    
    ap.add_argument("--gc-version", type=int, default=int(os.environ.get("GC_VERSION", "2000696")))
    ap.add_argument("--poll-interval", type=int, default=300, help="Seconds between checks")
    
    # Rate limiting settings
    ap.add_argument("--download-delay", type=int, 
                    default=int(os.environ.get("DOWNLOAD_DELAY_S", "30")),
                    help="Seconds between demo downloads (default: 30)")
    ap.add_argument("--rate-limit-delay", type=int,
                    default=int(os.environ.get("RATE_LIMIT_DELAY_S", "120")),
                    help="Seconds to wait when rate limited (default: 120)")
    
    ap.add_argument("--windows-url", default=os.environ.get("WINDOWS_BASE_URL", "http://10.0.0.108:8788"))
    ap.add_argument("--ubuntu-url", default=os.environ.get("UBUNTU_BASE_URL", "http://10.0.0.196:8787"))
    
    ap.add_argument("--api-port", type=int, default=8790)
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--local-only", action="store_true",
                    help="Process existing demos only, no Steam login/download")
    ap.add_argument("--reprocess", action="store_true",
                    help="Re-analyze demos even if already processed (use with --local-only)")
    
    args = ap.parse_args()
    
    log = setup_logging(args.debug)
    
    daemon = DemoDaemon(
        demo_dir=args.demo_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        inbox_dir=args.inbox_dir.expanduser().resolve(),
        player_name=args.player,
        gc_version=args.gc_version,
        poll_interval_s=args.poll_interval,
        windows_url=args.windows_url,
        ubuntu_url=args.ubuntu_url,
        clip_token=os.environ.get("CLIP_TOKEN", "token"),
        download_delay_s=args.download_delay,
        rate_limit_delay_s=args.rate_limit_delay,
        log=log,
    )
    
    def shutdown(sig, frame):
        log.info("Shutdown signal")
        daemon.stop()
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Start HTTP API
    if not args.no_api and HAS_FLASK:
        api = create_api(daemon)
        if api:
            api_thread = threading.Thread(
                target=lambda: api.run(host="0.0.0.0", port=args.api_port, debug=False, use_reloader=False),
                daemon=True,
            )
            api_thread.start()
            log.info(f"API: http://localhost:{args.api_port}")
    
    if args.local_only:
        # Process existing demos without Steam login
        daemon.run_local(reprocess=args.reprocess)
    else:
        daemon.run()


if __name__ == "__main__":
    main()
