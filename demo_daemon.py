#!/usr/bin/env python3
"""
demo_daemon.py - Persistent Steam demo download service

Stays logged into Steam and automatically downloads new matches as they appear.
2FA is only required ONCE when starting the daemon.

Usage:
    # Start the daemon (will prompt for 2FA interactively)
    python demo_daemon.py --demo-dir ./demos

    # Run in background with nohup
    nohup python demo_daemon.py --demo-dir ./demos > demo_daemon.log 2>&1 &

    # Check status
    curl http://localhost:8790/status

    # Trigger manual check
    curl -X POST http://localhost:8790/check

Features:
    - Stays logged into Steam indefinitely
    - Polls for new matches every N minutes (default: 5)
    - Downloads new demos automatically
    - HTTP API for status and manual triggers
    - Saves state to resume after restart
"""

from gevent import monkey
monkey.patch_all()

import argparse
import bz2
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict

from steam.client import SteamClient
from steam.enums.common import EResult

from csgo.client import CSGOClient
from csgo import sharecode
from csgo.enums import EGCBaseClientMsg, ECsgoGCMsg

# Optional: HTTP API
try:
    from flask import Flask, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


# =========================
# Configuration
# =========================

@dataclass
class DaemonState:
    """Persistent state saved to disk"""
    last_sharecode: str = ""
    last_check: str = ""
    total_downloaded: int = 0
    demos_list: list = None
    
    def __post_init__(self):
        if self.demos_list is None:
            self.demos_list = []
    
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


# =========================
# Logging
# =========================

def setup_logging(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("demo_daemon")
    if not debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("steam").setLevel(logging.WARNING)
    return log


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


def download_file(url: str, out_path: Path, timeout_s: int = 120, log: logging.Logger = None) -> bool:
    import requests
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if log:
        log.info(f"Downloading: {url}")
    
    try:
        with requests.get(url, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        if log:
            log.error(f"Download failed: {e}")
        return False


def get_next_sharecode_webapi(
    web_api_key: str,
    steamid64: str,
    auth_code: str,
    known_code: str,
    log: logging.Logger,
) -> Optional[str]:
    """Get next sharecode using Valve WebAPI"""
    import requests
    
    url = "https://api.steampowered.com/ICSGOPlayers_730/GetNextMatchSharingCode/v1/"
    params = {
        "key": web_api_key,
        "steamid": steamid64,
        "steamidkey": auth_code,
        "knowncode": known_code,
    }
    
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        
        result = data.get("result", {})
        nextcode = result.get("nextcode", "")
        
        if isinstance(nextcode, str) and nextcode.startswith("CSGO-"):
            return nextcode
    except Exception as e:
        log.warning(f"WebAPI error: {e}")
    
    return None


# =========================
# Demo Daemon
# =========================

class DemoDaemon:
    def __init__(
        self,
        demo_dir: Path,
        state_path: Path,
        gc_version: int,
        poll_interval_s: int = 300,  # 5 minutes
        log: logging.Logger = None,
    ):
        self.demo_dir = demo_dir
        self.gc_version = gc_version
        self.poll_interval_s = poll_interval_s
        self.log = log or logging.getLogger("demo_daemon")
        
        self.state = StateManager(state_path)
        
        self.client = SteamClient()
        self.cs = CSGOClient(self.client)
        
        self.gc_ready = False
        self.running = False
        self.last_error = None
        
        # Config from env
        self.steam_web_api_key = os.environ.get("STEAM_WEB_API_KEY", "").strip()
        self.target_steamid64 = os.environ.get("TARGET_STEAMID64", "").strip()
        self.target_auth_code = os.environ.get("TARGET_AUTH_CODE", "").strip()
        self.start_code = os.environ.get("TARGET_KNOWN_CODE", "").strip()
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self.client.on("disconnected")
        def on_disconnect():
            self.log.warning("Steam disconnected")
            self.gc_ready = False
        
        @self.client.on("reconnect")
        def on_reconnect(delay):
            self.log.info(f"Reconnecting in {delay}s...")
        
        @self.cs.on(EGCBaseClientMsg.EMsgGCClientWelcome)
        def on_gc_welcome(msg):
            self.gc_ready = True
            self.log.info("GC ready")
        
        @self.cs.on(ECsgoGCMsg.EMsgGCCStrike15_v2_ClientLogonFatalError)
        def on_gc_fatal(msg):
            self.log.error("GC fatal error - may need to restart")
            self.gc_ready = False
    
    def login(self) -> bool:
        """Interactive login with 2FA prompt"""
        self.log.info("Logging into Steam (interactive)...")
        
        user = os.environ.get("BOT_STEAM_USER", "").strip()
        pw = os.environ.get("BOT_STEAM_PASS", "").strip()
        
        if user and pw:
            # Try with saved credentials first
            result = self.client.relogin()
            if result == EResult.OK:
                self.log.info("Relogin successful (using saved session)")
                return True
            
            # Interactive login
            self.log.info("Enter Steam Guard code when prompted...")
            result = self.client.cli_login(user, pw)
        else:
            # Full interactive
            result = self.client.cli_login()
        
        if result != EResult.OK:
            self.log.error(f"Login failed: {result}")
            return False
        
        self.log.info("Steam login successful")
        return True
    
    def connect_gc(self) -> bool:
        """Connect to CS2 Game Coordinator"""
        self.log.info("Connecting to CS2 GC...")
        
        self.client.games_played([730])
        
        # Send hello until welcome
        start = time.time()
        timeout = 60
        
        while time.time() - start < timeout:
            if self.gc_ready:
                self.log.info("GC connection established")
                return True
            
            self.cs.send(EGCBaseClientMsg.EMsgGCClientHello, {"version": self.gc_version})
            self.client.sleep(3)
        
        self.log.error("GC connection timeout")
        return False
    
    def download_demo(self, matchid: int, outcomeid: int, token: int) -> bool:
        """Download a single demo via GC"""
        dem_path = self.demo_dir / f"{matchid}.dem"
        bz2_path = self.demo_dir / f"{matchid}.dem.bz2"
        
        if dem_path.exists():
            self.log.info(f"Already have {dem_path.name}")
            return False
        
        self.log.info(f"Requesting match info for {matchid}...")
        self.cs.request_full_match_info(matchid, outcomeid, token)
        
        ev = self.cs.wait_event("full_match_info", timeout=60)
        if not ev:
            self.log.error(f"Timeout getting match info for {matchid}")
            return False
        
        msg = ev[0]
        d = MessageToDict(msg, preserving_proto_field_name=True)
        
        # Save match info
        info_path = self.demo_dir / f"{matchid}_info.json"
        info_path.write_text(json.dumps(d, indent=2))
        
        # Find demo URL
        url = find_demo_url(d)
        if not url:
            self.log.error(f"No demo URL found for {matchid}")
            return False
        
        # Download
        if not download_file(url, bz2_path, log=self.log):
            return False
        
        # Decompress if bz2
        try:
            dem_path.write_bytes(bz2.decompress(bz2_path.read_bytes()))
            bz2_path.unlink()
            self.log.info(f"Downloaded and decompressed: {dem_path.name}")
        except Exception:
            self.log.info(f"Downloaded (not bz2): {bz2_path.name}")
        
        return True
    
    def check_for_new_matches(self) -> int:
        """Check for and download new matches. Returns count of new demos."""
        if not self.gc_ready:
            if not self.connect_gc():
                return 0
        
        # Determine starting sharecode
        known_code = self.state.state.last_sharecode or self.start_code
        if not known_code:
            self.log.warning("No starting sharecode configured")
            return 0
        
        if not all([self.steam_web_api_key, self.target_steamid64, self.target_auth_code]):
            self.log.warning("WebAPI credentials not configured - can only download starting match")
            # Just try the start code
            try:
                matchid, outcomeid, token = decode_share_code(known_code)
                if self.download_demo(matchid, outcomeid, token):
                    self.state.update(
                        last_sharecode=known_code,
                        last_check=datetime.now().isoformat(),
                        total_downloaded=self.state.state.total_downloaded + 1,
                    )
                    return 1
            except Exception as e:
                self.log.error(f"Failed to download: {e}")
            return 0
        
        # Walk forward through sharecodes
        downloaded = 0
        current = known_code
        max_new = 20  # Don't download more than 20 at once
        
        self.log.info(f"Checking for new matches starting from {current[:20]}...")
        
        for _ in range(max_new):
            next_code = get_next_sharecode_webapi(
                self.steam_web_api_key,
                self.target_steamid64,
                self.target_auth_code,
                current,
                self.log,
            )
            
            if not next_code:
                self.log.info("No more new matches")
                break
            
            self.log.info(f"New match found: {next_code}")
            
            try:
                matchid, outcomeid, token = decode_share_code(next_code)
                if self.download_demo(matchid, outcomeid, token):
                    downloaded += 1
                    
                    # Update state
                    demos_list = self.state.state.demos_list or []
                    demos_list.append({
                        "sharecode": next_code,
                        "matchid": matchid,
                        "downloaded_at": datetime.now().isoformat(),
                    })
                    
                    self.state.update(
                        last_sharecode=next_code,
                        total_downloaded=self.state.state.total_downloaded + 1,
                        demos_list=demos_list[-100:],  # Keep last 100
                    )
            except Exception as e:
                self.log.error(f"Failed to download {next_code}: {e}")
            
            current = next_code
            time.sleep(2)  # Be nice to Valve
        
        self.state.update(last_check=datetime.now().isoformat())
        
        if downloaded > 0:
            self.log.info(f"Downloaded {downloaded} new demos")
        
        return downloaded
    
    def run(self):
        """Main daemon loop"""
        self.running = True
        self.log.info("Demo daemon starting...")
        
        # Initial login
        if not self.login():
            self.log.error("Failed to login - exiting")
            return
        
        # Connect to GC
        if not self.connect_gc():
            self.log.error("Failed to connect to GC - exiting")
            return
        
        self.log.info(f"Daemon running. Checking every {self.poll_interval_s}s")
        self.log.info(f"Demos will be saved to: {self.demo_dir}")
        
        # Initial check
        self.check_for_new_matches()
        
        # Main loop
        while self.running:
            try:
                # Sleep in small intervals to allow shutdown
                for _ in range(self.poll_interval_s):
                    if not self.running:
                        break
                    self.client.sleep(1)
                
                if self.running:
                    self.check_for_new_matches()
                    
            except KeyboardInterrupt:
                self.log.info("Interrupted")
                break
            except Exception as e:
                self.log.exception(f"Error in main loop: {e}")
                self.last_error = str(e)
                time.sleep(30)  # Wait before retry
        
        self.log.info("Daemon stopping...")
        try:
            self.client.disconnect()
        except Exception:
            pass
    
    def stop(self):
        self.running = False
    
    def get_status(self) -> dict:
        return {
            "running": self.running,
            "gc_ready": self.gc_ready,
            "last_check": self.state.state.last_check,
            "last_sharecode": self.state.state.last_sharecode,
            "total_downloaded": self.state.state.total_downloaded,
            "last_error": self.last_error,
            "demo_dir": str(self.demo_dir),
            "poll_interval_s": self.poll_interval_s,
        }


# =========================
# HTTP API (optional)
# =========================

def create_api(daemon: DemoDaemon):
    if not HAS_FLASK:
        return None
    
    app = Flask(__name__)
    
    @app.route("/status")
    def status():
        return jsonify(daemon.get_status())
    
    @app.route("/check", methods=["POST"])
    def trigger_check():
        count = daemon.check_for_new_matches()
        return jsonify({"ok": True, "new_demos": count})
    
    @app.route("/demos")
    def list_demos():
        demos = sorted(daemon.demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
        return jsonify({
            "count": len(demos),
            "demos": [{"name": d.name, "size_mb": round(d.stat().st_size / 1024 / 1024, 1)} for d in demos[:50]]
        })
    
    return app


# =========================
# Main
# =========================

def main():
    load_dotenv()
    
    ap = argparse.ArgumentParser(description="CS2 Demo Download Daemon")
    ap.add_argument("--demo-dir", type=Path, default=Path("./demos"), help="Where to save demos")
    ap.add_argument("--gc-version", type=int, default=int(os.environ.get("GC_VERSION", "2000696")))
    ap.add_argument("--poll-interval", type=int, default=300, help="Seconds between checks (default: 300)")
    ap.add_argument("--api-port", type=int, default=8790, help="HTTP API port (default: 8790)")
    ap.add_argument("--no-api", action="store_true", help="Disable HTTP API")
    ap.add_argument("--debug", action="store_true")
    
    args = ap.parse_args()
    
    log = setup_logging(args.debug)
    
    demo_dir = args.demo_dir.expanduser().resolve()
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    state_path = demo_dir / ".daemon_state.json"
    
    daemon = DemoDaemon(
        demo_dir=demo_dir,
        state_path=state_path,
        gc_version=args.gc_version,
        poll_interval_s=args.poll_interval,
        log=log,
    )
    
    # Handle shutdown
    def shutdown(sig, frame):
        log.info("Shutdown signal received")
        daemon.stop()
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Start HTTP API in background
    if not args.no_api and HAS_FLASK:
        api = create_api(daemon)
        if api:
            api_thread = threading.Thread(
                target=lambda: api.run(host="0.0.0.0", port=args.api_port, debug=False, use_reloader=False),
                daemon=True,
            )
            api_thread.start()
            log.info(f"HTTP API running on http://localhost:{args.api_port}")
    
    # Run daemon
    daemon.run()


if __name__ == "__main__":
    main()
