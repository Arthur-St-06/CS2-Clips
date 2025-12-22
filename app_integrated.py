#!/usr/bin/env python3
"""
app.py - CS2 Coach Web UI with auto-starting pipeline

Features:
- Auto-starts data gathering when page opens
- Shows demo info as soon as available
- Shows clips as they arrive
- Auto-refreshes to show progress

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =========================
# Configuration
# =========================

# These can be overridden via environment variables
DEFAULT_CONFIG = {
    "PLAYER_NAME": os.environ.get("PLAYER_NAME", "Remag"),
    "DEMO_DIR": os.environ.get("DEMO_DIR", "./demos"),
    "OUTPUT_DIR": os.environ.get("OUTPUT_DIR", "./output"),
    "INBOX_DIR": os.environ.get("INBOX_DIR", "./inbox"),
    "WINDOWS_URL": os.environ.get("WINDOWS_BASE_URL", "http://10.0.0.108:8788"),
    "UBUNTU_URL": os.environ.get("UBUNTU_BASE_URL", "http://10.0.0.196:8787"),
    "DAEMON_URL": os.environ.get("DAEMON_URL", "http://localhost:8790"),
    "AUTO_REFRESH_SECONDS": int(os.environ.get("AUTO_REFRESH_SECONDS", "5")),
}


# =========================
# Status reading
# =========================

def get_status_path() -> Path:
    return Path(DEFAULT_CONFIG["OUTPUT_DIR"]).expanduser().resolve() / "pipeline_status.json"


def read_pipeline_status() -> dict:
    """Read pipeline status from disk"""
    status_path = get_status_path()
    if not status_path.exists():
        return {"phase": "not_started", "message": "Pipeline not started yet"}
    
    try:
        return json.loads(status_path.read_text())
    except Exception as e:
        return {"phase": "error", "message": f"Failed to read status: {e}"}


def is_pipeline_running() -> bool:
    """Check if pipeline is currently running"""
    status = read_pipeline_status()
    phase = status.get("phase", "")
    return phase not in ["not_started", "complete", "error", "idle"]


def get_lock_path() -> Path:
    return Path(DEFAULT_CONFIG["OUTPUT_DIR"]).expanduser().resolve() / "pipeline.lock"


def is_pipeline_locked() -> bool:
    """Check if pipeline lock exists (another instance running)"""
    lock = get_lock_path()
    if not lock.exists():
        return False
    
    # Check if lock is stale (older than 1 hour)
    try:
        age = time.time() - lock.stat().st_mtime
        if age > 3600:
            lock.unlink()
            return False
    except Exception:
        pass
    
    return True


def get_daemon_status() -> dict:
    """Get status from demo daemon"""
    try:
        r = requests.get(DEFAULT_CONFIG["DAEMON_URL"] + "/status", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"running": False, "error": "Daemon not reachable"}


# =========================
# Pipeline control
# =========================

def start_pipeline_background():
    """Start the pipeline in a background process"""
    if is_pipeline_locked():
        return False
    
    # Create lock
    lock = get_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    
    demo_dir = Path(DEFAULT_CONFIG["DEMO_DIR"]).expanduser().resolve()
    out_dir = Path(DEFAULT_CONFIG["OUTPUT_DIR"]).expanduser().resolve()
    inbox_dir = Path(DEFAULT_CONFIG["INBOX_DIR"]).expanduser().resolve()
    
    cmd = [
        "python", "pipeline_runner.py",
        "--player", DEFAULT_CONFIG["PLAYER_NAME"],
        "--demo-dir", str(demo_dir),
        "--out", str(out_dir),
        "--inbox", str(inbox_dir),
        "--windows-url", DEFAULT_CONFIG["WINDOWS_URL"],
        "--ubuntu-url", DEFAULT_CONFIG["UBUNTU_URL"],
    ]
    
    # Start subprocess
    log_path = out_dir / "pipeline.log"
    with open(log_path, "w") as log_file:
        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    
    return True


def reset_pipeline():
    """Reset pipeline status and clear lock"""
    try:
        get_lock_path().unlink(missing_ok=True)
    except Exception:
        pass
    
    try:
        status_path = get_status_path()
        if status_path.exists():
            status_path.unlink()
    except Exception:
        pass


# =========================
# UI Helpers
# =========================

def format_datetime(iso_str: Optional[str]) -> str:
    if not iso_str:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_str


def phase_to_emoji(phase: str) -> str:
    return {
        "not_started": "⏸️",
        "idle": "⏸️",
        "starting": "🚀",
        "downloading": "⬇️",
        "indexing": "📋",
        "analyzing": "🔍",
        "requesting": "📤",
        "receiving": "📥",
        "complete": "✅",
        "error": "❌",
    }.get(phase, "❓")


def show_phase_progress(status: dict):
    """Show pipeline phase progress"""
    phase = status.get("phase", "unknown")
    message = status.get("message", "")
    
    phases = ["downloading", "indexing", "analyzing", "requesting", "receiving", "complete"]
    
    if phase == "error":
        st.error(f"❌ Error: {status.get('error', message)}")
        return
    
    if phase in ["not_started", "idle"]:
        st.info("⏸️ Pipeline not running")
        return
    
    # Progress bar
    try:
        current_idx = phases.index(phase) if phase in phases else 0
        progress = (current_idx + 1) / len(phases)
    except ValueError:
        progress = 0.1
    
    st.progress(progress)
    st.write(f"{phase_to_emoji(phase)} **{phase.title()}**: {message}")


def show_demos_table(demos_info: list):
    """Show indexed demos table"""
    if not demos_info:
        return
    
    df = pd.DataFrame(demos_info)
    
    # Format columns
    if "played_at" in df.columns:
        df["played_at"] = df["played_at"].apply(format_datetime)
    
    if "size_mb" in df.columns:
        df["size_mb"] = df["size_mb"].apply(lambda x: f"{x:.1f} MB")
    
    display_cols = ["map", "played_at", "size_mb"]
    display_cols = [c for c in display_cols if c in df.columns]
    
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


def show_clips_grid(clips_info: list, inbox_dir: Path):
    """Show received clips in a grid"""
    if not clips_info:
        # Check inbox directly
        clips = sorted(inbox_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not clips:
            st.info("No clips received yet...")
            return
        
        clips_info = [{"filename": c.name, "path": str(c)} for c in clips]
    
    st.write(f"**{len(clips_info)} clips received**")
    
    # Show clips in columns
    cols = st.columns(2)
    for i, clip in enumerate(clips_info):
        with cols[i % 2]:
            st.write(f"**{clip['filename']}**")
            try:
                st.video(clip["path"])
            except Exception as e:
                st.warning(f"Could not load video: {e}")


# =========================
# Main UI
# =========================

st.set_page_config(
    page_title="CS2 Coach",
    page_icon="🎮",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; }
    h1 { margin-top: 0 !important; }
    .stProgress > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎮 CS2 Coach")
st.caption(f"Analyzing mistakes for player: **{DEFAULT_CONFIG['PLAYER_NAME']}**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.text_input("Player Name", value=DEFAULT_CONFIG["PLAYER_NAME"], key="cfg_player", disabled=True)
    st.text_input("Demo Directory", value=DEFAULT_CONFIG["DEMO_DIR"], key="cfg_demo_dir", disabled=True)
    st.text_input("Windows Server", value=DEFAULT_CONFIG["WINDOWS_URL"], key="cfg_windows", disabled=True)
    
    st.divider()
    
    # Daemon status
    st.header("📡 Demo Daemon")
    daemon_status = get_daemon_status()
    
    if daemon_status.get("running"):
        st.success("✅ Daemon running")
        st.caption(f"Last check: {daemon_status.get('last_check', 'Never')[:19]}")
        st.caption(f"Total downloaded: {daemon_status.get('total_downloaded', 0)}")
        
        if st.button("🔄 Check for new demos", use_container_width=True):
            try:
                r = requests.post(DEFAULT_CONFIG["DAEMON_URL"] + "/check", timeout=30)
                r.raise_for_status()
                result = r.json()
                if result.get("new_demos", 0) > 0:
                    st.success(f"Downloaded {result['new_demos']} new demos!")
                else:
                    st.info("No new demos available")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Check failed: {e}")
    else:
        st.warning("⚠️ Daemon not running")
        st.caption("Start with: `python demo_daemon.py`")
    
    st.divider()
    
    st.caption("To change settings, edit environment variables or .env file")
    
    st.divider()
    
    # Manual controls
    st.header("🔧 Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Reset", use_container_width=True):
            reset_pipeline()
            st.success("Pipeline reset!")
            st.rerun()
    
    if st.button("▶️ Start Pipeline", use_container_width=True, type="primary"):
        if start_pipeline_background():
            st.success("Pipeline started!")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("Pipeline already running")

# Read current status
status = read_pipeline_status()
phase = status.get("phase", "not_started")

# Auto-start pipeline if not running and not complete
if phase == "not_started" and not is_pipeline_locked():
    st.info("🚀 Starting pipeline automatically...")
    if start_pipeline_background():
        time.sleep(2)
        st.rerun()

# Main content
st.divider()

# Status section
st.header("📊 Pipeline Status")
show_phase_progress(status)

# Stats row
if status.get("demos_indexed", 0) > 0 or status.get("clips_received", 0) > 0:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Demos Indexed", status.get("demos_indexed", 0))
    col2.metric("Demos Analyzed", status.get("demos_analyzed", 0))
    col3.metric("Clips Requested", status.get("clips_requested", 0))
    col4.metric("Clips Received", status.get("clips_received", 0))

st.divider()

# Tabs for different views
tab_clips, tab_demos, tab_log = st.tabs(["📹 Clips", "🎮 Demos", "📋 Log"])

with tab_clips:
    st.header("Received Clips")
    
    inbox_dir = Path(DEFAULT_CONFIG["INBOX_DIR"]).expanduser().resolve()
    clips_info = status.get("clips_info", [])
    
    show_clips_grid(clips_info, inbox_dir)

with tab_demos:
    st.header("Indexed Demos")
    
    demos_info = status.get("demos_info", [])
    
    if demos_info:
        show_demos_table(demos_info)
    else:
        demo_dir = Path(DEFAULT_CONFIG["DEMO_DIR"]).expanduser().resolve()
        demos = list(demo_dir.glob("*.dem"))
        if demos:
            st.write(f"Found {len(demos)} demo files (not yet indexed)")
        else:
            st.info("No demos found. Enable download or add demos to folder.")

with tab_log:
    st.header("Pipeline Log")
    
    log_path = Path(DEFAULT_CONFIG["OUTPUT_DIR"]).expanduser().resolve() / "pipeline.log"
    
    if log_path.exists():
        try:
            log_content = log_path.read_text()
            # Show last 100 lines
            lines = log_content.strip().split("\n")
            recent = "\n".join(lines[-100:])
            st.code(recent, language="text")
        except Exception as e:
            st.warning(f"Could not read log: {e}")
    else:
        st.info("No log file yet")

# Auto-refresh while pipeline is running
if phase not in ["complete", "error", "not_started", "idle"]:
    refresh_seconds = DEFAULT_CONFIG["AUTO_REFRESH_SECONDS"]
    st.caption(f"🔄 Auto-refreshing every {refresh_seconds} seconds...")
    time.sleep(refresh_seconds)
    st.rerun()

# Show completion message
if phase == "complete":
    st.success("✅ Pipeline complete! All clips have been received.")
    st.balloons()
