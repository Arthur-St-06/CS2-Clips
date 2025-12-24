#!/usr/bin/env python3
"""
app.py - CS2 Coach Web UI

Shows daemon status, recent demos, and clips as they arrive.

Usage:
    streamlit run app_integrated.py
"""

from __future__ import annotations

import json
import os
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
    return Path(DEFAULT_CONFIG["OUTPUT_DIR"]).expanduser().resolve() / "daemon_status.json"


def read_daemon_status() -> dict:
    """Read daemon status from disk"""
    status_path = get_status_path()
    if not status_path.exists():
        return {"running": False, "phase": "not_started", "message": "Daemon not started"}
    
    try:
        return json.loads(status_path.read_text())
    except Exception as e:
        return {"running": False, "phase": "error", "message": f"Failed to read status: {e}"}


def get_daemon_status() -> dict:
    """Get status from demo daemon"""
    try:
        r = requests.get(DEFAULT_CONFIG["DAEMON_URL"] + "/status", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"running": False, "error": "Daemon not reachable"}


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
    
    st.dataframe(df[display_cols], width='stretch', hide_index=True)


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
    
    # Daemon controls
    st.header("📡 Daemon Controls")
    daemon_status = get_daemon_status()
    
    if daemon_status.get("running"):
        st.success("✅ Daemon running")
        
        if st.button("🔄 Check for new demos", width='stretch'):
            try:
                r = requests.post(DEFAULT_CONFIG["DAEMON_URL"] + "/check", timeout=60)
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
        
        if st.button("🔄 Reprocess existing demos", width='stretch'):
            try:
                r = requests.post(DEFAULT_CONFIG["DAEMON_URL"] + "/process", timeout=300)
                r.raise_for_status()
                st.success("Processing started!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Process failed: {e}")
    else:
        st.warning("⚠️ Daemon not running")
        st.code(f"python demo_daemon.py --player {DEFAULT_CONFIG['PLAYER_NAME']}", language="bash")
    
    st.divider()
    
    if st.button("🔄 Refresh", width='stretch'):
        st.rerun()
    
    st.divider()
    st.caption("Edit .env file to change settings")

# Read current status
status = read_daemon_status()
phase = status.get("phase", "not_started")
is_running = status.get("running", False)

# Main content
st.divider()

# Status section
st.header("📊 Status")

# Show daemon status
if is_running:
    col_status, col_phase = st.columns([1, 3])
    with col_status:
        if status.get("gc_ready"):
            st.success("🟢 Connected")
        elif status.get("steam_connected"):
            st.warning("🟡 Steam OK, GC pending")
        else:
            st.error("🔴 Connecting...")
    with col_phase:
        st.write(f"**{phase.title()}**: {status.get('message', '')}")
else:
    st.warning("⚠️ Daemon not running. Start with: `python demo_daemon.py --player Remag`")

# Stats row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Demos Downloaded", status.get("total_downloaded", 0))
col2.metric("Demos Analyzed", status.get("demos_analyzed", 0))

# Show pending vs requested based on phase
pending_clips = status.get("pending_clips", 0)
clips_requested = status.get("clips_requested", 0)

if pending_clips > 0 and not status.get("batch_sent"):
    col3.metric("Clips Pending", pending_clips)
else:
    col3.metric("Clips Requested", clips_requested)

col4.metric("Clips Received", status.get("clips_received", 0))

# Show pending batch info
if status.get("pending_demos", 0) > 0:
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.info(f"📦 Batch pending: {status.get('pending_demos', 0)} demos, {pending_clips} clips (will send when all demos downloaded)")
    with col_btn:
        if st.button("Send Now", type="primary"):
            try:
                r = requests.post(DEFAULT_CONFIG["DAEMON_URL"] + "/send_batch", timeout=60)
                r.raise_for_status()
                st.success("Batch sent!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

if status.get("last_check"):
    st.caption(f"Last check: {status.get('last_check', '')[:19]}")

st.divider()

# Tabs for different views
tab_clips, tab_demos, tab_log = st.tabs(["📹 Clips", "🎮 Demos", "📋 Status"])

with tab_clips:
    st.header("Recent Clips")
    
    inbox_dir = Path(DEFAULT_CONFIG["INBOX_DIR"]).expanduser().resolve()
    recent_clips = status.get("recent_clips", [])
    
    if recent_clips:
        st.write(f"**{len(recent_clips)} recent clips**")
        
        cols = st.columns(2)
        for i, clip in enumerate(recent_clips[:10]):
            with cols[i % 2]:
                st.write(f"**{clip.get('filename', 'Unknown')}**")
                clip_path = clip.get("path", "")
                if clip_path and Path(clip_path).exists():
                    try:
                        st.video(clip_path)
                    except Exception:
                        st.warning("Could not load video")
                else:
                    st.info(f"File: {clip_path}")
                st.caption(f"Received: {clip.get('received_at', '')[:19]}")
    else:
        # Fallback: check inbox directly
        clips = sorted(inbox_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if clips:
            st.write(f"**{len(clips)} clips in inbox**")
            cols = st.columns(2)
            for i, clip in enumerate(clips[:10]):
                with cols[i % 2]:
                    st.write(f"**{clip.name}**")
                    try:
                        st.video(str(clip))
                    except Exception:
                        st.warning("Could not load video")
        else:
            st.info("No clips received yet. Waiting for Windows to process...")

with tab_demos:
    st.header("Recent Demos")
    
    recent_demos = status.get("recent_demos", [])
    
    if recent_demos:
        df = pd.DataFrame(recent_demos)
        display_cols = [c for c in ["map", "downloaded_at", "size_mb"] if c in df.columns]
        if display_cols:
            st.dataframe(df[display_cols], width='stretch', hide_index=True)
    else:
        demo_dir = Path(DEFAULT_CONFIG["DEMO_DIR"]).expanduser().resolve()
        demos = list(demo_dir.glob("*.dem"))
        if demos:
            st.write(f"Found {len(demos)} demo files")
            demo_list = [{"name": d.name, "size_mb": round(d.stat().st_size / 1024 / 1024, 1)} for d in sorted(demos, key=lambda p: p.stat().st_mtime, reverse=True)[:20]]
            st.dataframe(pd.DataFrame(demo_list), width='stretch', hide_index=True)
        else:
            st.info("No demos found. Daemon will download them automatically.")

with tab_log:
    st.header("Daemon Status")
    
    st.json(status)
    
    # Also show last error if any
    if status.get("last_error"):
        st.error(f"Last error: {status.get('last_error')}")

# Auto-refresh
if is_running or phase not in ["stopped", "error"]:
    refresh_seconds = DEFAULT_CONFIG["AUTO_REFRESH_SECONDS"]
    st.caption(f"🔄 Auto-refreshing every {refresh_seconds} seconds...")
    time.sleep(refresh_seconds)
    st.rerun()
