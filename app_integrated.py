#!/usr/bin/env python3
"""
app_integrated.py - CS2 Coach Web UI

Clean UI showing analysis results and coaching tips.

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

DEFAULT_CONFIG = {
    "PLAYER_NAME": os.environ.get("PLAYER_NAME", "Remag"),
    "DEMO_DIR": os.environ.get("DEMO_DIR", "./demos"),
    "OUTPUT_DIR": os.environ.get("OUTPUT_DIR", "./output"),
    "INBOX_DIR": os.environ.get("INBOX_DIR", "./inbox"),
    "DAEMON_URL": os.environ.get("DAEMON_URL", "http://localhost:8790"),
    "AUTO_REFRESH_SECONDS": int(os.environ.get("AUTO_REFRESH_SECONDS", "10")),
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


def get_analysis_summary(output_dir: Path, player_name: str = "") -> dict:
    """Read analysis results and compute summary statistics for all mistake types"""
    summary = {
        # Overspray stats
        "overspray_count": 0,
        "avg_bullets": 0,
        "avg_spray_duration": 0,
        "avg_time_to_death": 0,
        # Knife death stats
        "knife_death_count": 0,
        # General
        "demos_analyzed": 0,
        "clips_exported": 0,
    }
    
    if not output_dir.exists():
        return summary
    
    all_oversprays = []
    all_knife_deaths = []
    demos_with_data = set()
    
    # Read all analysis files
    for demo_dir in output_dir.iterdir():
        if demo_dir.is_dir():
            # Overspray candidates
            overspray_file = demo_dir / "overspray_candidates.parquet"
            if overspray_file.exists():
                try:
                    df = pd.read_parquet(overspray_file)
                    if not df.empty:
                        if player_name and "player" in df.columns:
                            df = df[df["player"].str.contains(player_name, case=False, na=False)]
                        if not df.empty:
                            all_oversprays.append(df)
                            demos_with_data.add(demo_dir.name)
                except Exception:
                    pass
            
            # Knife deaths
            knife_file = demo_dir / "knife_deaths.parquet"
            if knife_file.exists():
                try:
                    df = pd.read_parquet(knife_file)
                    if not df.empty:
                        if player_name and "player" in df.columns:
                            df = df[df["player"].str.contains(player_name, case=False, na=False)]
                        if not df.empty:
                            all_knife_deaths.append(df)
                            demos_with_data.add(demo_dir.name)
                except Exception:
                    pass
    
    summary["demos_analyzed"] = len(demos_with_data)
    
    # Overspray stats
    if all_oversprays:
        combined = pd.concat(all_oversprays, ignore_index=True)
        summary["overspray_count"] = len(combined)
        if len(combined) > 0:
            summary["avg_bullets"] = combined["bullets"].mean()
            summary["avg_spray_duration"] = combined["duration_s"].mean()
            if "time_to_death_after_burst_s" in combined.columns:
                summary["avg_time_to_death"] = combined["time_to_death_after_burst_s"].mean()
    
    # Knife death stats
    if all_knife_deaths:
        combined_knife = pd.concat(all_knife_deaths, ignore_index=True)
        summary["knife_death_count"] = len(combined_knife)
    
    # Count clips in inbox
    inbox_dir = Path(DEFAULT_CONFIG["INBOX_DIR"]).expanduser().resolve()
    if inbox_dir.exists():
        summary["clips_exported"] = len(list(inbox_dir.glob("*.mp4")))
    
    return summary


# =========================
# Main UI
# =========================

st.set_page_config(
    page_title="CS2 Coach",
    page_icon="🎮",
    layout="wide",
)

# Custom CSS - hide running indicator and clean up
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; }
    h1 { margin-top: 0 !important; }
    
    /* Hide the running/stop indicator in top right */
    [data-testid="stStatusWidget"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    
    /* Clean metric styling */
    [data-testid="stMetricValue"] { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "player_name" not in st.session_state:
    st.session_state.player_name = DEFAULT_CONFIG["PLAYER_NAME"]

# Sidebar - simplified
with st.sidebar:
    st.header("🎮 CS2 Coach")
    
    # Editable player name
    new_player = st.text_input(
        "Player Name", 
        value=st.session_state.player_name,
        help="Your in-game name to filter analysis"
    )
    if new_player != st.session_state.player_name:
        st.session_state.player_name = new_player
        st.rerun()
    
    st.divider()
    
    if st.button("🔄 Refresh", width='stretch'):
        st.rerun()

# Read current status
status = read_daemon_status()
phase = status.get("phase", "not_started")
is_running = status.get("running", False)

# Header with status
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("🎮 CS2 Coach")
    st.caption(f"Analyzing mistakes for player: **{st.session_state.player_name}**")
with col_status:
    if is_running:
        if status.get("gc_ready"):
            st.success("🟢 Connected")
        elif status.get("steam_connected"):
            st.warning("🟡 Connecting...")
        else:
            st.info("🔵 Starting...")
    else:
        st.error("🔴 Offline")

st.divider()

# Status message
if phase == "complete":
    st.success(f"✅ {status.get('message', 'Complete!')}")
elif phase in ["downloading", "analyzing", "requesting", "receiving"]:
    st.info(f"⏳ {status.get('message', phase.title() + '...')}")
elif phase == "error":
    st.error(f"❌ {status.get('last_error', 'Unknown error')}")

# Stats row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Demos Downloaded", status.get("total_downloaded", 0))
col2.metric("Demos Analyzed", status.get("demos_analyzed", 0))

pending_clips = status.get("pending_clips", 0)
clips_requested = status.get("clips_requested", 0)
if pending_clips > 0 and not status.get("batch_sent"):
    col3.metric("Clips Pending", pending_clips)
else:
    col3.metric("Clips Requested", clips_requested)

col4.metric("Clips Received", status.get("clips_received", 0))

# Get analysis summary
output_dir = Path(DEFAULT_CONFIG["OUTPUT_DIR"]).expanduser().resolve()
summary = get_analysis_summary(output_dir, st.session_state.player_name)

st.divider()

# Mistake Analysis Tabs - only show when we have data
has_oversprays = summary["overspray_count"] > 0
has_knife_deaths = summary["knife_death_count"] > 0

if summary["clips_exported"] > 0 or has_oversprays or has_knife_deaths:
    
    # Create tabs for different mistake types
    mistake_tabs = st.tabs(["🔫 Oversprays", "🔪 Knife Deaths", "📹 Clips", "🎮 Demos"])
    
    # === Overspray Tab ===
    with mistake_tabs[0]:
        if has_oversprays:
            st.markdown(f"""
### Detected {summary['overspray_count']} overspray cases

You kept spraying, got no kill, and died soon after.

**Stats from exported clips:**
- Avg bullets fired: **{summary['avg_bullets']:.1f}**
- Avg spray duration: **{summary['avg_spray_duration']:.2f}s**
- Avg time-to-death after burst: **{summary['avg_time_to_death']:.2f}s**
            """)
            
            with st.expander("💡 How to Fix", expanded=True):
                st.markdown("""
**Try instead:**
- 🎯 **Burst 3–5 bullets, then reset.** Don't commit to full sprays unless you're confident in the kill.
- 🏃 **Strafe/reposition after first contact** instead of committing to a long spray. Movement beats aim in many situations.
- ⏸️ **If you whiff, stop shooting briefly** to regain accuracy, then re-peek intentionally. Panic spraying rarely works.
- 🔄 **Counter-strafe before shooting** to ensure first-bullet accuracy on your re-peek.
                """)
        else:
            st.info("No overspray mistakes detected. Great spray control! 🎯")
    
    # === Knife Death Tab ===
    with mistake_tabs[1]:
        if has_knife_deaths:
            st.markdown(f"""
### Detected {summary['knife_death_count']} knife death cases

You died while holding your knife instead of a weapon.
            """)
            
            with st.expander("💡 How to Fix", expanded=True):
                st.markdown("""
**Try instead:**
- 🔫 **Always hold a gun when enemies might appear.** The speed boost from knife isn't worth dying for.
- 🗺️ **Know your timings.** Only pull out knife when you're 100% sure no enemy can reach you.
- 👂 **Listen for audio cues.** Footsteps, reloads, or utility usage means enemies are close - switch to gun.
- 🚪 **Gun out near corners and chokepoints.** These are high-danger zones where enemies commonly appear.
- ⏱️ **After plant/defuse situations**, always expect enemies to push - never knife out.
- 🏃 **If you need speed**, consider a pistol instead - nearly as fast but you can fight back.
                """)
        else:
            st.info("No knife deaths detected. Good weapon discipline! 🔫")
    
    # === Clips Tab ===
    with mistake_tabs[2]:
        inbox_dir = Path(DEFAULT_CONFIG["INBOX_DIR"]).expanduser().resolve()
        
        if inbox_dir.exists():
            clips = sorted(inbox_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            clips = []
        
        if clips:
            st.write(f"**{len(clips)} clips available**")
            
            cols = st.columns(2)
            for i, clip in enumerate(clips[:10]):
                with cols[i % 2]:
                    display_name = clip.stem
                    if len(display_name) > 40:
                        display_name = display_name[:37] + "..."
                    
                    st.write(f"**{display_name}**")
                    try:
                        st.video(str(clip))
                    except Exception:
                        st.warning("Could not load video")
                    
                    size_mb = clip.stat().st_size / 1024 / 1024
                    mtime = datetime.fromtimestamp(clip.stat().st_mtime).strftime("%H:%M:%S")
                    st.caption(f"{size_mb:.1f} MB • {mtime}")
        else:
            if is_running and phase in ["requesting", "receiving"]:
                st.info("⏳ Waiting for clips from Windows...")
            else:
                st.info("No clips yet. Run the daemon to analyze demos and generate clips.")
    
    # === Demos Tab ===
    with mistake_tabs[3]:
        demo_dir = Path(DEFAULT_CONFIG["DEMO_DIR"]).expanduser().resolve()
        
        if demo_dir.exists():
            demos = sorted(demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            demos = []
        
        if demos:
            st.write(f"**{len(demos)} demo files**")
            
            demo_data = []
            for d in demos[:20]:
                demo_data.append({
                    "Name": d.stem[:40] + ("..." if len(d.stem) > 40 else ""),
                    "Size (MB)": round(d.stat().st_size / 1024 / 1024, 1),
                    "Modified": datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
            
            st.dataframe(pd.DataFrame(demo_data), hide_index=True, width='stretch')
        else:
            st.info("No demos found. The daemon will download them automatically.")

else:
    # No data yet - show simple tabs
    tab_clips, tab_demos = st.tabs(["📹 Clips", "🎮 Demos"])
    
    with tab_clips:
        inbox_dir = Path(DEFAULT_CONFIG["INBOX_DIR"]).expanduser().resolve()
        
        if inbox_dir.exists():
            clips = sorted(inbox_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            clips = []
        
        if clips:
            st.write(f"**{len(clips)} clips available**")
            
            cols = st.columns(2)
            for i, clip in enumerate(clips[:10]):
                with cols[i % 2]:
                    display_name = clip.stem
                    if len(display_name) > 40:
                        display_name = display_name[:37] + "..."
                    
                    st.write(f"**{display_name}**")
                    try:
                        st.video(str(clip))
                    except Exception:
                        st.warning("Could not load video")
                    
                    size_mb = clip.stat().st_size / 1024 / 1024
                    mtime = datetime.fromtimestamp(clip.stat().st_mtime).strftime("%H:%M:%S")
                    st.caption(f"{size_mb:.1f} MB • {mtime}")
        else:
            if is_running and phase in ["requesting", "receiving"]:
                st.info("⏳ Waiting for clips from Windows...")
            else:
                st.info("No clips yet. Run the daemon to analyze demos and generate clips.")
    
    with tab_demos:
        demo_dir = Path(DEFAULT_CONFIG["DEMO_DIR"]).expanduser().resolve()
        
        if demo_dir.exists():
            demos = sorted(demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            demos = []
        
        if demos:
            st.write(f"**{len(demos)} demo files**")
            
            demo_data = []
            for d in demos[:20]:
                demo_data.append({
                    "Name": d.stem[:40] + ("..." if len(d.stem) > 40 else ""),
                    "Size (MB)": round(d.stat().st_size / 1024 / 1024, 1),
                    "Modified": datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
            
            st.dataframe(pd.DataFrame(demo_data), hide_index=True, width='stretch')
        else:
            st.info("No demos found. The daemon will download them automatically.")

# Auto-refresh (silent, no visual indicator)
if is_running and phase not in ["complete", "error", "idle"]:
    time.sleep(DEFAULT_CONFIG["AUTO_REFRESH_SECONDS"])
    st.rerun()
