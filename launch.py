#!/usr/bin/env python3
"""
CS2 Coaching Clip Generator - Launch Script

This script starts all Ubuntu-side services needed for the automated
CS2 coaching clip pipeline.

Services started:
  1. Demo Daemon     - Downloads demos from Steam, analyzes for mistakes
  2. Demo Server     - Serves .dem files to Windows (port 8789)
  3. Clip Receiver   - Receives clips from Windows (port 8787)
  4. Web UI          - Streamlit dashboard (port 8791)

Usage:
    python launch.py                    # Start all services
    python launch.py --no-ui            # Start without web UI
    python launch.py --stop             # Stop all services

Requirements:
    - .env file with configuration (copy from .env.template)
    - All Python dependencies installed
    - Windows machine running clip_server_batch.py
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# =========================
# Configuration
# =========================

PLAYER_NAME = os.environ.get("PLAYER_NAME", "Remag")
DEMO_DIR = Path(os.environ.get("DEMO_DIR", "./demos")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./output")).expanduser().resolve()
INBOX_DIR = Path(os.environ.get("INBOX_DIR", "./inbox")).expanduser().resolve()

DEMO_SERVER_PORT = int(os.environ.get("DEMO_SERVER_PORT", "8789"))
CLIP_RECEIVER_PORT = int(os.environ.get("CLIP_RECEIVER_PORT", "8787"))
DAEMON_PORT = int(os.environ.get("DAEMON_PORT", "8790"))
WEB_UI_PORT = int(os.environ.get("WEB_UI_PORT", "8791"))

PID_FILE = OUTPUT_DIR / ".launcher_pids.txt"

# =========================
# Process Management
# =========================

processes = []


def start_process(name: str, cmd: list, log_file: Path = None) -> subprocess.Popen:
    """Start a subprocess with optional log file"""
    print(f"  Starting {name}...")
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    
    processes.append((name, proc))
    print(f"  ✓ {name} started (PID: {proc.pid})")
    return proc


def stop_all():
    """Stop all managed processes"""
    print("\nStopping services...")
    
    for name, proc in processes:
        try:
            if proc.poll() is None:
                print(f"  Stopping {name} (PID: {proc.pid})...")
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception as e:
            print(f"  Warning: Could not stop {name}: {e}")
    
    # Also try to stop from PID file
    if PID_FILE.exists():
        try:
            pids = PID_FILE.read_text().strip().split("\n")
            for line in pids:
                if ":" in line:
                    name, pid = line.split(":", 1)
                    try:
                        os.killpg(int(pid), signal.SIGTERM)
                        print(f"  Stopped {name} (PID: {pid})")
                    except (ProcessLookupError, ValueError):
                        pass
            PID_FILE.unlink()
        except Exception:
            pass
    
    print("All services stopped.")


def save_pids():
    """Save PIDs to file for later cleanup"""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PID_FILE, "w") as f:
        for name, proc in processes:
            f.write(f"{name}:{proc.pid}\n")


def check_requirements():
    """Check that required files exist"""
    required_files = [
        "demo_daemon.py",
        "demo_server.py", 
        "ubuntu_node.py",
        "app_integrated.py",
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)
    
    # Check .env
    if not Path(".env").exists():
        print("WARNING: No .env file found. Copy .env.template to .env and configure it.")
        print()


def print_status():
    """Print service status"""
    print()
    print("=" * 60)
    print("CS2 Coaching Clip Generator - Running")
    print("=" * 60)
    print()
    print("Services:")
    for name, proc in processes:
        status = "✓ Running" if proc.poll() is None else "✗ Stopped"
        print(f"  {name}: {status} (PID: {proc.pid})")
    print()
    print("Endpoints:")
    print(f"  Web UI:         http://localhost:{WEB_UI_PORT}")
    print(f"  Daemon API:     http://localhost:{DAEMON_PORT}/status")
    print(f"  Demo Server:    http://localhost:{DEMO_SERVER_PORT}")
    print(f"  Clip Receiver:  http://localhost:{CLIP_RECEIVER_PORT}")
    print()
    print("Logs:")
    print(f"  {OUTPUT_DIR}/daemon.log")
    print(f"  {OUTPUT_DIR}/demo_server.log")
    print(f"  {OUTPUT_DIR}/clip_receiver.log")
    print()
    print("Press Ctrl+C to stop all services")
    print("=" * 60)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="CS2 Coaching Clip Generator Launcher")
    parser.add_argument("--no-ui", action="store_true", help="Don't start web UI")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--player", default=PLAYER_NAME, help="Player name")
    args = parser.parse_args()
    
    if args.stop:
        stop_all()
        return
    
    # Ensure directories exist
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    
    check_requirements()
    
    print()
    print("=" * 60)
    print("CS2 Coaching Clip Generator")
    print("=" * 60)
    print()
    print(f"Player: {args.player}")
    print(f"Demo directory: {DEMO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Starting services...")
    print()
    
    # 1. Demo Server (serves .dem files to Windows)
    start_process(
        "Demo Server",
        [
            sys.executable, "demo_server.py", "serve",
            "--demo-dir", str(DEMO_DIR),
            "--port", str(DEMO_SERVER_PORT),
        ],
        log_file=OUTPUT_DIR / "demo_server.log",
    )
    
    # 2. Clip Receiver (receives clips from Windows)
    start_process(
        "Clip Receiver",
        [
            sys.executable, "ubuntu_node.py", "serve",
            "--host", "0.0.0.0",
            "--port", str(CLIP_RECEIVER_PORT),
            "--inbox", str(INBOX_DIR),
        ],
        log_file=OUTPUT_DIR / "clip_receiver.log",
    )
    
    # Give servers time to start
    time.sleep(1)
    
    # 3. Demo Daemon (main service - downloads, analyzes, requests clips)
    start_process(
        "Demo Daemon",
        [
            sys.executable, "demo_daemon.py",
            "--player", args.player,
            "--demo-dir", str(DEMO_DIR),
            "--output-dir", str(OUTPUT_DIR),
            "--inbox-dir", str(INBOX_DIR),
            "--api-port", str(DAEMON_PORT),
        ],
        log_file=OUTPUT_DIR / "daemon.log",
    )
    
    # 4. Web UI (Streamlit)
    if not args.no_ui:
        start_process(
            "Web UI",
            [
                sys.executable, "-m", "streamlit", "run",
                "app_integrated.py",
                "--server.port", str(WEB_UI_PORT),
                "--server.headless", "true",
            ],
        )
    
    # Save PIDs for cleanup
    save_pids()
    
    # Wait a moment for everything to start
    time.sleep(2)
    
    print_status()
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Monitor processes
    try:
        while True:
            time.sleep(5)
            
            # Check if any process died
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\nWARNING: {name} stopped unexpectedly (exit code: {proc.returncode})")
                    print(f"Check log: {OUTPUT_DIR}/{name.lower().replace(' ', '_')}.log")
    except KeyboardInterrupt:
        pass
    
    stop_all()


if __name__ == "__main__":
    main()
