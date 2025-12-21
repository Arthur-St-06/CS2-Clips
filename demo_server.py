#!/usr/bin/env python3
"""
demo_server.py - HTTP server for sharing demo files from Ubuntu to Windows

Ubuntu serves demos via HTTP. Windows downloads on-demand when creating clips.
This avoids manual file syncing and works through firewalls.

Usage (Ubuntu):
    python demo_server.py serve --demo-dir ./demos --port 8789

    # With authentication
    DEMO_TOKEN=secret python demo_server.py serve --demo-dir ./demos

Usage (test download):
    python demo_server.py download --url http://ubuntu:8789 --demo-id match123 --out ./local_demos

Features:
    - List available demos
    - Download demos by ID (filename stem)
    - Optional token authentication
    - Range requests for resumable downloads
    - Caching headers for efficiency
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import uvicorn

# =========================
# Configuration
# =========================

TOKEN = os.environ.get("DEMO_TOKEN", "")  # Empty = no auth required
DEMO_DIR = Path(os.environ.get("DEMO_DIR", "./demos")).resolve()
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming

app = FastAPI(title="CS2 Demo Server")
log = logging.getLogger("demo_server")


# =========================
# Helpers
# =========================

def _check_token(x_token: Optional[str]) -> None:
    """Validate token if authentication is enabled"""
    if TOKEN and x_token != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


def _get_demo_dir() -> Path:
    """Get configured demo directory"""
    return DEMO_DIR


def _list_demos(demo_dir: Path) -> list[dict]:
    """List all .dem files with metadata"""
    demos = []
    for p in sorted(demo_dir.glob("*.dem")):
        stat = p.stat()
        demos.append({
            "id": p.stem,
            "filename": p.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "md5": None,  # Computed on request to avoid startup delay
        })
    return demos


def _find_demo(demo_id: str, demo_dir: Path) -> Optional[Path]:
    """Find demo by ID (stem) or exact filename"""
    # Try exact stem match
    candidate = demo_dir / f"{demo_id}.dem"
    if candidate.exists():
        return candidate
    
    # Try as full filename
    candidate = demo_dir / demo_id
    if candidate.exists() and candidate.suffix == ".dem":
        return candidate
    
    # Partial match (contains)
    for p in demo_dir.glob("*.dem"):
        if demo_id in p.stem:
            return p
    
    return None


def _compute_md5(path: Path) -> str:
    """Compute MD5 hash of file"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================
# API Endpoints
# =========================

@app.get("/")
def root():
    """Health check"""
    return {"status": "ok", "service": "demo_server"}


@app.get("/demos")
def list_demos(x_token: Optional[str] = Header(default=None)):
    """List all available demos"""
    _check_token(x_token)
    demo_dir = _get_demo_dir()
    
    if not demo_dir.exists():
        raise HTTPException(status_code=500, detail=f"Demo directory not found: {demo_dir}")
    
    demos = _list_demos(demo_dir)
    return {
        "ok": True,
        "demo_dir": str(demo_dir),
        "count": len(demos),
        "demos": demos,
    }


@app.get("/demos/{demo_id}")
def get_demo_info(demo_id: str, x_token: Optional[str] = Header(default=None)):
    """Get info about a specific demo"""
    _check_token(x_token)
    demo_dir = _get_demo_dir()
    
    path = _find_demo(demo_id, demo_dir)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Demo not found: {demo_id}")
    
    stat = path.stat()
    return {
        "ok": True,
        "id": path.stem,
        "filename": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "md5": _compute_md5(path),
    }


@app.get("/demos/{demo_id}/download")
async def download_demo(
    demo_id: str,
    request: Request,
    x_token: Optional[str] = Header(default=None),
):
    """
    Download a demo file.
    Supports Range requests for resumable downloads.
    """
    _check_token(x_token)
    demo_dir = _get_demo_dir()
    
    path = _find_demo(demo_id, demo_dir)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Demo not found: {demo_id}")
    
    file_size = path.stat().st_size
    
    # Check for Range header (resumable download)
    range_header = request.headers.get("range")
    
    if range_header:
        # Parse range header: "bytes=start-end"
        try:
            range_spec = range_header.replace("bytes=", "")
            if "-" in range_spec:
                parts = range_spec.split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else file_size - 1
            else:
                start = int(range_spec)
                end = file_size - 1
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Range header")
        
        if start >= file_size or end >= file_size or start > end:
            raise HTTPException(status_code=416, detail="Range not satisfiable")
        
        # Stream partial content
        def iter_file():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk_size = min(CHUNK_SIZE, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type="application/octet-stream",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(end - start + 1),
                "Accept-Ranges": "bytes",
                "Content-Disposition": f'attachment; filename="{path.name}"',
            },
        )
    
    # Full file download
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )


@app.get("/demos/{demo_id}/md5")
def get_demo_md5(demo_id: str, x_token: Optional[str] = Header(default=None)):
    """Get MD5 hash of demo (for cache validation)"""
    _check_token(x_token)
    demo_dir = _get_demo_dir()
    
    path = _find_demo(demo_id, demo_dir)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Demo not found: {demo_id}")
    
    return {"ok": True, "id": path.stem, "md5": _compute_md5(path)}


# =========================
# Client Functions (for Windows)
# =========================

class DemoClient:
    """Client for downloading demos from Ubuntu server"""
    
    def __init__(self, base_url: str, token: str = "", cache_dir: Optional[Path] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.cache_dir = cache_dir or Path("./demo_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _headers(self) -> dict:
        h = {}
        if self.token:
            h["X-Token"] = self.token
        return h
    
    def list_demos(self) -> list[dict]:
        """List available demos on server"""
        r = requests.get(f"{self.base_url}/demos", headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json().get("demos", [])
    
    def get_demo_info(self, demo_id: str) -> dict:
        """Get demo info including MD5"""
        r = requests.get(f"{self.base_url}/demos/{demo_id}", headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    
    def get_remote_md5(self, demo_id: str) -> str:
        """Get MD5 of remote demo"""
        r = requests.get(f"{self.base_url}/demos/{demo_id}/md5", headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json().get("md5", "")
    
    def _local_path(self, demo_id: str) -> Path:
        """Get local cache path for demo"""
        return self.cache_dir / f"{demo_id}.dem"
    
    def _local_md5_path(self, demo_id: str) -> Path:
        """Get path for cached MD5"""
        return self.cache_dir / f"{demo_id}.dem.md5"
    
    def _is_cached(self, demo_id: str) -> bool:
        """Check if demo is cached and valid"""
        local = self._local_path(demo_id)
        md5_file = self._local_md5_path(demo_id)
        
        if not local.exists() or not md5_file.exists():
            return False
        
        try:
            cached_md5 = md5_file.read_text().strip()
            remote_md5 = self.get_remote_md5(demo_id)
            return cached_md5 == remote_md5
        except Exception:
            return False
    
    def download(self, demo_id: str, force: bool = False) -> Path:
        """
        Download demo to cache. Returns local path.
        Uses cache if available and valid.
        """
        local = self._local_path(demo_id)
        md5_file = self._local_md5_path(demo_id)
        
        # Check cache
        if not force and self._is_cached(demo_id):
            log.info(f"Using cached demo: {local}")
            return local
        
        # Get remote info
        info = self.get_demo_info(demo_id)
        remote_md5 = info.get("md5", "")
        total_size = info.get("size_bytes", 0)
        
        log.info(f"Downloading {demo_id} ({info.get('size_mb', '?')} MB)...")
        
        # Download with progress
        r = requests.get(
            f"{self.base_url}/demos/{demo_id}/download",
            headers=self._headers(),
            stream=True,
            timeout=600,
        )
        r.raise_for_status()
        
        downloaded = 0
        with open(local, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  Progress: {pct:.1f}% ({downloaded // (1024*1024)} MB)", end="", flush=True)
        
        print()  # Newline after progress
        
        # Verify MD5
        local_md5 = _compute_md5(local)
        if remote_md5 and local_md5 != remote_md5:
            local.unlink()
            raise RuntimeError(f"MD5 mismatch! Expected {remote_md5}, got {local_md5}")
        
        # Save MD5 for cache validation
        md5_file.write_text(local_md5)
        
        log.info(f"Downloaded: {local}")
        return local
    
    def ensure_demo(self, demo_id: str) -> Path:
        """Ensure demo is available locally (download if needed)"""
        return self.download(demo_id, force=False)


# =========================
# CLI
# =========================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    ap = argparse.ArgumentParser(description="CS2 Demo File Server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    # Serve command
    serve_p = sub.add_parser("serve", help="Run demo server (Ubuntu)")
    serve_p.add_argument("--demo-dir", type=Path, default=Path("./demos"),
                         help="Directory containing .dem files")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8789)
    
    # List command
    list_p = sub.add_parser("list", help="List demos on server")
    list_p.add_argument("--url", required=True, help="Demo server URL")
    list_p.add_argument("--token", default="", help="Auth token")
    
    # Download command
    dl_p = sub.add_parser("download", help="Download demo from server")
    dl_p.add_argument("--url", required=True, help="Demo server URL")
    dl_p.add_argument("--demo-id", required=True, help="Demo ID to download")
    dl_p.add_argument("--out", type=Path, default=Path("./demo_cache"),
                      help="Output directory")
    dl_p.add_argument("--token", default="", help="Auth token")
    dl_p.add_argument("--force", action="store_true", help="Force re-download")
    
    args = ap.parse_args()
    
    if args.cmd == "serve":
        global DEMO_DIR
        DEMO_DIR = args.demo_dir.expanduser().resolve()
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Serving demos from: {DEMO_DIR}")
        log.info(f"Auth: {'enabled' if TOKEN else 'disabled'}")
        uvicorn.run(app, host=args.host, port=args.port)
    
    elif args.cmd == "list":
        client = DemoClient(args.url, token=args.token)
        demos = client.list_demos()
        print(f"Found {len(demos)} demos:")
        for d in demos:
            print(f"  {d['id']}: {d['size_mb']} MB ({d['modified']})")
    
    elif args.cmd == "download":
        client = DemoClient(args.url, token=args.token, cache_dir=args.out)
        path = client.download(args.demo_id, force=args.force)
        print(f"Demo saved to: {path}")


if __name__ == "__main__":
    main()
