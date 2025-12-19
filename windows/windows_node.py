# windows_node.py
import os
from pathlib import Path
import time
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

TOKEN = os.environ.get("CLIP_TOKEN", "devtoken")

# Safety root: only allow requests for directories inside this folder.
# Example: set WIN_ALLOWED_ROOT=D:\hlae_out
WIN_ALLOWED_ROOT = Path(os.environ.get("WIN_ALLOWED_ROOT", r"D:\hlae_out")).resolve()

# Which file types count as "video"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}

app = FastAPI()


def _is_under_root(p: Path, root: Path) -> bool:
    # Why: block path traversal / exfil
    try:
        p.relative_to(root)
        return True
    except Exception:
        return False


def _pick_video_file(dir_path: Path) -> Path:
    # Prefer video files; choose newest
    candidates = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            candidates.append(p)

    if not candidates:
        # fallback: if directory has exactly one file, send it
        files = [p for p in dir_path.iterdir() if p.is_file()]
        if len(files) == 1:
            return files[0]
        raise HTTPException(status_code=404, detail=f"No video found in {dir_path}")

    return max(candidates, key=lambda x: x.stat().st_mtime)


def _upload_file(ubuntu_upload_url: str, file_path: Path) -> dict:
    # Why: requests streaming upload
    with file_path.open("rb") as f:
        r = requests.post(
            ubuntu_upload_url,
            files={"file": (file_path.name, f, "application/octet-stream")},
            headers={"X-Token": TOKEN},
            timeout=300,
        )
    r.raise_for_status()
    return r.json()


@app.post("/request")
def request_video(payload: dict, x_token: str | None = Header(default=None)):
    if x_token != TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")

    windows_dir = payload.get("windows_dir")
    ubuntu_upload_url = payload.get("ubuntu_upload_url")

    if not windows_dir or not ubuntu_upload_url:
        raise HTTPException(status_code=400, detail="Need windows_dir and ubuntu_upload_url")

    dir_path = Path(windows_dir).resolve()

    # Security: only allow within WIN_ALLOWED_ROOT
    if not _is_under_root(dir_path, WIN_ALLOWED_ROOT):
        raise HTTPException(
            status_code=403,
            detail=f"Requested dir is outside WIN_ALLOWED_ROOT ({WIN_ALLOWED_ROOT})",
        )

    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Dir not found: {dir_path}")

    video_path = _pick_video_file(dir_path)

    # Optional small delay if file is still being written
    time.sleep(0.2)

    resp = _upload_file(ubuntu_upload_url, video_path)
    return JSONResponse({"ok": True, "sent": str(video_path), "ubuntu_response": resp})


def main():
    """
    Run:
      set CLIP_TOKEN=devtoken
      set WIN_ALLOWED_ROOT=D:\hlae_out
      python windows_node.py
    """
    uvicorn.run(app, host="0.0.0.0", port=8788)


if __name__ == "__main__":
    main()
