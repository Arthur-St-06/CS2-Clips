# ubuntu_node.py
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# =========================
# Config
# =========================
TOKEN = os.environ.get("CLIP_TOKEN", "token")
WINDOWS_BASE_URL = os.environ.get("WINDOWS_BASE_URL", "http://10.0.0.108:8788").rstrip("/")

INBOX_DIR = Path("./inbox").resolve()
INBOX_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# =========================
# Server: receive clip
# =========================
def _safe_name(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1]
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip()
    if not safe:
        raise HTTPException(400, "Bad filename")
    return safe


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    x_token: str | None = Header(default=None),
):
    if x_token != TOKEN:
        raise HTTPException(401, "Bad token")

    fname = _safe_name(file.filename or "clip.mp4")
    out = INBOX_DIR / fname

    with out.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return JSONResponse({"ok": True, "saved": str(out)})


# =========================
# Client: request clip(s)
# =========================
def _require_windows_base_url() -> None:
    if not WINDOWS_BASE_URL:
        raise SystemExit("WINDOWS_BASE_URL not set (e.g. http://10.0.0.108:8788)")


def request_clip(
    demo_id: str,
    username: str,
    ubuntu_base_url: str,
    start_s: float | None = None,
    duration_s: float | None = None,
    clips: List[Dict[str, float]] | None = None,
) -> dict:
    """
    If clips is provided, sends multi-clip request:
      clips=[{"start_s": 40.0, "duration_s": 10.0}, ...]
    Else sends single clip request with start_s/duration_s.
    """
    _require_windows_base_url()

    upload_url = ubuntu_base_url.rstrip("/") + "/upload"

    payload: Dict[str, Any] = {
        "demo_id": demo_id,
        "username": username,
        "ubuntu_upload_url": upload_url,
    }

    if clips is not None:
        if not clips:
            raise SystemExit("clips list is empty")
        payload["clips"] = [{"start_s": float(c["start_s"]), "duration_s": float(c["duration_s"])} for c in clips]
    else:
        if start_s is None or duration_s is None:
            raise SystemExit("start_s and duration_s are required for single clip mode")
        payload["start_s"] = float(start_s)
        payload["duration_s"] = float(duration_s)

    r = requests.post(
        WINDOWS_BASE_URL + "/clip",
        json=payload,
        headers={"X-Token": TOKEN},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


# =========================
# CLI parsing helpers
# =========================
def _get_arg(flag: str) -> str:
    if flag not in sys.argv:
        raise SystemExit(f"Missing {flag}")
    return sys.argv[sys.argv.index(flag) + 1]


def _parse_multi_clips() -> List[Dict[str, float]]:
    """
    Parse repeated: --clip <start_s> <duration_s>
    Example:
      --clip 40 10 --clip 70.5 6
    """
    clips: List[Dict[str, float]] = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--clip":
            if i + 2 >= len(sys.argv):
                raise SystemExit("Each --clip needs 2 values: --clip <start_s> <duration_s>")
            try:
                s = float(sys.argv[i + 1])
                d = float(sys.argv[i + 2])
            except ValueError:
                raise SystemExit("Invalid number in --clip <start_s> <duration_s>")
            clips.append({"start_s": s, "duration_s": d})
            i += 3
        else:
            i += 1
    if not clips:
        raise SystemExit("No --clip entries provided")
    return clips


# =========================
# CLI
# =========================
def main():
    """
    Serve (receiver):
      python ubuntu_node.py serve --host 0.0.0.0 --port 8787

    Single clip request:
      python ubuntu_node.py clip \
        --demo-id demo392 \
        --username Remag \
        --start-s 40 \
        --duration-s 10 \
        --ubuntu-base-url http://10.0.0.196:8787

    Multi-clip request (one Windows run, multiple segments):
      python ubuntu_node.py clips \
        --demo-id demo392 \
        --username Remag \
        --ubuntu-base-url http://10.0.0.196:8787 \
        --clip 40 10 \
        --clip 70.5 6 \
        --clip 120 8
    """
    if len(sys.argv) < 2:
        print(main.__doc__)
        sys.exit(2)

    cmd = sys.argv[1]

    if cmd == "serve":
        host = "0.0.0.0"
        port = 8787
        if "--host" in sys.argv:
            host = _get_arg("--host")
        if "--port" in sys.argv:
            port = int(_get_arg("--port"))

        uvicorn.run(app, host=host, port=port)
        return

    if cmd == "clip":
        demo_id = _get_arg("--demo-id")
        username = _get_arg("--username")
        start_s = float(_get_arg("--start-s"))
        duration_s = float(_get_arg("--duration-s"))
        ubuntu_base_url = _get_arg("--ubuntu-base-url")

        resp = request_clip(
            demo_id=demo_id,
            username=username,
            ubuntu_base_url=ubuntu_base_url,
            start_s=start_s,
            duration_s=duration_s,
            clips=None,
        )
        print(resp)
        return

    if cmd == "clips":
        demo_id = _get_arg("--demo-id")
        username = _get_arg("--username")
        ubuntu_base_url = _get_arg("--ubuntu-base-url")
        clips = _parse_multi_clips()

        resp = request_clip(
            demo_id=demo_id,
            username=username,
            ubuntu_base_url=ubuntu_base_url,
            clips=clips,
        )
        print(resp)
        return

    print(main.__doc__)
    sys.exit(2)


if __name__ == "__main__":
    main()
