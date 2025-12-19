# ubuntu_node.py
import os
import sys
from pathlib import Path
import requests
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# =========================
# Config
# =========================
TOKEN = os.environ.get("CLIP_TOKEN", "devtoken")
WINDOWS_BASE_URL = os.environ.get("WINDOWS_BASE_URL", "").rstrip("/")

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
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    return JSONResponse({"ok": True, "saved": str(out)})


# =========================
# Client: request clip
# =========================
def request_clip(
    demo_id: str,
    username: str,
    start_s: float,
    duration_s: float,
    ubuntu_base_url: str,
) -> dict:
    if not WINDOWS_BASE_URL:
        raise SystemExit("WINDOWS_BASE_URL not set")

    upload_url = ubuntu_base_url.rstrip("/") + "/upload"

    r = requests.post(
        WINDOWS_BASE_URL + "/clip",
        json={
            "demo_id": demo_id,
            "username": username,
            "start_s": float(start_s),
            "duration_s": float(duration_s),
            "ubuntu_upload_url": upload_url,
        },
        headers={"X-Token": TOKEN},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


# =========================
# CLI
# =========================
def main():
    """
    Serve:
      python ubuntu_node.py serve --host 0.0.0.0 --port 8787

    Request clip:
      python ubuntu_node.py clip \
        --demo-id demo392 \
        --username Remag \
        --start-s 40 \
        --duration-s 10 \
        --ubuntu-base-url http://10.0.0.127:8787
    """
    if len(sys.argv) < 2:
        print(main.__doc__)
        sys.exit(2)

    cmd = sys.argv[1]

    if cmd == "serve":
        host = "0.0.0.0"
        port = 8787
        if "--host" in sys.argv:
            host = sys.argv[sys.argv.index("--host") + 1]
        if "--port" in sys.argv:
            port = int(sys.argv[sys.argv.index("--port") + 1])

        uvicorn.run(app, host=host, port=port)
        return

    if cmd == "clip":
        demo_id = sys.argv[sys.argv.index("--demo-id") + 1]
        username = sys.argv[sys.argv.index("--username") + 1]
        start_s = float(sys.argv[sys.argv.index("--start-s") + 1])
        duration_s = float(sys.argv[sys.argv.index("--duration-s") + 1])
        ubuntu_base_url = sys.argv[sys.argv.index("--ubuntu-base-url") + 1]

        resp = request_clip(demo_id, username, start_s, duration_s, ubuntu_base_url)
        print(resp)
        return

    print(main.__doc__)
    sys.exit(2)


if __name__ == "__main__":
    main()
