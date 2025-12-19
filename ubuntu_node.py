# ubuntu_node.py
import os
import sys
from pathlib import Path
import requests
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Shared secret token (set same on Windows)
TOKEN = "token"

# Where Ubuntu stores received videos
INBOX_DIR = Path(os.environ.get("INBOX_DIR", "./inbox")).resolve()
INBOX_DIR.mkdir(parents=True, exist_ok=True)

# Where Windows server is (host:port), example: http://10.0.0.108:8788
WINDOWS_BASE_URL = os.environ.get("WINDOWS_BASE_URL", "").rstrip("/")

app = FastAPI()


def _safe_filename(name: str) -> str:
    # Why: prevent path traversal like ../../etc/passwd
    name = name.replace("\\", "/").split("/")[-1]
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip()
    if not safe:
        raise HTTPException(status_code=400, detail="Bad file name")
    return safe


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    x_token: str | None = Header(default=None),
):
    # Why: basic auth gate
    if x_token != TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")

    safe_name = _safe_filename(file.filename or "video.bin")
    out_path = INBOX_DIR / safe_name

    # Why: stream to disk; avoids loading large files into RAM
    with out_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return JSONResponse({"ok": True, "saved_to": str(out_path)})


def request_video_from_windows(windows_dir: str, ubuntu_base_url: str) -> dict:
    if not WINDOWS_BASE_URL:
        raise SystemExit("Set WINDOWS_BASE_URL, e.g. http://10.0.0.108:8788")

    # Windows will upload to this Ubuntu URL:
    upload_url = ubuntu_base_url.rstrip("/") + "/upload"

    r = requests.post(
        WINDOWS_BASE_URL + "/request",
        json={"windows_dir": windows_dir, "ubuntu_upload_url": upload_url},
        headers={"X-Token": TOKEN},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main():
    """
    Usage:
      1) Start receiver:
         python3 ubuntu_node.py serve --host 0.0.0.0 --port 8787

      2) Request a video from Windows:
         python3 ubuntu_node.py request --windows-dir "D:\\hlae_out\\take0012" --ubuntu-base-url "http://10.0.0.127:8787"
    """
    if len(sys.argv) < 2:
        print(main.__doc__)
        raise SystemExit(2)

    mode = sys.argv[1].lower()

    if mode == "serve":
        host = "0.0.0.0"
        port = 8787
        if "--host" in sys.argv:
            host = sys.argv[sys.argv.index("--host") + 1]
        if "--port" in sys.argv:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        uvicorn.run(app, host=host, port=port)
        return

    if mode == "request":
        if "--windows-dir" not in sys.argv or "--ubuntu-base-url" not in sys.argv:
            print(main.__doc__)
            raise SystemExit(2)

        windows_dir = sys.argv[sys.argv.index("--windows-dir") + 1]
        ubuntu_base_url = sys.argv[sys.argv.index("--ubuntu-base-url") + 1]

        resp = request_video_from_windows(windows_dir, ubuntu_base_url)
        print(resp)
        return

    print(main.__doc__)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
