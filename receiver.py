# receiver.py
import os
from pathlib import Path
import requests
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

APP_TOKEN = os.environ.get("CLIP_TOKEN", "devtoken")
CLIPS_DIR = Path(os.environ.get("CLIPS_DIR", "./clips")).resolve()
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# Where Ubuntu should fetch from on Windows:
WINDOWS_BASE_URL = os.environ.get("WINDOWS_BASE_URL", "http://10.0.0.123:8788").rstrip("/")

app = FastAPI()

def _safe_name(name: str) -> str:
    # Why: prevent path traversal like ../../etc/passwd
    name = name.replace("\\", "/").split("/")[-1]
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip()
    if not safe:
        raise HTTPException(status_code=400, detail="Bad file name")
    return safe

def _auth(authorization: str | None) -> None:
    # Why: prevent unauthorized uploads/fetches
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    if token != APP_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")

@app.post("/upload")
async def upload_clip(
    file: UploadFile = File(...),
    x_clip_name: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    _auth(authorization)

    clip_name = _safe_name(x_clip_name or file.filename or "clip.mp4")
    if not clip_name.lower().endswith((".mp4", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Only video files allowed")

    out_path = CLIPS_DIR / clip_name

    # Why: stream to disk to avoid loading big videos into memory
    with out_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return JSONResponse({"ok": True, "saved_as": str(out_path), "name": clip_name})

@app.post("/fetch")
def fetch_from_windows(
    name: str,
    authorization: str | None = Header(default=None),
):
    """
    Caller: POST /fetch?name=out.mp4
    Receiver: pulls the file from Windows and stores it locally.
    """
    _auth(authorization)
    safe = _safe_name(name)

    url = f"{WINDOWS_BASE_URL}/files/{safe}"
    headers = {"Authorization": f"Bearer {APP_TOKEN}"}

    out_path = CLIPS_DIR / safe

    # Why: stream download to disk (no RAM blowups for big files)
    try:
        with requests.get(url, headers=headers, stream=True, timeout=300) as r:
            if r.status_code == 404:
                raise HTTPException(status_code=404, detail="Windows does not have that file")
            r.raise_for_status()

            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            tmp.replace(out_path)

    except requests.RequestException as e:
        # Why: make failures visible to caller
        raise HTTPException(status_code=502, detail=f"Failed to fetch from Windows: {e}")

    return JSONResponse({"ok": True, "name": safe, "saved_as": str(out_path)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8787)
