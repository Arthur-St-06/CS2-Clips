
# receiver.py
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

APP_TOKEN = os.environ.get("CLIP_TOKEN", "devtoken")  # set a real one later
CLIPS_DIR = Path(os.environ.get("CLIPS_DIR", "./clips")).resolve()
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

def _safe_name(name: str) -> str:
    # Why: prevent path traversal like ../../etc/passwd
    name = name.replace("\\", "/").split("/")[-1]
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip()

@app.post("/upload")
async def upload_clip(
    file: UploadFile = File(...),
    x_clip_name: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    # Why: basic auth so random people can’t upload files
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    if token != APP_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8787)
