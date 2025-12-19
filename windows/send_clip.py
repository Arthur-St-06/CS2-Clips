# windows_file_server.py
import os
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
import uvicorn

APP_TOKEN = os.environ.get("CLIP_TOKEN", "devtoken")
BASE_DIR = Path(os.environ.get("HLAE_OUT_DIR", r"D:\hlae_out")).resolve()

app = FastAPI()

def _safe_name(name: str) -> str:
    # Why: avoid path traversal (../../) and weird separators
    name = name.replace("\\", "/").split("/")[-1]
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip()
    if not safe:
        raise HTTPException(status_code=400, detail="Bad file name")
    return safe

def _auth(authorization: str | None) -> None:
    # Why: don't expose your file server to anyone on the network
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    if token != APP_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")

@app.get("/files/{name}")
def get_file(name: str, authorization: str | None = Header(default=None)):
    _auth(authorization)
    safe = _safe_name(name)

    # Why: keep it simple for v1: only serve from one directory
    p = (BASE_DIR / safe).resolve()

    # Why: ensure resolved path still stays under BASE_DIR
    if BASE_DIR not in p.parents and p != BASE_DIR:
        raise HTTPException(status_code=403, detail="Forbidden path")

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Why: FileResponse streams efficiently (no full file load into RAM)
    return FileResponse(
        path=str(p),
        media_type="video/mp4",     # ok even if it's mkv; client treats as bytes
        filename=p.name,
    )

if __name__ == "__main__":
    # Expose on LAN. In production you’d run behind HTTPS/VPN.
    uvicorn.run(app, host="0.0.0.0", port=8788)
