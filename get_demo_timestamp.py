#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
import subprocess, re, sys

def main(dem_path: str):
    dem = Path(dem_path).resolve()
    info = Path(str(dem) + ".info")
    if not info.exists():
        info = dem.with_suffix(dem.suffix + ".info")
    if not info.exists():
        raise FileNotFoundError(f"No .dem.info found for {dem.name}")

    p = subprocess.run(
        ["protoc", "--decode_raw"],
        input=info.read_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    m = re.search(r"(?m)^2:\s*(\d+)\s*$", p.stdout.decode("utf-8", errors="replace"))
    if not m:
        raise RuntimeError("Could not find top-level '2: <epoch>' in decode_raw output")

    epoch = int(m.group(1))
    dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
    dt_local = dt_utc.astimezone()

    print("epoch:", epoch)
    print("UTC  :", dt_utc.isoformat())
    print("Local:", dt_local.isoformat())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 demo_time_via_protoc.py <file.dem>")
        sys.exit(1)
    main(sys.argv[1])
