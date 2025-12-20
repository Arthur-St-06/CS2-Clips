#!/usr/bin/env python3
# steam_login.py

# WHY: Must be first. If urllib3/ssl imports happen before this, you can get recursion/hangs.
from gevent import monkey
monkey.patch_all()

import argparse
import bz2
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from google.protobuf.json_format import MessageToDict

from steam.client import SteamClient
from steam.enums.common import EResult

from csgo.client import CSGOClient
from csgo import sharecode
from csgo.enums import EGCBaseClientMsg, ECsgoGCMsg


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Keep urllib3 quieter unless you really want it.
    if not debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def find_demo_url(obj) -> Optional[str]:
    # WHY: protobuf/dict shape changes; robust recursive search is safer than hardcoded keys.
    exts = (".dem", ".dem.bz2", ".bz2")
    if isinstance(obj, dict):
        for _, v in obj.items():
            u = find_demo_url(v)
            if u:
                return u
    elif isinstance(obj, list):
        for v in obj:
            u = find_demo_url(v)
            if u:
                return u
    elif isinstance(obj, str):
        if obj.startswith("http") and any(ext in obj.lower() for ext in exts):
            return obj
    return None


def download(url: str, out_path: Path, timeout_s: int = 60) -> None:
    # WHY: import requests here so urllib3 loads after gevent patching.
    import requests

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("download")
    log.info("GET %s", url)

    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    log.info("progress %.1f%% (%d/%d)", done * 100.0 / total, done, total)
                else:
                    log.info("progress %d bytes", done)

    log.info("saved %s", out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("share_code")
    ap.add_argument("out_dir")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--gc-version", type=int, required=True)
    ap.add_argument("--gc-timeout", type=int, default=90)
    ap.add_argument("--match-timeout", type=int, default=60)
    ap.add_argument("--download-timeout", type=int, default=60)
    args = ap.parse_args()

    setup_logging(args.debug)
    log = logging.getLogger("main")

    # Decode sharecode (fixes your "matchid=matchid" placeholder issue)
    ids = sharecode.decode(args.share_code)
    matchid = int(ids["matchid"])
    outcomeid = int(ids["outcomeid"])
    token = int(ids["token"])

    log.info("[SHARECODE] matchid=%d outcomeid=%d token=%d", matchid, outcomeid, token)
    log.info("[GC] Using GC ClientHello version=%d", args.gc_version)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    client = SteamClient()
    cs = CSGOClient(client)

    gc_welcome = False
    got_fatal = False
    fatal_dump = None

    @cs.on(ECsgoGCMsg.EMsgGCCStrike15_v2_ClientLogonFatalError)
    def _fatal(msg):
        nonlocal got_fatal, fatal_dump
        got_fatal = True
        try:
            fatal_dump = MessageToDict(msg, preserving_proto_field_name=True)
        except Exception:
            fatal_dump = {"_raw": str(msg)}
        log.error("[GC] ClientLogonFatalError (9187). Payload=%s", json.dumps(fatal_dump)[:2000])

    @cs.on(EGCBaseClientMsg.EMsgGCClientWelcome)
    def _welcome(_msg):
        nonlocal gc_welcome
        gc_welcome = True
        log.info("[GC] ClientWelcome received (GC session established)")

    log.info("[STAGE] Logging into Steam (terminal prompts)")
    res = client.cli_login()
    if res != EResult.OK:
        log.error("Steam login failed: %s", res)
        return 2

    log.info("[STAGE] Logged on OK. Marking as playing app 730 and starting GC handshake.")
    client.games_played([730])

    # Send hello repeatedly until welcome or fatal/timeout
    start = time.time()
    while True:
        if got_fatal:
            log.error("GC rejected login. Usually means library mismatch OR hello fields changed.")
            return 3
        if gc_welcome:
            break
        if time.time() - start > args.gc_timeout:
            log.error("Timed out waiting for GC welcome after %ds.", args.gc_timeout)
            return 4

        # WHY: Explicit version often required after CS2 updates.
        cs.send(EGCBaseClientMsg.EMsgGCClientHello, {"version": int(args.gc_version)})
        log.debug("[GC] Sent ClientHello(version=%d); waiting...", args.gc_version)
        client.sleep(2.0)

    # Request match info (job-based response)
    log.info("[GC] Requesting full match info...")
    cs.request_full_match_info(matchid, outcomeid, token)

    # Wait for the event
    ev = cs.wait_event("full_match_info", timeout=args.match_timeout)
    if not ev:
        log.error("Timed out waiting for full_match_info after %ds.", args.match_timeout)
        return 5

    (msg,) = ev
    d = MessageToDict(msg, preserving_proto_field_name=True)
    info_path = out_dir / f"{matchid}_full_match_info.json"
    info_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    log.info("[OUT] wrote %s", info_path)

    url = find_demo_url(d)
    if not url:
        # Fallback: find any URL at all
        blob = json.dumps(d)
        m = re.search(r"https?://[^\"\\s]+", blob)
        url = m.group(0) if m else None

    if not url:
        log.error("Could not find demo URL in match info. Inspect %s", info_path)
        return 6

    log.info("[DEMO] URL: %s", url)

    bz2_path = out_dir / f"{matchid}.dem.bz2"
    dem_path = out_dir / f"{matchid}.dem"

    download(url, bz2_path, timeout_s=args.download_timeout)

    # Try bz2 decompress
    try:
        dem_path.write_bytes(bz2.decompress(bz2_path.read_bytes()))
        log.info("[DONE] decompressed demo: %s", dem_path)
    except OSError:
        log.info("[DONE] download not bz2 (kept): %s", bz2_path)

    try:
        client.disconnect()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
