#!/usr/bin/env python3
# steam_login.py

# WHY: Must be first. If urllib3/ssl imports happen before this, you can get recursion/hangs.
from gevent import monkey
monkey.patch_all()

import argparse
import bz2
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Set, Tuple

from dotenv import load_dotenv
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
        low = obj.lower()
        if obj.startswith("http") and any(ext in low for ext in exts):
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


def decode_share_code(code: str) -> Tuple[int, int, int]:
    # WHY: sharecode.decode returns real ids; avoids placeholder printing.
    ids = sharecode.decode(code)
    return int(ids["matchid"]), int(ids["outcomeid"]), int(ids["token"])


def webapi_get_next_sharecode(
    web_api_key: str,
    steamid64: str,
    steamidkey: str,
    knowncode: str,
    timeout_s: int = 20,
) -> Optional[str]:
    """
    Calls Valve WebAPI to get the next sharecode after 'knowncode'.
    Requires Match History Authentication Code (steamidkey).
    """
    import requests

    url = "https://api.steampowered.com/ICSGOPlayers_730/GetNextMatchSharingCode/v1/"
    params = {
        "key": web_api_key,
        "steamid": steamid64,
        "steamidkey": steamidkey,
        "knowncode": knowncode,
    }

    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    result = data.get("result") if isinstance(data, dict) else None
    nextcode = result.get("nextcode") if isinstance(result, dict) else None

    if isinstance(nextcode, str) and nextcode.startswith("CSGO-"):
        return nextcode
    return None


def download_one_demo_via_gc(
    cs: CSGOClient,
    matchid: int,
    outcomeid: int,
    token: int,
    out_dir: Path,
    match_timeout: int,
    download_timeout: int,
    log: logging.Logger,
) -> bool:
    dem_path = out_dir / f"{matchid}.dem"
    bz2_path = out_dir / f"{matchid}.dem.bz2"

    if dem_path.exists():
        log.info("[SKIP] already have %s", dem_path)
        return False

    log.info("[GC] Requesting full match info matchid=%d ...", matchid)
    cs.request_full_match_info(matchid, outcomeid, token)

    ev = cs.wait_event("full_match_info", timeout=match_timeout)
    if not ev:
        log.error("[FAIL] timed out waiting for full_match_info after %ds (matchid=%d)", match_timeout, matchid)
        return False

    (msg,) = ev
    d = MessageToDict(msg, preserving_proto_field_name=True)

    info_path = out_dir / f"{matchid}_full_match_info.json"
    info_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    log.info("[OUT] wrote %s", info_path)

    url = find_demo_url(d)
    if not url:
        blob = json.dumps(d)
        m = re.search(r"https?://[^\"\\s]+", blob)
        url = m.group(0) if m else None

    if not url:
        log.error("[FAIL] could not find demo URL in %s", info_path)
        return False

    log.info("[DEMO] URL: %s", url)
    download(url, bz2_path, timeout_s=download_timeout)

    # Try bz2 decompress
    try:
        dem_path.write_bytes(bz2.decompress(bz2_path.read_bytes()))
        log.info("[DONE] decompressed demo: %s", dem_path)
    except OSError:
        log.info("[DONE] download not bz2 (kept): %s", bz2_path)

    return True


def steam_login_from_env_or_prompt(client: SteamClient, log: logging.Logger) -> EResult:
    """
    Uses BOT_STEAM_USER/BOT_STEAM_PASS if present; otherwise falls back to interactive cli_login().

    WHY: You asked to use .env keys. Still supports Steam Guard prompts safely in terminal.
    """
    user = (os.environ.get("BOT_STEAM_USER") or "").strip()
    pw = (os.environ.get("BOT_STEAM_PASS") or "").strip()

    if user and pw:
        log.info("[STAGE] Logging into Steam using BOT_STEAM_USER/BOT_STEAM_PASS from .env")
        # NOTE: Steam Guard (2FA/email) may still be required; ValvePython will emit events.
        return client.login(user, pw)

    log.info("[STAGE] BOT_STEAM_USER/BOT_STEAM_PASS missing; using interactive login prompts")
    return client.cli_login()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="./demos", help="Output folder (default: ./demos)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--gc-version", type=int, required=True)

    ap.add_argument("--gc-timeout", type=int, default=90)
    ap.add_argument("--match-timeout", type=int, default=60)
    ap.add_argument("--download-timeout", type=int, default=60)

    ap.add_argument("--max-matches", type=int, default=200, help="Max demos to attempt")
    ap.add_argument("--webapi-delay", type=float, default=1.0, help="Seconds between WebAPI calls")
    ap.add_argument("--start-code", default=None, help="Override starting sharecode (else TARGET_KNOWN_CODE)")
    ap.add_argument("--start-only", action="store_true", help="Only download the starting match (no walk-forward)")

    args = ap.parse_args()

    # WHY: load .env here so all getenv() reads see your keys.
    load_dotenv()

    setup_logging(args.debug)
    log = logging.getLogger("main")

    # Read required env keys for walk-forward
    steam_web_api_key = (os.environ.get("STEAM_WEB_API_KEY") or "").strip()
    target_steamid64 = (os.environ.get("TARGET_STEAMID64") or "").strip()
    target_auth_code = (os.environ.get("TARGET_AUTH_CODE") or "").strip()  # match history auth code
    target_known_code = (os.environ.get("TARGET_KNOWN_CODE") or "").strip()

    start_code = (args.start_code or target_known_code).strip()
    if not start_code:
        log.error("No starting sharecode. Provide --start-code or set TARGET_KNOWN_CODE in .env")
        return 2

    log.info("[GC] Using GC ClientHello version=%d", args.gc_version)
    log.info("[START] sharecode=%s", start_code)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    client = SteamClient()
    cs = CSGOClient(client)

    gc_welcome = False
    got_fatal = False

    # Steam Guard helpers (only needed if Steam requires extra step)
    auth_needed = {"required": False}

    @client.on("auth_code_required")
    def _auth_code_required(_):
        auth_needed["required"] = True
        log.warning("Steam Guard code required (you will be prompted in terminal).")

    @cs.on(ECsgoGCMsg.EMsgGCCStrike15_v2_ClientLogonFatalError)
    def _fatal(msg):
        nonlocal got_fatal
        got_fatal = True
        try:
            payload = MessageToDict(msg, preserving_proto_field_name=True)
        except Exception:
            payload = {"_raw": str(msg)}
        log.error("[GC] ClientLogonFatalError (9187). Payload=%s", json.dumps(payload)[:2000])

    @cs.on(EGCBaseClientMsg.EMsgGCClientWelcome)
    def _welcome(_msg):
        nonlocal gc_welcome
        gc_welcome = True
        log.info("[GC] ClientWelcome received (GC session established)")

    log.info("[STAGE] Logging into Steam")
    res = steam_login_from_env_or_prompt(client, log)

    # If login() returned auth required, ValvePython may want you to call cli_login.
    # Best universal fallback: if login didn't succeed, use cli_login.
    if res != EResult.OK:
        log.warning("Non-OK login result (%s). Falling back to interactive cli_login().", res)
        res = client.cli_login()

    if res != EResult.OK:
        log.error("Steam login failed: %s", res)
        return 3

    log.info("[STAGE] Logged on OK. Marking as playing app 730 and starting GC handshake.")
    client.games_played([730])

    # Send hello until welcome or fatal/timeout
    start_t = time.time()
    while True:
        if got_fatal:
            log.error("GC rejected login. Common causes: wrong GC version or library mismatch.")
            return 4
        if gc_welcome:
            break
        if time.time() - start_t > args.gc_timeout:
            log.error("Timed out waiting for GC welcome after %ds.", args.gc_timeout)
            return 5

        cs.send(EGCBaseClientMsg.EMsgGCClientHello, {"version": int(args.gc_version)})
        log.debug("[GC] Sent ClientHello(version=%d); waiting...", args.gc_version)
        client.sleep(2.0)

    # Build sharecode list by walking forward (unless start-only)
    share_codes = [start_code]
    if args.start_only:
        log.info("[MODE] start-only enabled (no walk-forward)")
    else:
        # Validate env for walk-forward
        missing = []
        if not steam_web_api_key:
            missing.append("STEAM_WEB_API_KEY")
        if not target_steamid64:
            missing.append("TARGET_STEAMID64")
        if not target_auth_code:
            missing.append("TARGET_AUTH_CODE")

        if missing:
            log.warning(
                "Walk-forward disabled because missing in .env: %s. "
                "Set them to enable downloading multiple matches.",
                ", ".join(missing),
            )
        else:
            log.info("[WEBAPI] Walking forward from knowncode to collect matches (max=%d)", args.max_matches)
            seen: Set[str] = {start_code}
            current = start_code
            while len(share_codes) < args.max_matches:
                try:
                    nxt = webapi_get_next_sharecode(
                        web_api_key=steam_web_api_key,
                        steamid64=target_steamid64,
                        steamidkey=target_auth_code,
                        knowncode=current,
                    )
                except Exception as e:
                    log.error("[WEBAPI] failed: %s", e)
                    break

                if not nxt:
                    log.info("[WEBAPI] no nextcode returned (end reached or knowncode too old)")
                    break
                if nxt in seen:
                    log.info("[WEBAPI] nextcode repeated; stopping (%s)", nxt)
                    break

                share_codes.append(nxt)
                seen.add(nxt)
                current = nxt
                log.info("[WEBAPI] + %s (total=%d)", nxt, len(share_codes))
                time.sleep(max(0.0, args.webapi_delay))

    # Download each demo via GC
    downloaded = 0
    for i, code in enumerate(share_codes, 1):
        try:
            matchid, outcomeid, token = decode_share_code(code)
        except Exception as e:
            log.error("[FAIL] cannot decode share code %s: %s", code, e)
            continue

        log.info("[JOB] (%d/%d) code=%s matchid=%d", i, len(share_codes), code, matchid)
        ok = download_one_demo_via_gc(
            cs=cs,
            matchid=matchid,
            outcomeid=outcomeid,
            token=token,
            out_dir=out_dir,
            match_timeout=args.match_timeout,
            download_timeout=args.download_timeout,
            log=log,
        )
        if ok:
            downloaded += 1

    log.info("[SUMMARY] attempted=%d downloaded_or_decompressed=%d out_dir=%s", len(share_codes), downloaded, out_dir)

    try:
        client.disconnect()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
