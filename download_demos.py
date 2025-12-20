"""
Download CS2/CS:GO Valve MM demos.

Two modes:
1) File mode:  python download_demos.py sharecodes.txt ./demos
2) AUTO mode:  python download_demos.py AUTO ./demos
   - Uses STEAM_WEB_API_KEY + TARGET_STEAMID64 + TARGET_AUTH_CODE + TARGET_KNOWN_CODE
   - Enumerates share codes via Steam Web API, then fetches demo URLs via Steam GC.

.env variables used:
- BOT_STEAM_USER / BOT_STEAM_PASS  (or STEAM_USER / STEAM_PASS fallback)
- STEAM_WEB_API_KEY               (AUTO mode)
- TARGET_STEAMID64                (AUTO mode)
- TARGET_AUTH_CODE                (AUTO mode)
- TARGET_KNOWN_CODE               (AUTO mode)
"""

from __future__ import annotations

import bz2
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ValvePython stack (the modules you actually have: pkgutil showed "csgo")
from steam.client import SteamClient
from steam.enums import EResult
from csgo.client import CSGOClient
from csgo.sharecode import decode as decode_sharecode


SHARECODE_RE = re.compile(r"CSGO-[A-Za-z0-9]{5}(?:-[A-Za-z0-9]{5}){4}")
STEAM_NEXT_CODE_URL = "https://api.steampowered.com/ICSGOPlayers_730/GetNextMatchSharingCode/v1"


def _env_first(*names: str) -> str | None:
    # Why: allow BOT_* names but keep backward compatibility with STEAM_* names.
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.strip()
    return None


def read_sharecodes(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    codes = SHARECODE_RE.findall(txt)
    # preserve order but de-dup
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def steam_get_next_share_code(api_key: str, steamid64: str, auth_code: str, known_code: str) -> str | None:
    params = {
        "key": api_key,
        "steamid": steamid64,
        "steamidkey": auth_code,
        "knowncode": known_code,
    }
    r = requests.get(STEAM_NEXT_CODE_URL, params=params, timeout=30)

    # Why: Steam often returns 412 when knowncode is invalid (like "n/a").
    # Also prevents secrets from being printed in a full URL traceback.
    if r.status_code == 412:
        # treat as "can't continue" rather than crash
        return None

    r.raise_for_status()

    data = r.json()
    result = data.get("result", {}) if isinstance(data, dict) else {}
    nxt = result.get("nextcode")

    if not isinstance(nxt, str):
        return None

    nxt = nxt.strip()
    if not nxt or nxt.lower() == "n/a":
        return None

    # Why: stop if Steam returns garbage; prevents cur becoming invalid and causing 412.
    if not SHARECODE_RE.fullmatch(nxt):
        return None

    return nxt

def enumerate_share_codes_from_env(max_codes: int = 5000, sleep_s: float = 0.25) -> list[str]:
    api_key = os.environ.get("STEAM_WEB_API_KEY", "").strip()
    steamid64 = os.environ.get("TARGET_STEAMID64", "").strip()
    auth_code = os.environ.get("TARGET_AUTH_CODE", "").strip()
    known_code = os.environ.get("TARGET_KNOWN_CODE", "").strip()

    missing = [k for k, v in {
        "STEAM_WEB_API_KEY": api_key,
        "TARGET_STEAMID64": steamid64,
        "TARGET_AUTH_CODE": auth_code,
        "TARGET_KNOWN_CODE": known_code,
    }.items() if not v]
    if missing:
        raise SystemExit("AUTO mode requires these .env vars:\n" + "\n".join(f"  - {m}" for m in missing))

    codes: list[str] = [known_code]
    seen: set[str] = {known_code}
    cur = known_code

    for _ in tqdm(range(max_codes), desc="Fetching share codes"):
        time.sleep(sleep_s)  # Why: avoid rate limits
        nxt = steam_get_next_share_code(api_key, steamid64, auth_code, cur)
        if not nxt or nxt in seen:
            break
        codes.append(nxt)
        seen.add(nxt)
        cur = nxt

    return codes


def download_stream(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0") or "0")
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name, leave=False) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))


def decompress_bz2(bz2_path: Path, dem_path: Path) -> None:
    with bz2.open(bz2_path, "rb") as src, open(dem_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 256)
            if not chunk:
                break
            dst.write(chunk)


def _walk_values(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield v
            yield from _walk_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield v
            yield from _walk_values(v)


def extract_demo_url_loose(matchinfo: Any) -> str | None:
    # Why: proto fields vary; find the first *.dem.bz2 URL anywhere.
    url_re = re.compile(r"(https?://[^\s\"']+\.dem\.bz2)", re.IGNORECASE)
    host_re = re.compile(r"(replay\d+\.valve\.net/730/[^\s\"']+\.dem\.bz2)", re.IGNORECASE)

    if isinstance(matchinfo, str):
        m = url_re.search(matchinfo) or host_re.search(matchinfo)
        if not m:
            return None
        u = m.group(1)
        return u if u.startswith("http") else "http://" + u

    # If it’s a proto message, str() often includes URLs; scan that too.
    try:
        s = str(matchinfo)
        m = url_re.search(s) or host_re.search(s)
        if m:
            u = m.group(1)
            return u if u.startswith("http") else "http://" + u
    except Exception:
        pass

    for v in _walk_values(matchinfo):
        if isinstance(v, str):
            m = url_re.search(v) or host_re.search(v)
            if m:
                u = m.group(1)
                return u if u.startswith("http") else "http://" + u

    return None

import logging

def gc_login_and_ready(username: str, password: str, timeout_s: int = 90) -> CSGOClient:
    # Why: SteamClient has internal logs that explain connect failures (timeouts, SSL, blocked ports, etc.)
    logging.basicConfig(level=logging.INFO)

    steam = SteamClient()
    cs = CSGOClient(steam)

    # Extra verbose from the steam library
    logging.getLogger("steam").setLevel(logging.DEBUG)

    # Try connecting a few times
    for attempt in range(1, 6):
        print(f"[steam] connect attempt {attempt} ...")
        steam.connect()

        # wait for either connected or disconnected/exception events
        ev_connected = steam.wait_event("connected", timeout=10)
        if ev_connected:
            print("[steam] connected")
            break

        # if it didn't connect, dump state we can see
        print(f"[steam] not connected yet (attempt {attempt}); retrying ...")
        try:
            steam.disconnect()
        except Exception:
            pass
        time.sleep(min(2 ** attempt, 10))
    else:
        raise RuntimeError(
            "Steam: failed to connect to CM servers after retries.\n"
            "At this point it's almost always network blocking:\n"
            " - firewall blocks outbound TCP\n"
            " - restrictive network (campus/work)\n"
            " - VPN/proxy interference\n"
            "Try a different network (phone hotspot) to confirm."
        )

    # login and wait for logged_on
    print("[steam] logging in ...")
    steam.login(username, password)

    if not steam.wait_event("logged_on", timeout=timeout_s):
        raise RuntimeError("Steam: login timed out. Steam Guard may be required or network is blocking.")

    print("[steam] logged on, launching GC ...")
    cs.launch()

    if not cs.wait_event("ready", timeout=timeout_s):
        raise RuntimeError("CS2 GC: not ready. Launch CS2 once in Steam GUI, then retry.")

    print("[steam] GC ready")
    return cs

def request_full_match_info(cs: CSGOClient, matchid: int, outcomeid: int, token: int, timeout_s: int = 30) -> Any:
    """
    Why: Match feature exposes request_full_match_info; response event is 'full_match_info'.
    We try to wait by job id if returned; otherwise wait for the event.
    """
    jobid = None
    try:
        jobid = cs.match.request_full_match_info(matchid, outcomeid, token)
    except Exception:
        # fallback: older versions might expose method elsewhere; re-raise with hint
        raise

    # Prefer waiting by jobid if it's truthy
    if jobid:
        try:
            msg = cs.wait_msg(jobid, timeout=timeout_s, raises=False)
            if msg is not None:
                return msg
        except Exception:
            pass

    # Fallback: wait for event
    msg = cs.wait_event("full_match_info", timeout=timeout_s)
    if not msg:
        raise TimeoutError("Timed out waiting for full_match_info")
    return msg


def main() -> int:
    load_dotenv()

    print("DEBUG TARGET_KNOWN_CODE =", repr(os.environ.get("TARGET_KNOWN_CODE")))

    if len(sys.argv) != 3:
        print("Usage: python download_demos.py <sharecodes.txt|AUTO> <out_dir>")
        return 2

    sharecodes_arg = sys.argv[1]
    out_dir = Path(sys.argv[2]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    steam_user = _env_first("BOT_STEAM_USER", "STEAM_USER")
    steam_pass = _env_first("BOT_STEAM_PASS", "STEAM_PASS")
    if not steam_user or not steam_pass:
        print("Set BOT_STEAM_USER and BOT_STEAM_PASS (or STEAM_USER/STEAM_PASS) in .env.")
        return 2

    if sharecodes_arg.strip().upper() == "AUTO":
        sharecodes = enumerate_share_codes_from_env()
    else:
        sharecodes_file = Path(sharecodes_arg).expanduser().resolve()
        if not sharecodes_file.exists():
            print(f"Sharecodes file not found: {sharecodes_file}")
            return 2
        sharecodes = read_sharecodes(sharecodes_file)

    if not sharecodes:
        print("No share codes found.")
        return 2

    # GC login
    cs = gc_login_and_ready(steam_user, steam_pass)

    ok = 0
    fail = 0
    try:
        for code in tqdm(sharecodes, desc="Matches"):
            try:
                parts = decode_sharecode(code)  # {'matchid':..., 'outcomeid':..., 'token':...}
                matchid = int(parts["matchid"])
                outcomeid = int(parts["outcomeid"])
                token = int(parts["token"])

                dem_path = out_dir / f"{matchid}.dem"
                if dem_path.exists():
                    continue

                matchinfo = request_full_match_info(cs, matchid, outcomeid, token, timeout_s=45)
                demo_url = extract_demo_url_loose(matchinfo)
                if not demo_url:
                    raise RuntimeError("Could not find demo URL in match info payload.")

                bz2_path = out_dir / f"{matchid}.dem.bz2"
                download_stream(demo_url, bz2_path)
                decompress_bz2(bz2_path, dem_path)
                bz2_path.unlink(missing_ok=True)

                ok += 1
            except Exception as e:
                fail += 1
                print(f"[WARN] {code}: {e}")
    finally:
        try:
            cs.exit()
        except Exception:
            pass
        try:
            cs.steam.disconnect()
        except Exception:
            pass

    print(f"Done. Saved demos to: {out_dir}")
    print(f"Downloaded: {ok}, Failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
