#!/usr/bin/env python3
"""
test_totp.py - Test Steam TOTP generation

Usage:
    python test_totp.py <shared_secret>
    
Example:
    python test_totp.py "Daa211QWEld2gPEfUt5uN8WJ/Ug="
    
This will generate 2FA codes and show a countdown. Compare with your
Steam Mobile app or Steam Desktop Authenticator to verify it works.
"""

import base64
import hashlib
import hmac
import struct
import sys
import time

STEAM_ALPHABET = "23456789BCDFGHJKMNPQRTVWXY"


def generate_steam_totp(shared_secret: str) -> str:
    """Generate a Steam-style 2FA code from the shared_secret."""
    key = base64.b64decode(shared_secret)
    timestamp = int(time.time()) // 30
    msg = struct.pack(">Q", timestamp)
    auth = hmac.new(key, msg, hashlib.sha1)
    digest = auth.digest()
    start = digest[19] & 0x0F
    code_int = struct.unpack(">I", digest[start:start + 4])[0] & 0x7FFFFFFF
    
    code_chars = []
    for _ in range(5):
        code_chars.append(STEAM_ALPHABET[code_int % len(STEAM_ALPHABET)])
        code_int //= len(STEAM_ALPHABET)
    
    return "".join(code_chars)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_totp.py <shared_secret>")
        print()
        print("Example:")
        print('  python test_totp.py "Daa211QWEld2gPEfUt5uN8WJ/Ug="')
        sys.exit(1)
    
    shared_secret = sys.argv[1].strip()
    
    print("=" * 50)
    print("Steam TOTP Generator Test")
    print("=" * 50)
    print()
    print("Compare these codes with your Steam Mobile app")
    print("or Steam Desktop Authenticator.")
    print()
    print("Press Ctrl+C to stop")
    print()
    print("-" * 50)
    
    last_code = ""
    
    try:
        while True:
            code = generate_steam_totp(shared_secret)
            remaining = 30 - (int(time.time()) % 30)
            
            if code != last_code:
                print()
                last_code = code
            
            # Clear line and print
            bar = "█" * remaining + "░" * (30 - remaining)
            print(f"\r  Code: {code}  |  {bar}  {remaining:2d}s  ", end="", flush=True)
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nDone!")


if __name__ == "__main__":
    main()
