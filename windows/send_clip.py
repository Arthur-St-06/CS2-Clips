
# send_clip.py
import os
import sys
from pathlib import Path
import requests

def main():
    if len(sys.argv) < 3:
        print("Usage: python send_clip.py <ubuntu_upload_url> <video_path>")
        print('Example: python send_clip.py http://YOUR_UBUNTU_IP:8787/upload D:\\hlae_out\\take0012\\out.mp4')
        sys.exit(2)

    url = sys.argv[1]
    video_path = Path(sys.argv[2]).resolve()
    token = os.environ.get("CLIP_TOKEN", "")
    if not token:
        print("Set CLIP_TOKEN env var first.")
        sys.exit(2)

    if not video_path.exists():
        print(f"File not found: {video_path}")
        sys.exit(2)

    headers = {
        "Authorization": f"Bearer {token}",
        "X-Clip-Name": video_path.name,  # “by name”
    }

    with video_path.open("rb") as f:
        files = {"file": (video_path.name, f, "video/mp4")}
        r = requests.post(url, headers=headers, files=files, timeout=300)

    print("Status:", r.status_code)
    print(r.text)
    r.raise_for_status()

if __name__ == "__main__":
    main()
