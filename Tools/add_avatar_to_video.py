"""Put a demo's avatar video into the bottom-right box of its OBS recording.

    python Tools/add_avatar_to_video.py "<OBS video>.mp4" "<session folder>"

The session folder holds face_tracking.csv and avatar.mp4 (VR Emotion > Avatar Replay in
Unity). The result is written next to the OBS video as "<name>_with_avatar.mp4"; the OBS video
itself is not changed.

Syncing. The avatar's time 0 is the first get-ready screen. Where that falls in the OBS video
is found in two steps:
  1. by the clock: the OBS file name holds its local start time (to the second) and
     face_tracking.csv holds the UTC time of time 0;
  2. by the picture: within a few seconds of that estimate, the frame where the game half of
     the video changes most is the screen switching to the first get-ready screen.
Use --offset to give the time by hand instead.

Layout (the OBS canvas set up for the demo): 1920 x 720, game 1280 x 720 at the left, webcam
640 x 360 at the top right, avatar 640 x 360 at the bottom right.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

GAME_CROP = "1280:720:0:0"
AVATAR_POSITION = (1280, 360)
AVATAR_SIZE = (640, 360)  # the box; avatar.mp4 is rendered larger (1280 x 720) and scaled down
SEARCH_SECONDS = 4.0  # either side of the clock estimate
FPS = 30


def ffmpeg_tool(name):
    path = shutil.which(name)
    # A terminal opened before `winget install ffmpeg` does not have it on its PATH yet; winget
    # puts its link here.
    if path is None:
        winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / f"{name}.exe"
        path = str(winget) if winget.exists() else None
    if path is None:
        sys.exit(f"{name} not found. Install it with: winget install ffmpeg")
    return path


def clock_estimate(video, face_csv):
    match = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{2})-(\d{2})-(\d{2})", video.name)
    if match is None:
        return None
    started_local = datetime.strptime(" ".join(match.groups()), "%Y-%m-%d %H %M %S")
    started_utc = started_local.astimezone().astimezone(timezone.utc)
    with open(face_csv, encoding="utf-8") as f:
        first = next(csv.DictReader(f))
    time0 = datetime.strptime(first["timestamp_utc"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return (time0 - started_utc).total_seconds()


def screen_change(video, around):
    """The time of the largest frame-to-frame change in the game half near `around`."""
    start = max(0.0, around - SEARCH_SECONDS)
    width, height = 160, 90
    raw = subprocess.run(
        [ffmpeg_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}",
         "-t", f"{2 * SEARCH_SECONDS:.3f}", "-i", str(video),
         "-vf", f"crop={GAME_CROP},scale={width}:{height},fps={FPS},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    size = width * height
    frames = [raw[i:i + size] for i in range(0, len(raw) - size + 1, size)]
    diffs = [(sum(abs(a - b) for a, b in zip(frames[k], frames[k - 1])) / size, k)
             for k in range(1, len(frames))]
    if not diffs:
        return None, 0.0
    best, k = max(diffs)
    rest = sorted(d for d, _ in diffs)[-2] if len(diffs) > 1 else 0.0
    return start + k / FPS, best / max(rest, 1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="the OBS recording")
    parser.add_argument("session", type=Path, help="the session folder with face_tracking.csv and avatar.mp4")
    parser.add_argument("--offset", type=float, help="seconds into the OBS video where the avatar starts")
    args = parser.parse_args()

    face_csv = args.session / "face_tracking.csv"
    avatar = args.session / "avatar.mp4"
    for path in (args.video, face_csv, avatar):
        if not path.exists():
            sys.exit(f"Not found: {path}")

    offset = args.offset
    if offset is None:
        estimate = clock_estimate(args.video, face_csv)
        if estimate is None:
            sys.exit("The OBS file name has no start time; give --offset by hand.")
        found, contrast = screen_change(args.video, estimate)
        print(f"clock estimate: {estimate:.2f} s")
        if found is not None and contrast >= 2.0 and abs(found - estimate) <= SEARCH_SECONDS:
            offset = found
            print(f"screen change:  {found:.3f} s (x{contrast:.1f} the next largest change) -> used")
        else:
            offset = estimate
            print("no clear screen change found -> using the clock estimate; check it, or give --offset")

    out = args.video.with_name(args.video.stem + "_with_avatar.mp4")
    x, y = AVATAR_POSITION
    w, h = AVATAR_SIZE
    encoder = ["-c:v", "h264_nvenc", "-cq", "19", "-preset", "p5"]
    command = [ffmpeg_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-stats", "-y",
               "-i", str(args.video), "-i", str(avatar),
               "-filter_complex",
               f"[1:v]scale={w}:{h}:flags=lanczos,setpts=PTS-STARTPTS+{offset:.3f}/TB[av];"
               f"[0:v][av]overlay={x}:{y}:eof_action=pass[v]",
               "-map", "[v]", "-map", "0:a?", *encoder, "-c:a", "copy", str(out)]
    print(f"avatar starts at {offset:.3f} s; writing {out}")
    if subprocess.run(command).returncode != 0:
        print("NVENC failed; encoding on the CPU instead")
        command[command.index("h264_nvenc") - 1:command.index("p5") + 1] = ["-c:v", "libx264", "-crf", "18"]
        subprocess.run(command, check=True)
    print("done")


if __name__ == "__main__":
    main()
