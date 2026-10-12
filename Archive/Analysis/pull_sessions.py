#!/usr/bin/env python3
"""Copy new session recordings off a connected Quest.

    python pull_sessions.py              # pull whatever is new
    python pull_sessions.py --list       # show what is on the headset, copy nothing
    python pull_sessions.py --latest     # pull just the most recent recording
    python pull_sessions.py --analyze    # pull, then report on what came across

Sessions already sitting in the destination folder are skipped, so this is safe to run
after every playtest: it only fetches recordings you do not already have, and it never
deletes anything from the headset.

Nothing needs configuring. adb is located automatically (PATH, the Android SDK, or the
copy Unity ships with its Android build support), and the app's package name is read out
of ProjectSettings.asset, so this keeps working if either changes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import session_analysis as sa

# Where the game writes on the headset. Unity's Application.persistentDataPath on Android
# is this folder, and adb can read it even though apps cannot read each other's.
REMOTE_TEMPLATE = "/sdcard/Android/data/{package}/files/sessions"

# Kept next to the other pulled recordings, under the gitignored Logs/ folder so a
# playtest never lands in a commit.
DEFAULT_DEST = Path("Logs") / "quest-sessions" / "sessions"


class AdbError(RuntimeError):
    """An adb invocation failed, or adb could not be found at all."""


# ------------------------------------------------------------------------------ adb


def find_adb(explicit: str | None = None) -> Path:
    """Locate an adb binary without making the user set one up."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
        raise AdbError(f"--adb {explicit} is not a file")

    on_path = shutil.which("adb")
    if on_path:
        return Path(on_path)

    exe = "adb.exe" if sys.platform == "win32" else "adb"
    candidates: list[Path] = []

    for variable in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root) / "platform-tools" / exe)

    home = Path.home()
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        candidates.append(local / "Android" / "Sdk" / "platform-tools" / exe)
        unity_roots = [Path(r"C:\Program Files\Unity\Hub\Editor")]
        # Meta's tooling ships its own adb, and is often the only one a headset-facing
        # machine has.
        candidates += [
            local / "Programs" / name / "resources" / "bin" / exe
            for name in ("meta-quest-developer-hub", "Oculus Developer Hub")
        ]
    elif sys.platform == "darwin":
        candidates.append(home / "Library" / "Android" / "sdk" / "platform-tools" / exe)
        unity_roots = [Path("/Applications/Unity/Hub/Editor")]
    else:
        candidates.append(home / "Android" / "Sdk" / "platform-tools" / exe)
        unity_roots = [home / "Unity" / "Hub" / "Editor"]

    # Unity's Android build support bundles a platform-tools; newest editor first.
    for root in unity_roots:
        if root.is_dir():
            for editor in sorted(root.iterdir(), reverse=True):
                candidates.append(
                    editor / "Editor" / "Data" / "PlaybackEngines" / "AndroidPlayer"
                    / "SDK" / "platform-tools" / exe
                )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise AdbError(
        "could not find adb. Install Android platform-tools or Unity's Android Build "
        "Support, or pass --adb C:\\path\\to\\adb.exe"
    )


def run_adb(adb: Path, args: list[str], serial: str | None = None) -> subprocess.CompletedProcess:
    command = [str(adb)]
    if serial:
        command += ["-s", serial]
    command += args
    return subprocess.run(command, capture_output=True, text=True, errors="replace")


def list_devices(adb: Path) -> list[tuple[str, str, str]]:
    """(serial, state, description) for everything adb can see."""
    result = run_adb(adb, ["devices", "-l"])
    if result.returncode != 0:
        raise AdbError(f"adb devices failed: {result.stderr.strip()}")

    devices = []
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        serial, state = parts[0], parts[1] if len(parts) > 1 else "unknown"
        model = next((p.split(":", 1)[1] for p in parts[2:] if p.startswith("model:")), "")
        devices.append((serial, state, model.replace("_", " ")))
    return devices


def pick_device(adb: Path, wanted: str | None) -> str:
    """Choose which headset to talk to, and explain what to do when there isn't one."""
    devices = list_devices(adb)

    if wanted:
        for serial, state, _ in devices:
            if serial == wanted:
                if state != "device":
                    raise AdbError(f"device {serial} is '{state}', not ready")
                return serial
        raise AdbError(f"device {wanted} is not connected")

    ready = [d for d in devices if d[1] == "device"]
    if len(ready) == 1:
        serial, _, model = ready[0]
        print(f"headset: {model or serial}")
        return serial

    if len(ready) > 1:
        listing = "\n".join(f"  {s}  {m}" for s, _, m in ready)
        raise AdbError(f"several devices are connected; pick one with --device:\n{listing}")

    unauthorized = [d for d in devices if d[1] == "unauthorized"]
    if unauthorized:
        raise AdbError(
            "the headset is connected but not authorised. Put it on: there is an "
            "'Allow USB debugging' prompt waiting inside, and tick 'Always allow'."
        )
    if devices:
        states = ", ".join(f"{s} ({state})" for s, state, _ in devices)
        raise AdbError(f"no headset is ready: {states}")

    raise AdbError(
        "no headset detected. Check that it is plugged in with a data cable (not a "
        "charge-only one), powered on, and that Developer Mode is enabled for it in "
        "the Meta Quest phone app."
    )


# -------------------------------------------------------------------------- sessions


def remote_sessions(adb: Path, serial: str, remote: str) -> list[str]:
    """Names of the session folders on the headset."""
    result = run_adb(adb, ["shell", "ls", "-1", remote], serial=serial)
    output = (result.stdout + result.stderr).strip()

    if "No such file or directory" in output:
        return []
    if result.returncode != 0:
        raise AdbError(f"could not list {remote}: {output}")

    return sorted(
        line.strip() for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("ls:")
    )


def app_is_running(adb: Path, serial: str, package: str) -> bool:
    """True if the game is still open, which means its newest session is mid-write."""
    result = run_adb(adb, ["shell", "pidof", package], serial=serial)
    return bool(result.stdout.strip())


def installed_packages(adb: Path, serial: str, needle: str) -> list[str]:
    """Used only to suggest a package name when the configured one is not on the device."""
    result = run_adb(adb, ["shell", "pm", "list", "packages"], serial=serial)
    names = [line.replace("package:", "").strip() for line in result.stdout.splitlines()]
    return [n for n in names if needle.lower() in n.lower()]


def pull(adb: Path, serial: str, remote: str, name: str, dest: Path) -> bool:
    target = dest / name
    result = run_adb(adb, ["pull", f"{remote}/{name}", str(target)], serial=serial)
    if result.returncode != 0:
        print(f"  {name}: FAILED  {(result.stderr or result.stdout).strip()}", file=sys.stderr)
        return False
    if not (target / "predictions.csv").is_file():
        print(f"  {name}: pulled, but has no predictions.csv", file=sys.stderr)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy new session recordings off a connected Quest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dest", type=Path, metavar="DIR",
                        help=f"where to put them (default: <project>/{DEFAULT_DEST})")
    parser.add_argument("--package", metavar="NAME",
                        help="Android package to pull from (default: read from ProjectSettings)")
    parser.add_argument("--device", metavar="SERIAL",
                        help="which device, when more than one is connected")
    parser.add_argument("--adb", metavar="PATH", help="path to adb, if it cannot be found")
    parser.add_argument("--all", action="store_true",
                        help="re-pull sessions that are already here, overwriting them")
    parser.add_argument("--latest", action="store_true",
                        help="pull only the most recent recording, even if it is already here")
    parser.add_argument("--list", action="store_true",
                        help="show what is on the headset without copying anything")
    parser.add_argument("--analyze", action="store_true",
                        help="run the analysis on the sessions that were pulled")
    args = parser.parse_args(argv)

    repo = sa.find_repo_root()
    dest = args.dest or ((repo or Path.cwd()) / DEFAULT_DEST)
    package = args.package or sa.android_package(repo)
    if not package:
        print("error: could not read the package name from ProjectSettings.asset; "
              "pass --package com.Company.Product", file=sys.stderr)
        return 2

    try:
        adb = find_adb(args.adb)
        serial = pick_device(adb, args.device)
        remote = REMOTE_TEMPLATE.format(package=package)
        names = remote_sessions(adb, serial, remote)

        if not names:
            print(f"no sessions on the headset under {remote}")
            matches = installed_packages(adb, serial, package.split(".")[-1][:8])
            if matches and package not in matches:
                print("\nthat package is not installed. Similar ones on the device:")
                for match in matches:
                    print(f"  {match}")
                print("\nRe-run with --package <name>.")
            else:
                print("(play a session first, or the app has not written any yet)")
            return 0

        existing = {p.name for p in dest.iterdir()} if dest.is_dir() else set()
        if args.latest:
            # Names start with the recording's timestamp, so the last one is the newest.
            fresh = names[-1:]
        elif args.all:
            fresh = names
        else:
            fresh = [n for n in names if n not in existing]

        if args.latest:
            print(f"{len(names)} session(s) on the headset, newest is {fresh[0]}")
        else:
            print(f"{len(names)} session(s) on the headset, "
                  f"{len(names) - len(fresh)} already here, {len(fresh)} to copy")

        if app_is_running(adb, serial, package) and fresh:
            print(f"note: the game is still open, so {fresh[-1]} may still be "
                  "being written to. Quit it first if that session matters.")

        if args.list:
            for name in names:
                print(f"  {'have' if name in existing else 'new '}  {name}")
            return 0

        if not fresh:
            print("nothing new.")
            return 0

        dest.mkdir(parents=True, exist_ok=True)
        pulled = []
        for name in fresh:
            if pull(adb, serial, remote, name, dest):
                pulled.append(dest / name)
                print(f"  pulled {name}")

    except AdbError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"\n{len(pulled)} session(s) copied to {dest}")

    if pulled and args.analyze:
        import analyze_sessions
        return analyze_sessions.main([str(p) for p in pulled])
    if pulled:
        print("run: python Analysis/analyze_sessions.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
