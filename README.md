# TrueEmotionVR

A VR game for the Meta Quest Pro that detects the emotion on the player's face. The emotion
model and its optional per-player calibration come from the
[VREmotionDetection](https://github.com/vcu-swim-lab/VREmotionDetection) library; this project
runs the sessions around it.


## 📋 Prerequisites

- Unity **6000.0.33f1** (Unity 6) with **Android Build Support**.
- Meta Quest Pro in **developer mode**, with face tracking allowed.
- A USB cable, to build to the headset and to copy results.


## 📦 Installation

1. Clone this repository.
2. In Unity Hub, choose `Add > Add project from disk` and select the folder.
3. Open it. The VREmotionDetection library is downloaded automatically by the Package Manager.

**Working on the library at the same time?** Put a copy (or a folder link) of the library at
`Packages/com.vcu-swim-lab.vr-emotion-detection`. Unity then uses that local copy instead of the
GitHub one. Do not commit `Packages/packages-lock.json` while you work this way.


## 🎮 The game

`Assets/Scenes/Main.unity` runs one program, the **Protocol Runner** component. The player acts
each of the seven emotions and sees what the system predicted.

The screen is placed in front of the player's gaze when the game starts, and again when the
player recenters the headset. Its distance is the Canvas's Pos Z in the scene.

**Use Calibration on — two parts:**

1. **Part 1: Calibration** — 7 trials, one per emotion, shown as "Calibration 3 of 7". The
   system learns how this player makes each expression. No result is shown.
2. **Part 2** — 7 trials, one per emotion in a new random order, shown as "Trial 5 of 7", each
   followed by its result.

**Use Calibration off — one part:** Part 1 is skipped, and the results are the model's own.

Each trial:

1. **Get ready** (before the first trial of each part, and before every calibration trial):
   "Trial 1 of 7 / Relax yourself. / Next: Anger", then "Starts in" with a circle counting 5-4-3-2-1.
2. **Act** (10 s): "Act Anger", the emoji, and a bar showing the time left. No live result.
3. **Result** (Part 2 only): "In this trial, your predicted emotion was Anger" (green when it
   matches, red when not), the next emotion in yellow, then the circle countdown to it. After the
   last trial: "That was the last one. / Results in".

**End screen:** "Demo complete. Thank you." and a Target / Predicted list of the 7 Part 2 trials.

**The result** of a trial is the emotion predicted most often over its seconds 1–9 (the first
second is not used): the calibrated emotion when calibration is on, which compares both the
model's scores and the face movements with this player's own calibration; the model's own
otherwise. A second without face tracking does not count. If face tracking is lost during
calibration, the game continues without calibration and the logs say why.

The screens have a dark background and narrow text, so no line needs a head turn. The protocol
version is `demo-v2`. The earlier study flow (prompt, 3-2-1 countdown, live result and score, 14
test trials; versions `session-local-v1` to `v3`) and the older emotion game (`Progress Game`)
were removed on 2026-10-19; both are in git history, and their recordings are never pooled with
`demo-v2` ones.


## ⚙️ Settings

| Setting | Where (Inspector) | Default | What it does |
|---|---|---|---|
| **Use Calibration** | `New Face Au Model` | Off | Adds Part 1 and adapts the predictions to each player (movement calibration). Off: the model's own answer is used everywhere |
| **Save Results** | `New Face Au Model` | On | Saves every prediction to a results file on the headset (see *Results*) |
| **Face Model** | `New Face Au Model` | the acted-data model | The emotion model; leave it as it is |
| Intro / Part / Reading / Circle Countdown Seconds | `Protocol Runner` | 10 / 6 / 5 / 5 s | The start screen, the part screens, the reading time before the circle countdown, and the countdown itself |
| **Save Face Tracking** | `Protocol Runner` | On | Saves the 70 face-tracking values every frame to `face_tracking.csv`, for the avatar video (see *Video with avatar*) |
| **Fast Rehearsal** | `Protocol Runner` | Off | Very short trials, to walk through the flow at a desk. Not valid data; cannot be combined with calibration |
| Randomization Seed | `Protocol Runner` | 0 | The trial order; 0 draws one from the clock |
| **Calibration Debug Logging** | `Protocol Runner` | Off | Writes each calibrated decision to the Unity console (`adb logcat`); development builds only |

The settings are saved in the scene, so **after changing one, Build and Run again** for the
headset (in the Editor, press Play again). A running game never changes mode halfway.


## 🎥 Video with avatar

For a video of the game: the game (cast from the headset) on the left, the player's webcam at the
top right, and an avatar replaying the player's face at the bottom right.

1. **Record.** Tick **Save Face Tracking**, Build and Run. Cast the headset with Meta Quest
   Developer Hub (**Device Manager > Cast**) and record the cast window and the webcam with OBS
   (canvas 1920 x 720: game 1280 x 720 at the left, webcam 640 x 360 at the top right). Do not
   build in Unity while recording: the laptop runs out of memory.
2. **Copy the session.** Unity menu **VR Emotion > Get Sessions From Headset** copies every
   session folder to `HeadsetResults/sessions/`. A session with Save Face Tracking has
   `face_tracking.csv`: one row per frame (about 72 a second) with the UTC time, the time since the
   first get-ready screen, the screen showing, the trial, and the 70 values.
3. **Render the avatar.** Import **Meta XR Movement SDK > Samples > Body Tracking Samples** once
   (Package Manager; not committed, 134 MB). Then **VR Emotion > Avatar Replay > Render Face
   File…** and pick the session's `face_tracking.csv`. Unity builds a scene with the HighFidelity
   character, plays the file back on its face and saves `avatar.mp4` (1280 x 720) next to it, then
   stops by itself. Open `Main.unity` again afterwards.
4. **Combine.** `python Tools/add_avatar_to_video.py "<OBS video>.mp4" "<session folder>"` (needs
   ffmpeg: `winget install ffmpeg`). It finds where the first get-ready screen appears in the OBS
   video (from the clocks, then the exact screen change), puts the avatar in the bottom-right box
   from there, and writes `<name>_with_avatar.mp4`. `--offset` sets the time by hand.

The avatar's face uses `Assets/DemoReplay/retargeting_exact_1to1.json`: each tracked value moves
only the face shape of its own name, at full strength, so the avatar shows the values the model
received (Meta's own mapping amplifies and mixes several). The eye-direction values turn the eyes.
Not shown: head movement (not recorded) and the 7 tongue values (the character has no tongue
shapes).


## 📊 Results

Two kinds of files are saved on the headset:

| | Simple results (from the library) | Detailed logs (from this game) |
|---|---|---|
| Content | Time, emotion, confidence, calibrated yes/no, the model's own emotion | Every prediction with all 7 scores and the 70 face-tracking values, the prompted emotion, trials, session settings; with Save Face Tracking, also `face_tracking.csv` (every frame) |
| Saved by | `New Face Au Model` (**Save Results**) | `SessionLogger` (and `FaceTrackingRecorder`) |
| On the headset | `/sdcard/Android/data/com.DefaultCompany.EmotionGatheringTool/files/VREmotionDetection/results/` | `/sdcard/Android/data/com.DefaultCompany.EmotionGatheringTool/files/sessions/` |
| In the Editor (Play mode) | `%USERPROFILE%\AppData\LocalLow\DefaultCompany\Emotion Gathering Tool\VREmotionDetection\results\` | `%USERPROFILE%\AppData\LocalLow\DefaultCompany\Emotion Gathering Tool\sessions\` |
| Copy to the PC | Unity menu **VR Emotion > Get Results From Headset** → `HeadsetResults/` | Unity menu **VR Emotion > Get Sessions From Headset** → `HeadsetResults/sessions/` (or `python Analysis/pull_sessions.py` → `Logs/quest-sessions/sessions/`) |

**Copying the simple results:** connect the headset by USB, put it on and allow USB debugging if
asked, then in Unity choose **VR Emotion > Get Results From Headset** in the top menu bar. All
session files are copied and the folder opens; open them in Excel.

- File and folder names show when the session started, in **UTC**: `session_<date>_<time>.csv`.
  All times inside the files are UTC too (EDT is UTC − 4).
- **The files do not say who played.** Write down which session time belongs to whom.
- `pull_sessions.py` lives in the local `Analysis/` folder (a copy is in `Archive/Analysis/`).


## 🔒 Data and privacy

The results describe real people's faces. These folders are ignored by git and must never be
committed: `HeadsetResults/`, `Logs/`, `Analysis/` (scripts and participant data) and
`notebooks/`. A player's calibration is kept only for the running game.


## 🗂️ Repository layout

| Folder | Contents |
|---|---|
| `Assets/Scripts/` | `ProtocolRunner` (the game), `GameScreen` (its trial screens), `FaceTrackingRecorder` (the face file), `SessionLogger` (detailed logs), `Calibration/` (session settings and trial voting) |
| `Assets/Editor/` | Menu **VR Emotion > Get Sessions From Headset** |
| `Assets/DemoReplay/` | Avatar replay: the CSV face player, the exact 1:1 face mapping, and the **Avatar Replay** menu (Editor only) |
| `Tools/` | `add_avatar_to_video.py`: puts the avatar into the OBS recording |
| `Assets/Tests/EditMode/` | Tests: calibration math, parity with the Python analysis, scene setup |
| `Assets/Scenes/Main.unity` | The scene |
| `Docs/` | Study protocol and reports |
| `Archive/` | A snapshot of the offline analysis scripts, notebooks (code only) and reports, without data |


## ✅ Tests

In Unity: **Window > General > Test Runner > EditMode > Run All**. The scene tests expect the
committed scene to have **Use Calibration** off; untick it before committing the scene.


## 📝 License

This project is licensed under the [MIT License](./LICENSE).
