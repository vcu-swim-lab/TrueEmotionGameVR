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


## 🎮 What the scene contains

`Assets/Scenes/Main.unity` holds two programs on the same object. **Only one may be ticked at a
time**, because both use the same screen text and the same predictor.

| Program | Component | What it is | In the committed scene |
|---|---|---|---|
| **Study protocol** | `Protocol Runner` | The research session: fixed parts, no score, detailed logs | **On** |
| **Emotion game** | `Progress Game` | The demo game: act the emoji, get a score | Off |

To switch, untick one and tick the other, then **Build and Run** again.


## ⚙️ Settings

| Setting | Where (Inspector) | Default | What it does |
|---|---|---|---|
| **Use Calibration** | `New Face Au Model` | Off | **Overridden at start-up** by the condition you enabled in `ConditionSetup.Configure` (see *The four conditions*): conditions 2 and 4 switch it on, 1 and 3 off. Leave it unticked in the scene |
| **Save Results** | `New Face Au Model` | On | Saves every prediction to a results file on the headset (see *Results*) |
| **Face Model** | `New Face Au Model` | the acted-data model | The emotion model; leave it as it is |
| **Player Id** | `Progress Game` | `player1` | The name a player's calibration is saved under |
| **Calibration Debug Logging** | `Protocol Runner` | Off | Writes each calibrated decision to the Unity console (`adb logcat`); development builds only |
| **Fast Rehearsal** | `Protocol Runner` | Off | Very short trials, to walk through the flow at a desk. Not valid data; cannot be combined with calibration |
| Timings, Randomization Seed | `Protocol Runner` | 2 / 3 / 1 / 4 s, 0 | Prompt, countdown, complete and relax times; seed 0 draws one from the clock |

The settings are saved in the scene, so **after changing one, Build and Run again** for the
headset (in the Editor, press Play again). A running session never changes mode halfway.


## 🔀 The four conditions

Every run uses exactly **one** of four conditions: which model predicts, crossed with whether the
player's calibration is used. The game uses the VREmotionDetection package for the model, the
calibration and the prediction, and nothing else.

| # | Condition | Model | Calibration |
|---|---|---|---|
| 1 | `sparsevr_nocal` | SparseVR, on the headset | off |
| 2 | `sparsevr_cal` | SparseVR, on the headset | on |
| 3 | `jev_nocal` | Jev, Typesafe's cloud model | off |
| 4 | `jev_cal` | Jev, Typesafe's cloud model | on (built from Jev's scores, about 70 requests) |

**Choosing one.** Open `Assets/Scripts/Conditions/ConditionSetup.cs`. `Configure` holds the four
blocks, each a few lines of package calls with comments. Exactly one is active (condition 1 as
committed). To test another: put `//` in front of the active block's lines, remove the `//` from the
block you want, **Build and Run**. Both programs (study protocol and emotion game) use it.

**Jev key (conditions 3 and 4).** Paste your own Typesafe key into `Assets/Resources/jev_api_key.txt`
(replacing the placeholder line), then build. The file is git-ignored; it is packed into the build,
so anyone with the APK can read the key: delete it from the file when you are done. In the Editor
the `TYPESAFE_API_KEY` environment variable also works.

**Checked before every run.** The screen shows the condition; for Jev it sends one test request
first. A rejected key or no connection stops the run with the reason on screen (the game offers
"press any button to try again"), instead of playing a session of failed predictions.

**No fallback.** If Jev does not answer for a prediction, that prediction fails: nothing is scored,
the log row says `backend_error` with the reason, and the local model is **not** asked instead.

**What is logged.** `session_metadata.json` -> `models`: `condition`, `condition_label`,
`face_backend` (`SparseVr` / `Jev`), `backend_id`, `use_calibration`, and for Jev the model, URL,
timeout, retries and `jev_send_own_faces` (never the key). `predictions.csv` ends with
`backend_id`, `error_code` and `raw_<Emotion>` (Jev's seven answers before they are divided by
their sum; blank for SparseVR). `status` is `ok`, `no_tracking` or `backend_error`. Jev scores
(`score_*`) are its answers divided by their sum; the game's confidence is Jev's own answer
(`raw_*`). The old `jev_predictions.csv` no longer exists. A pooled analysis must filter on
`condition`.

**Calibration files.** Saved per model under `<Player Id>-sparsevr` or `<Player Id>-jev`: the
library refuses to load one model's calibration into the other, and one file each means testing
both never overwrites the other.

**Privacy.** In conditions 3 and 4 the player's face-tracking numbers leave the device. Get consent.

Step-by-step headset checks for each condition: `Docs/CONDITION_TEST_CHECKLIST.md`.


## 🧪 Study protocol (`Protocol Runner`)

Each trial: prompt (emoji and name) → 3-2-1 countdown → hold the face for 10 seconds (one
prediction per second) → "Recording complete" → relax.

**Calibration on (conditions 2 and 4) — two parts:**

1. **Part 1: Calibration** — 7 trials, one per emotion, shown as "Calibration 3 of 7". The
   system learns how this person makes each expression.
2. **Part 2** — 14 trials (each emotion twice, shuffled), shown as "Trial 5 of 14". During each
   hold the screen shows **"Result: 😀 Happiness"**, updated every second. This is the final
   result: the calibrated emotion, which compares both the model's scores and the face movements
   with this person's own calibration.

**Calibration off (conditions 1 and 3) — one part:** the calibration part is skipped. The session is
the 14 trials, and "Result: …" shows the chosen model's own prediction (SparseVR's or Jev's).

"Result" is always the same value as the `emotion` column of the results file; the
`model_emotion` column holds the model's own prediction.

If face tracking is lost during calibration, the session continues without calibration and the
logs say why. The protocol version is `session-local-v2` (the participant sees the predicted
emotion); sessions from `session-local-v1`, which showed nothing, are not pooled with it.


## 😀 Emotion game (`Progress Game`)

1. **Calibration (only in conditions 2 and 4):** the player chooses
   - **A** — calibrate now (each of the 7 faces, about 90 seconds; saved for this Player Id),
   - **B** — skip,
   - **X** — use the saved calibration (shown only if one exists).
2. **Game:** six emotions in random order. For each: 3-2-1, then the emoji; the player acts it for
   10 seconds. A point for every second the predicted emotion matches, so a score out of 10.
3. **End screen:** the score per emotion and a summary of what the headset saw (share of each
   emotion and average confidence). Press any controller button to play again.

The first time it starts, the headset may ask for microphone permission. That comes from an
unfinished voice test and can be allowed or denied; the game works either way.


## 📊 Results

Two kinds of files are saved on the headset:

| | Simple results (from the library) | Detailed logs (from this game) |
|---|---|---|
| Content | Time, emotion, confidence, calibrated yes/no, the model's own emotion | Every prediction with all 7 scores and the 70 face-tracking values, the prompted emotion, trials, session settings |
| Saved by | `New Face Au Model` (**Save Results**) | `SessionLogger` |
| On the headset | `/sdcard/Android/data/com.DefaultCompany.EmotionGatheringTool/files/VREmotionDetection/results/` | `/sdcard/Android/data/com.DefaultCompany.EmotionGatheringTool/files/sessions/` |
| In the Editor (Play mode) | `%USERPROFILE%\AppData\LocalLow\DefaultCompany\Emotion Gathering Tool\VREmotionDetection\results\` | `%USERPROFILE%\AppData\LocalLow\DefaultCompany\Emotion Gathering Tool\sessions\` |
| Copy to the PC | Unity menu **VR Emotion > Get Results From Headset** → `HeadsetResults/` | `python Analysis/pull_sessions.py` → `Logs/quest-sessions/sessions/` |

**Copying the simple results:** connect the headset by USB, put it on and allow USB debugging if
asked, then in Unity choose **VR Emotion > Get Results From Headset** in the top menu bar. All
session files are copied and the folder opens; open them in Excel.

- File names show when the session started, in **UTC**: `session_<date>_<time>.csv`.
- **The files do not say who played.** Write down which session time belongs to whom.
- `pull_sessions.py` lives in the local `Analysis/` folder (a copy is in `Archive/Analysis/`).


## 🔒 Data and privacy

The results describe real people's faces. These folders are ignored by git and must never be
committed: `HeadsetResults/`, `Logs/`, `Analysis/` (scripts and participant data) and
`notebooks/`. A player's calibration is saved only on the headset.


## 🗂️ Repository layout

| Folder | Contents |
|---|---|
| `Assets/Scripts/` | `ProtocolRunner` (study), `ProgressGame` (emotion game), `SessionLogger` (detailed logs), `Conditions/` (`ConditionSetup`: the four conditions), `Calibration/` (session settings, trial voting, condition naming) |
| `Assets/Tests/EditMode/` | Tests: calibration math, parity with the Python analysis, scene setup, condition naming and key-file reading |
| `Assets/Resources/jev_api_key.txt` | Where you paste the Typesafe key for conditions 3 and 4 (git-ignored) |
| `Assets/Scenes/Main.unity` | The scene |
| `Docs/` | Study protocol and reports |
| `Archive/` | A snapshot of the offline analysis scripts, notebooks (code only) and reports, without data |


## ✅ Tests

In Unity: **Window > General > Test Runner > EditMode > Run All**. The scene tests expect the
committed scene to have **Use Calibration** off; untick it before committing the scene (the
conditions switch calibration on at run time).


## 📝 License

This project is licensed under the [MIT License](./LICENSE).
