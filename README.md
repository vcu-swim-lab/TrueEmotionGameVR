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
| **Use Calibration** | `New Face Au Model` | On (temporarily, while Jev is tested) | Adapts the predictions to each player (movement calibration). Off: the model's own answer is used everywhere |
| **Save Results** | `New Face Au Model` | On | Saves every prediction to a results file on the headset (see *Results*) |
| **Face Model** | `New Face Au Model` | the acted-data model | The emotion model; leave it as it is |
| **Player Id** | `Progress Game` | `player1` | The name a player's calibration is saved under |
| **Calibration Debug Logging** | `Protocol Runner` | Off | Writes each calibrated decision to the Unity console (`adb logcat`); development builds only |
| **Fast Rehearsal** | `Protocol Runner` | Off | Very short trials, to walk through the flow at a desk. Not valid data; cannot be combined with calibration |
| Timings, Randomization Seed | `Protocol Runner` | 2 / 3 / 1 / 4 s, 0 | Prompt, countdown, complete and relax times; seed 0 draws one from the clock |

The settings are saved in the scene, so **after changing one, Build and Run again** for the
headset (in the Editor, press Play again). A running session never changes mode halfway.


## 🌐 Jev (temporary, on by default while testing)

Besides the local model, each prediction can be sent to [Typesafe's Jev model](https://docs.typesafe.ai),
an online service. The local model still scores every tick and is what calibration and
`predictions.csv` use; Jev's answer is what the game scores and what the study screen shows.

- **It is on by default** (`Jev Scorer`, added to the Canvas when the game starts, `Use Jev` ticked).
  To switch it off, add a `Jev Scorer` component to the Canvas and untick **Use Jev**, or set the
  default in `Assets/Scripts/Jev/JevScorer.cs` to `false`. **Do this before collecting study data**:
  the study protocol is written for the local model.
- **Key:** set the `TYPESAFE_API_KEY` environment variable before opening Unity (Editor). For a
  headset build, create **Assets > Create > VR Emotion > Jev Settings**, paste the key there and
  assign it to the `Jev Scorer`. That asset is git-ignored; never commit it. A key inside a build can
  be extracted, so for anything shared use a proxy (the asset's Base Url).
- **No key or no network:** the game keeps running on the local model, and the console warns once.
- **Privacy:** while Jev is on, the player's face-tracking numbers leave the device. Get consent.
- **Logs:** `session_metadata.json` records `face_backend`; Jev's answers are in
  `jev_predictions.csv` (join on `trial_id` and `prediction_index`). Only evaluation trials ask Jev
  in the study, and the trial votes in `trials.csv` stay on the local model.
- **Calibration:** with **Use Calibration** on and a finished (or loaded) calibration, each request also
  carries that player's own movement signature for every emotion, as the library does, and
  `jev_predictions.csv` says so in `profile_sent`. Without a calibration Jev judges from general
  knowledge of expressions. The calibration itself is always built from the local model.


## 🧪 Study protocol (`Protocol Runner`)

Each trial: prompt (emoji and name) → 3-2-1 countdown → hold the face for 10 seconds (one
prediction per second) → "Recording complete" → relax.

**Use Calibration on — two parts:**

1. **Part 1: Calibration** — 7 trials, one per emotion, shown as "Calibration 3 of 7". The
   system learns how this person makes each expression.
2. **Part 2** — 14 trials (each emotion twice, shuffled), shown as "Trial 5 of 14". During each
   hold the screen shows **"Result: 😀 Happiness"**, updated every second. This is the final
   result: the calibrated emotion, which compares both the model's scores and the face movements
   with this person's own calibration.

**Use Calibration off — one part:** the calibration part is skipped. The session is the 14
trials, and "Result: …" shows the model's own prediction.

"Result" is always the same value as the `emotion` column of the results file; the
`model_emotion` column holds the model's own prediction.

If face tracking is lost during calibration, the session continues without calibration and the
logs say why. The protocol version is `session-local-v2` (the participant sees the predicted
emotion); sessions from `session-local-v1`, which showed nothing, are not pooled with it.


## 😀 Emotion game (`Progress Game`)

1. **Calibration (only when Use Calibration is ticked):** the player chooses
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
| `Assets/Scripts/` | `ProtocolRunner` (study), `ProgressGame` (emotion game), `SessionLogger` (detailed logs), `Calibration/` (session settings and trial voting), `Jev/` (temporary copy of the library's Jev client, plus `JevScorer`) |
| `Assets/Tests/EditMode/` | Tests: calibration math, parity with the Python analysis, scene setup |
| `Assets/Scenes/Main.unity` | The scene |
| `Docs/` | Study protocol and reports |
| `Archive/` | A snapshot of the offline analysis scripts, notebooks (code only) and reports, without data |


## ✅ Tests

In Unity: **Window > General > Test Runner > EditMode > Run All**. The scene tests expect the
committed scene to have **Use Calibration** on (temporary, while Jev is tested); change the scene and `SceneWiringTests` together when it goes back to off.


## 📝 License

This project is licensed under the [MIT License](./LICENSE).
