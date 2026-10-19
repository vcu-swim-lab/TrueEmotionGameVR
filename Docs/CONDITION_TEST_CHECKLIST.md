# Testing the four conditions on the headset

One condition per build. Test them in order 1 → 4; each needs one Build and Run.

| # | Model | Calibration | Needs the Jev key |
|---|---|---|---|
| 1 | SparseVR (on the headset) | off | no |
| 2 | SparseVR | on | no |
| 3 | Jev (Typesafe's cloud model) | off | **yes** |
| 4 | Jev | on | **yes** |

## Before the first build

1. Open the project in Unity 6 (6000.0.33f1) and wait until the Package Manager has finished (the
   library and Newtonsoft JSON are downloaded automatically).
2. The Console must show **no compile errors**. Then **Window > General > Test Runner > EditMode > Run
   All**: everything green.
3. Conditions 3 and 4 only: create `Assets/Resources/jev_api_key.txt` and put your own Typesafe key
   (console.typesafe.ai/keys) on its first line. The file is git-ignored; delete the key from it when
   you are done. On a second computer, create the file there by hand.
4. The headset must be online for conditions 3 and 4.

## Choosing the condition

Select the object with the **Protocol Runner** in `Main.unity` and pick the **Condition** in its
Inspector (1 to 4). Save the scene, then **Build and Run**.

The condition also switches the predictor's **Use Calibration** when the game starts, so change the
Condition, not that checkbox. Commit the scene only with condition 1 (the default).

## What to check, for every condition

Watch the log while you test: `adb logcat -s Unity | grep -E "\[condition\]|\[Jev\]|ProtocolRunner"`

In conditions 3 and 4 the library also writes one line per Jev request, for example
`[Jev] POST https://api.typesafe.ai/v1/systemone -> 200 in 412 ms (attempt 1 of 2)`. If you see these
lines with `200`, Jev is really being called. They never contain the key.

| Step | You should see |
|---|---|
| Start | Screen: `Condition N: …` and `Starting...` (Jev: `Checking the Jev connection...`). Log: `[condition] Condition N: … \| backend=… \| calibration=on/off`; Jev also `Jev OK (… ms)` |
| Part 1 (2, 4 only) | "Part 1: Calibration", then 7 trials: get ready with the circle, act for 10 s. No result screen. In condition 4 this sends about 70 Jev requests |
| Part 2 | 7 trials: get ready (first trial only), act for 10 s, then "In this trial, your predicted emotion was …" and the circle countdown to the next |
| End | "Demo complete. Thank you." and the Target / Predicted list |

## Per-condition expectations

| # | `backend=` in the log | Extra checks |
|---|---|---|
| 1 | the face-model name (`sparseVR_teacher_model_…`) | No Part 1. No network traffic |
| 2 | the face-model name | Part 1 runs; `session_metadata.json` says `calibration_valid: true` and 9 ticks per emotion |
| 3 | `jev:jev-1.13.0:p1` | No Part 1. `raw_*` columns filled in `predictions.csv` |
| 4 | `jev:jev-1.13.0:p1` | Part 1 runs, about 70 requests; then as condition 2 |

## Failure checks (Jev, conditions 3 and 4)

Do these once each:

| Do this | Expected |
|---|---|
| Put a wrong key in `jev_api_key.txt`, rebuild | `Cannot start.` + `Jev failed: Unauthorized. The key was rejected…`. **No session folder is created** |
| Delete the key file (or leave its first line empty), rebuild | `Cannot start.` + `No Jev key. Paste your Typesafe key into Assets/Resources/jev_api_key.txt…` |
| Turn the headset's Wi-Fi off after the game has started | Those seconds have no prediction; the CSV rows say `status=backend_error`, `error_code=Unreachable`. **The local model is never used instead** |
| Turn Wi-Fi off during Part 1 (condition 4) | Calibration cannot collect 9 seconds per emotion: the game continues uncalibrated and `session_metadata.json` says `calibration_valid: false` with the reason |

## Where the results are

- Unity menu **VR Emotion > Get Sessions From Headset** copies the session folders to
  `HeadsetResults/sessions/`. In each: `predictions.csv`, `trials.csv`, `session_metadata.json`
  (`models.condition`, `face_backend`, `backend_id`, `use_calibration`, and for Jev its settings — never
  the key) and, with Save Face Tracking, `face_tracking.csv`.
- The library's simple results: Unity menu **VR Emotion > Get Results From Headset**.
- File formats: [OUTPUT_FILE_FORMATS.md](OUTPUT_FILE_FORMATS.md).

## Reading a result

- Always filter a pooled analysis on `models.condition`. SparseVR and Jev scores mean different
  things: SparseVR's are probabilities; Jev's `score_*` are its seven answers divided by their sum, and
  `raw_*` are the answers themselves.
- A condition 2 or 4 session whose `calibration_valid` is `false` ran without a usable calibration; its
  calibrated columns are blank.

## Done when

All four conditions start, show their own label, predict with the right backend (`backend=` in the log
and `backend_id` in the CSV), and the failure checks behave as above.
