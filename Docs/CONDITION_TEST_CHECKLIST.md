# Testing the four conditions on the headset

One condition per build. Test them in order 1 → 4; each needs one Build and Run. Do the whole list
for the **emotion game** first (quicker), then once for the **study protocol**.

| # | Model | Calibration | Needs the Jev key |
|---|---|---|---|
| 1 | SparseVR | off | no |
| 2 | SparseVR | on | no |
| 3 | Jev | off | **yes** |
| 4 | Jev | on | **yes** |

## Before the first build

1. Open the project in Unity 6 (6000.0.33f1). The package resolves from
   `VREmotionDetection.git#jev_branch`; Unity rewrites `Packages/packages-lock.json` (review that diff).
2. The Console must show **no compile errors**. Then **Window > General > Test Runner > EditMode > Run
   All**: everything green.
3. Make sure only the program you are testing is ticked on the scene object: **Progress Game** (the
   game) or **Protocol Runner** (the study), not both.
4. Conditions 3 and 4 only: open `Assets/Resources/jev_api_key.txt` and replace the line
   `PASTE_YOUR_TYPESAFE_API_KEY_HERE` with your own key (console.typesafe.ai/keys). Save. The file is
   git-ignored; delete the key from it when you are done.
5. The headset must be online for conditions 3 and 4.

## Choosing the condition

Open `Assets/Scripts/Conditions/ConditionSetup.cs`, method `Configure`. Four blocks, one active:

```csharp
// ===== CONDITION 1: SparseVR, calibration OFF =====
predictor.UseSparseVr();
predictor.UseCalibration = false;
// ===== CONDITION 2: SparseVR, calibration ON =====
// predictor.UseSparseVr();
// predictor.UseCalibration = true;
...
```

Put `//` in front of the lines of the active block, remove the `//` from the lines of the block you
want. Only one block may be uncommented. Save, then **Build and Run**.

## What to check, for every condition

Watch the log while you test: `adb logcat -s Unity | grep -E "\[condition\]|\[game\]"`

| Step | You should see |
|---|---|
| Start | Screen: `Condition N: …`. Log: `[condition] Condition N: … \| backend=… \| calibration=on/off`. Jev: `Checking the Jev connection…`, then `Jev OK (… ms)` (the game shows it on screen; the study only logs it and goes on to the introduction) |
| Calibration (2, 4 only) | The game asks `A: calibrate / B: skip` (and `X: use your saved calibration` the second time). Press A: 7 faces, about 90 s. `Calibration done.` In condition 4 this sends about 70 requests |
| Play | The game: 6 rounds of 10 ticks, log lines `[game] round … tick …: <Emotion> <confidence> (<condition id>, calibrated: …)`. The study: `Result: <emoji> <Emotion>` every second in Part 2 |
| End | The game: scores plus `What the headset saw this session (Condition N: …)` |

## Per-condition expectations

| # | `backend=` in the log | Extra checks |
|---|---|---|
| 1 | the face-model name (`sparseVR_teacher_model_…`) | No calibration question. No network traffic |
| 2 | the face-model name | Calibration question. Second launch: press X, `Saved calibration loaded.`. The saved file is `player1-sparsevr.json` |
| 3 | `jev:jev-1.13.0:p1` | No calibration question. `raw_*` columns filled in `predictions.csv` |
| 4 | `jev:jev-1.13.0:p1` | Calibration question, about 70 requests. Second launch: press X. Saved file `player1-jev.json`; it is **not** offered or loaded in condition 2 |

## Failure checks (Jev, conditions 3 and 4)

Do these once each:

| Do this | Expected |
|---|---|
| Put a wrong key in `jev_api_key.txt`, rebuild | `Jev failed: Unauthorized. The key was rejected…`. The game offers `Press any button to try again`; the study shows `Cannot start.` + the reason. **No session folder is created** |
| Leave the placeholder in the file, rebuild | `Cannot start. No Jev key. Paste your Typesafe key into Assets/Resources/jev_api_key.txt…` |
| Turn the headset's Wi-Fi off after the game has started | Ticks give no point; the log says `no prediction (… Unreachable …)`; the CSV rows say `status=backend_error`, `error_code=Unreachable`. **The local model is never used instead** |
| Turn Wi-Fi off before pressing A to calibrate (condition 4) | `Calibration did not work: …` and the question again (A retry / B skip) |

## Where the results are

- Detailed logs: `/sdcard/Android/data/com.DefaultCompany.EmotionGatheringTool/files/sessions/session_*/`
  - `session_metadata.json` -> `models`: `condition`, `face_backend`, `backend_id`, `use_calibration`
    (Jev: model, URL, timeout, retries, `jev_send_own_faces`). Never the key.
  - `predictions.csv`: ends with `backend_id`, `error_code`, `raw_<Emotion>`; `status` is `ok`,
    `no_tracking` or `backend_error`.
- Simple results from the library: Unity menu **VR Emotion > Get Results From Headset**.

## Reading a result

- Always filter a pooled analysis on `models.condition`. SparseVR and Jev scores mean different things:
  SparseVR's are probabilities; Jev's `score_*` are its seven answers divided by their sum, and
  `raw_*` are the answers themselves (the confidence the game reports).
- A condition-4 session whose `calibration_valid` is `false` ran without a usable calibration; its
  calibrated columns are blank.

## If Unity shows "ExecuteMenuItem target for Audio does not exist"

That message is not from this project's code (nothing here calls `ExecuteMenuItem`) and it is not a C#
compile error. It means the Editor is in a half-loaded state, usually after the package changed. Try, in
order, and stop at the first that works:

1. **Window > Layouts > Default**, then Play.
2. Close Unity, delete `Packages/packages-lock.json` (Unity writes a new one), reopen, and wait until the
   spinner at the bottom right stops. Check **Window > Package Manager > In Project**: *VR Emotion
   Detector* 2.2.0 (git, `jev_branch`) must be listed without an error icon.
3. Close Unity, delete the `Library/` folder, reopen (a full reimport, 10-30 minutes).
4. Still blocked? Find the real error in the Editor log (PowerShell) and send it:
   ```powershell
   Select-String -Path "$env:LOCALAPPDATA\Unity\Editor\Editor.log" -Pattern "error CS|ExecuteMenuItem|error:|Exception|Package Manager|Safe Mode" | Select-Object -First 40
   ```

Remember `Assets/Resources/jev_api_key.txt` is git-ignored: on a second computer (for example the Windows
PC) create it there by hand before testing conditions 3 and 4.

## Done when

All four conditions start, show their own label, predict with the right backend (`backend=` in the log
and `backend_id` in the CSV), and the failure checks behave as above.
