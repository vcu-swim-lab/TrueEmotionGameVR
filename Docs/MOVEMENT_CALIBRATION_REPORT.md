# Movement calibration

**Status (2026-10-07):** implemented as the optional runtime mode `movement`, tested offline on 9
headset sessions and live on 1 session. All sessions so far except one are the developer's face.

## 1. Idea

The teacher model learns emotions from other people. People show the same emotion with different
face shapes:

- the developer's face on different days has a same-emotion shape cosine of 0.88;
- two different training people have 0.50;
- the developer's emotions are recognised at 92% from his own other days, and at 67% from training
  people (`Analysis/exploratory/day_vs_person.py`).

The current personal calibration works on the model's 7 output scores. The model has already reduced
70 action units (AUs) to 7 numbers there, so some detail of the player's face is lost.

Movement calibration also remembers **how the player's face moves** for each emotion, and compares
every new moment with those movements.

## 2. Method

**Terms**
- frame: one measurement of 70 values (63 face AUs and 7 tongue values), 10 per second;
- window: the last 5 frames (0.5 s);
- tick: one answer per second, made from the newest window. A trial has ticks 0 to 9; tick 0 is
  skipped.

**Calibration block** (each emotion once, ticks 1 to 9)
1. For each tick, make two features:
   - score feature: `log(max(score, 1e-12))` of the 7 model scores (the same as personal calibration);
   - movement feature: the mean over the window's 5 frames of each of the 63 face AUs (tongue left out).
2. For each emotion, average its 9 ticks. Result: 7 score centroids (7 × 7) and 7 movement
   centroids (7 × 63).
3. All 9 ticks of every emotion must be tracked. If not, the calibration is invalid and no
   calibrated prediction is made (no fallback).

**Evaluation block** (every tick)
1. Make the two features of the tick.
2. For each emotion: total = cosine(score feature, score centroid) + cosine(movement feature,
   movement centroid). Cosine compares the shape and ignores the strength, so a quiet face still
   matches.
3. The highest total wins. An exact tie goes to the lower class index.
4. A trial is the majority of its 9 tick decisions (the existing tie rule).

**Why it runs alone.** The score term is the personal calibration itself. Combining movement with
personal calibration would count it twice. A combination with the population profile was never
measured.

## 3. Offline result: 9 headset sessions

Script: `Analysis/exploratory/au_calibration.py`. Results:
`Analysis/results/au_calibration_sessions.csv` and `au_calibration_per_emotion.csv`.

The model is D (`sparseVR_teacher_model_acted_only.onnx`). Each method was fitted on the calibration
block and scored on the evaluation block of the same session. Nothing was tuned on the evaluation
block.

| Method | Tick macro-F1 | Trial macro-F1 | Sessions better than personal |
|---|---|---|---|
| No calibration | 0.587 | 0.588 | — |
| Personal calibration (scores) | 0.647 | 0.672 | — |
| Movement only (AU fingerprints) | 0.714 | 0.717 | 8 of 9 (Wilcoxon p = 0.008) |
| **Movement calibration (scores + movement)** | **0.730** | **0.725** | **9 of 9 (p = 0.004)** |

Other variants were tested and gave no clear gain over personal calibration (4 to 5 of 9 sessions):
Euclidean distance, Neutral-subtracted AUs, z-scored AUs, LDA, kNN, fused LDA, and blending with
population AU templates. Only cosine works with 9 ticks per emotion.

The result is the same with all 70 AUs, and with the last frame instead of the mean. With model C,
movement calibration gives 0.732.

Evaluation tick accuracy per emotion, mean of 9 sessions:

| Method | Anger | Disgust | Fear | Happiness | Neutral | Sadness | Surprise |
|---|---|---|---|---|---|---|---|
| No calibration | 0.49 | 0.59 | 0.07 | 0.86 | **0.94** | 0.88 | 0.52 |
| Personal | 0.62 | 0.77 | 0.17 | 0.83 | 0.69 | 0.82 | **0.78** |
| **Movement** | **0.83** | **0.84** | **0.25** | **0.87** | 0.78 | **0.91** | 0.77 |

## 4. Live result: session `20261007_153229_55a206`

The developer's face, model D, mode `movement`. Report:
`Analysis/results/runtime_calibration/movement_20261007_153229_55a206.json`.

**Parity: OK.** All 126 evaluation ticks and 14 trials were recomputed offline in Python. Every
class, similarity, margin and trial vote matches what the headset wrote.

| | Tick macro-F1 | Trial macro-F1 | Trials correct |
|---|---|---|---|
| No calibration (live, D) | 0.561 | 0.619 | 9 / 14 |
| Personal (offline replay, D) | 0.694 | 0.720 | — |
| **Movement (live, D)** | **0.809** | **0.848** | **12 / 14** |
| Movement (offline replay, C) | 0.823 | 0.848 | — |

Correct ticks per emotion (of 18):

| Emotion | Face size | No calibration | Personal | Movement | Where movement's errors went |
|---|---|---|---|---|---|
| Fear | 0.61 | 0 | 10 | **14** | Disgust 2, Neutral 2 |
| Disgust | 3.02 | 9 | 9 | **18** | — |
| Anger | 1.26 | 8 | 10 | **11** | Disgust 7 |
| Happiness | 3.08 | 9 | 9 | **11** | Neutral 6, Disgust 1 |
| Sadness | 2.11 | 14 | 14 | **16** | Neutral 2 |
| Surprise | 2.16 | 14 | 16 | **17** | Fear 1 |
| Neutral | 0.57 | **18** | **18** | 14 | Disgust 4 |
| **Total** | | 72 / 126 | 86 / 126 | **101 / 126** | |

Notes:
- Fear is the largest gain (0 → 14). The developer's Fear is as small as Neutral (size 0.61 vs 0.57),
  but its shape (eyes wide, brows up, jaw open) is distinct.
- Disgust is now always correct. Without calibration it went to Anger, and with personal
  calibration to Sadness.
- Anger is still the weakest emotion. Its misses moved from Neutral to Disgust, because both use
  "brows down + eyelids tight".
- The first Happiness evaluation trial is wrong in every method. In that trial the smile faded:
  the lip-corner value fell from 0.16 to 0.01, against about 0.6 in calibration and in the second
  trial. This is an acting problem, not a model problem.
- Neutral loses 4 ticks to Disgust. A resting face has almost no shape, so the cosine is unsure.

## 5. Implementation

| Part | Where |
|---|---|
| Mode, checkbox rules | `Assets/Scripts/Calibration/CalibrationMode.cs`, `CalibrationSessionConfig.cs` |
| Movement feature, sum rule | `Assets/Scripts/Calibration/CalibrationMath.cs` (`MovementFeature`, `NearestOfSum`) |
| Movement centroids | `Assets/Scripts/Calibration/MovementCentroidBuilder.cs` |
| Calibrate and classify | `Assets/Scripts/Calibration/RuntimeCalibrator.cs` |
| Inspector checkbox, exclusivity | `Assets/Scripts/RuntimeCalibrationSettings.cs` (`enableMovementCalibration`, `OnValidate`) |
| Window passed to the calibrator | `Assets/Scripts/ProtocolRunner.cs` |
| Logging | `Assets/Scripts/SessionLogger.cs`: column `movement_calibration_enabled`; metadata `movement_feature`, `movement_rule`, `movement_centroid_built` |
| Python reference | `Analysis/calibration_study.py` (`movement_feature`, `classify_movement`) |
| Live verification | `Analysis/runtime_calibration_report.py --verify` (recomputes from the logged score and AU columns) |
| Parity fixture | `Analysis/make_runtime_parity_fixture.py`, schema v2, case `movement_confused` |

> **Update 2026-10-12.** The personal, population and hybrid modes, `RuntimeCalibrator` and
> `RuntimeCalibrationSettings` were removed. Calibration is now movement calibration only, from
> the VREmotionDetection library (`MovementCalibrator`), and it is switched on by the **Use
> Calibration** checkbox on the **New Face Au Model** component (off by default). The debug
> logging checkbox is on `ProtocolRunner`. This section records the setup as it was.

**How to use.** On `RuntimeCalibrationSettings` (the `Canvas` object in `Main.unity`), tick only
**Enable Movement Calibration**. Ticking it clears the other two checkboxes, and ticking either of
them clears it. `CalibrationSessionConfig.Lock` refuses the combination at session start in case
the scene file was edited by hand. In movement mode, the winner, runner-up and margin columns are
sums of two cosines (0 to 2).

**Tests (2026-10-07)**

| Suite | Result |
|---|---|
| Unity EditMode, batch mode (6000.0.33f1) | 137 / 137 |
| `dotnet test Analysis/CsharpTests` | 123 / 123 |
| `python Analysis/test_runtime_calibration_report.py` | 27 / 27 |
| `python Analysis/test_calibration_study.py` | 67 / 67 |

The parity case `movement_confused` decides 30 ticks differently from personal-only, so an error
in the movement path fails it. `calibration_study.classify_movement` reproduces the exploratory
result exactly on all 9 sessions (0.730).

## 6. Limits

- **One person.** 9 of the 10 sessions are the developer. The one other person (`26908f`, offline)
  also improved: 0.576 → 0.647.
- **Same session.** Calibration and test are minutes apart. A new day or a refitted headset can
  change the shapes.
- **One calibration trial per emotion.** A badly acted calibration trial gives a bad fingerprint.
- **Neutral** gets worse than with no calibration (offline 0.94 → 0.78; live 18 → 14 ticks).
- **Selection.** 9 methods were compared offline before this one was chosen. It was among the
  prespecified ideas and won in every session, so the risk is small but not zero.

## 7. Next steps

1. Record **other people** with movement calibration, with an anonymous participant ID.
2. Model: D and C are about equal with movement calibration (offline 0.730 vs 0.732; live session
   0.809 vs 0.823). Alternating them between participants would answer which is better.
3. Neutral: test a rule for faces with very small movement (for example, a minimum face size before
   the movement term is used).
4. Anger vs Disgust: look at which AUs separate them for the developer, and whether a second
   calibration trial for these two would help.
