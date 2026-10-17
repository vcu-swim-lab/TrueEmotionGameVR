# What a run writes: the four pipelines and their files

Every run uses one condition (see `ConditionSetup.cs`):

| # | Condition id | Model | Calibration |
|---|---|---|---|
| 1 | `sparsevr_nocal` | SparseVR (on the headset) | off |
| 2 | `sparsevr_cal` | SparseVR | on |
| 3 | `jev_nocal` | Jev (cloud) | off |
| 4 | `jev_cal` | Jev | on |

The files have the **same columns in all four conditions**. What changes is which columns are filled in
(section 7). The example values below come from a test run with a fake face and a fake Jev server, so they show
the structure, not real accuracy.

## 1. Where the files are

On the headset, under `/sdcard/Android/data/com.DefaultCompany.EmotionGatheringTool/files/`
(in the Editor: `%USERPROFILE%\AppData\LocalLow\DefaultCompany\Emotion Gathering Tool\`):

| File | Written by | Game | Study | Conditions |
|---|---|---|---|---|
| `sessions/session_<utc>_<6 hex>/predictions.csv` | the game's `SessionLogger` | yes | yes | all |
| `sessions/session_<utc>_<6 hex>/session_metadata.json` | `SessionLogger` | yes | yes | all |
| `sessions/session_<utc>_<6 hex>/trials.csv` | `SessionLogger` | no | yes | all |
| `VREmotionDetection/results/session_<utc>.csv` | the library (`NewFaceAuModel`, **Save Results**) | yes | yes | all |
| `VREmotionDetection/calibration/<playerId>-<model>.json` | the library | yes | no | 2, 4 |

* A session folder exists only after the condition passed its check. A run that stops with "Cannot start"
  (no Jev key, key rejected, no connection) writes no session folder.
* `jev_predictions.csv` no longer exists; its content moved into `predictions.csv` (`raw_*` columns).
* The study does not save its calibration (it recalibrates each session); the game does, under
  `<Player Id>-sparsevr` or `<Player Id>-jev`.
* Copy `sessions/` with `python Analysis/pull_sessions.py`; copy the library files with the Unity menu
  **VR Emotion > Get Results From Headset**.

## 2. `predictions.csv` (one row per prediction tick)

Game: **386 columns**, 60 rows per round (6 emotions x 10 ticks). Study: **394 columns** (the same up to the
action units, then 12 calibration columns instead of the game's 4), 140 rows without calibration (14 trials x 10
ticks) or 210 with it (21 trials x 10). Columns appear in this order:

| # | Columns | Meaning |
|---|---|---|
| 1 | `timestamp_utc` | `yyyy-MM-ddTHH:mm:ss.fffZ` when the row was written |
| 2 | `session_id` | `<yyyyMMdd_HHmmss>_<6 hex>`; same on every row of the folder |
| 3 | `trial_id` | 0-based prompt number (game: one per emotion, counting on across restarts) |
| 4 | `block_role` | study: `calibration` or `evaluation`; **game: blank** |
| 5 | `repeat_index` | study: 1-based presentation of that emotion in its block; **game: blank** |
| 6 | `prompted_emotion` | the emotion the player was asked to act |
| 7 | `prediction_index` | 0-based tick inside the trial (0 to 9) |
| 8 | `predicted_emotion` | the **model's own** answer (argmax of `score_*`); blank when `status` is not `ok` |
| 9 | `correct` | `1` if `predicted_emotion` equals `prompted_emotion`, else `0` (also `0` on a failed tick) |
| 10 | `status` | `ok`, `no_tracking` (no valid face window) or `backend_error` (the model failed, see `error_code`) |
| 11 | `generation` | count of valid face frames so far; a small jump means tracking dropped out |
| 12-18 | `score_Anger` ... `score_Surprise` | seven scores in this order: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise |
| 19-23 | `frame_0_valid` ... `frame_4_valid` | always `1` for logged windows |
| 24-373 | `frame_<f>_au_<a>` | the 5 x 70 face-tracking values the model saw (frame 0 oldest, 63 face + 7 tongue values) |
| game only | `calibrated_predicted_emotion`, `calibrated_winner_similarity`, `calibrated_runner_up_similarity`, `calibrated_margin` | the calibration's decision for the tick, **what the game scored on** in conditions 2 and 4; all four blank when calibration is off or was skipped, and on a failed tick |
| study only | `calibration_mode`, `personal_calibration_enabled`, `population_calibration_enabled`, `movement_calibration_enabled`, `hybrid_alpha`, `raw_predicted_emotion`, `calibrated_predicted_emotion`, `calibrated_winner_similarity`, `calibrated_runner_up_similarity`, `calibrated_margin`, `calibration_valid`, `calibration_error` | see below |
| last 9 | `backend_id`, `error_code`, `raw_Anger` ... `raw_Surprise` | which model answered; why it failed; Jev's own answers |

In the **game**, `predicted_emotion`, `correct` and `score_*` are always the model's own answer; the score the
player saw is `calibrated_predicted_emotion` when it is filled and `predicted_emotion` otherwise. The four game
columns use the same names, and the same meaning, as the study's columns of the same name, so one analysis reads
both. They are present in every condition of the game (blank in 1 and 3), so the game has one schema.

The calibration columns (study only):

| Column | Value |
|---|---|
| `calibration_mode` | `raw` (conditions 1, 3) or `movement` (2, 4); fixed at session start |
| `personal_/population_calibration_enabled`, `hybrid_alpha` | kept for old readers: always `false`, `false`, blank |
| `movement_calibration_enabled` | `true` in conditions 2, 4 |
| `raw_predicted_emotion` | the model's argmax (equal to `predicted_emotion`) |
| `calibrated_predicted_emotion` | the calibration's decision; **blank** in conditions 1, 3, in the calibration block, on tick 0 and on a failed tick |
| `calibrated_winner_similarity`, `calibrated_runner_up_similarity`, `calibrated_margin` | the calibration's scores (0 to 2) for the winner and runner-up and their difference; blank when there is no decision |
| `calibration_valid`, `calibration_error` | `true`/`false` and the reason; blank on calibration-block rows (known only once the block ends). Note `true` for raw conditions too: there is nothing to invalidate |

`status` and `error_code`:

| `status` | `error_code` | When |
|---|---|---|
| `ok` | blank | a prediction was made |
| `no_tracking` | blank or `NoTracking` | no valid face window in time |
| `backend_error` | `NotConfigured`, `NotReady`, `Busy`, `InvalidInput`, `Unauthorized`, `InvalidRequest`, `RateLimited`, `Timeout`, `Unreachable`, `ServerError`, `InvalidResponse`, `Internal`, `Canceled` | the model could not answer. For Jev: `Unauthorized` = key, `Unreachable`/`Timeout` = network, `RateLimited` = too many requests, `InvalidResponse` = Jev's answer was unusable, `Canceled` = the model was switched or the app paused mid-request |

On a failed row `predicted_emotion`, the scores, the frames and `raw_*` are blank; `backend_id` is still filled.

## 3. `trials.csv` (study only; one row per trial)

17 columns: `session_id`, `trial_id`, `block_role`, `repeat_index`, `prompted_emotion`, `calibration_mode`,
`hybrid_alpha` (blank), `raw_trial_prediction`, `raw_trial_correct`, `raw_scored_ticks`, `raw_trial_tie`,
`calibrated_trial_prediction`, `calibrated_trial_correct`, `calibrated_scored_ticks`, `calibrated_trial_tie`,
`calibration_valid`, `calibration_error`.

A trial's prediction is the majority vote over ticks 1 to 9 (tick 0 is dropped). `*_scored_ticks` says how many
ticks voted, `*_tie` whether the vote was tied. The `calibrated_*` columns are blank in conditions 1, 3 and for
calibration-block trials. The file has no backend column: take the condition from `session_metadata.json`.

## 4. `session_metadata.json`

Top level (both programs): `session_id`, `utc_start`, `app_version`, `application_identifier`, `unity_version`,
`platform`, `device_model`, `window_frames` (5), `features_per_frame` (70), then `class_order`, `models`,
`columns`, `notes`. The study adds: `protocol_version` (`session-local-v2`), `randomization_seed`, `trial_count`
(14 or 21), `blocks`, `block_count`, `calibration_trials`, `evaluation_trials`, `calibration_passes`,
`evaluation_passes`, `evaluation_repeats`, `trial_counter_shown`, `ticks_per_trial`, `fast_rehearsal`,
`self_report`, and a `calibration` block.

`models` (every key is a string; **the API key is never written**):

| Key | Conditions | Value |
|---|---|---|
| `condition` | all | `sparsevr_nocal`, `sparsevr_cal`, `jev_nocal`, `jev_cal` |
| `condition_label` | all | e.g. `Condition 3: Jev, no calibration` |
| `face_backend` | all | `SparseVr` or `Jev` |
| `backend_id` | all | the model asset's name (SparseVR), or `jev:jev-1.13.0:p1` |
| `use_calibration` | all | `true` / `false` |
| `jev_model`, `jev_base_url`, `jev_timeout_s`, `jev_max_retries`, `jev_send_own_faces`, `jev_note` | 3, 4 | the Jev settings of the run |
| `face_model_asset`, `face_model_append_softmax`, `face_model_softmax_reason` | all | the face model assigned in the scene (also present in Jev runs; use `backend_id` for what actually answered) |
| `sound_model_*`, `logmel_model_*` | game | unused voice-test entries |

`calibration` (study only): `mode` (`raw`/`movement`), the calibration method (`feature_space`, `transform`,
`metric`, `estimator`, `movement_rule`, ...), `runtime_calibration_version`, and the outcome:
`calibration_valid`, `calibration_error`, `personal_centroid_built`, `movement_centroid_built`,
`calibration_ticks_per_class` (for example `Anger=9, ..., Surprise=9`; each class needs 9).

## 5. Library results file `VREmotionDetection/results/session_<utc>.csv`

Six columns, one row per `TryPredict` the game made (calibration-block ticks of the study are not saved):

| Column | Meaning |
|---|---|
| `time_utc` | `yyyy-MM-ddTHH:mm:ss.fffZ` |
| `seconds` | seconds since the first row |
| `emotion` | **what the game received: the calibrated emotion** when calibration decided, else the model's own |
| `confidence` | 4 decimals: SparseVR's probability for `emotion`, or **Jev's own answer** for it |
| `calibrated` | `yes` / `no` |
| `model_emotion` | the model's own answer; differs from `emotion` only when calibration changed it |

Failed rows read `<time>,<seconds>,no_tracking,,,` or `...,backend_error,,,` (nothing else is filled).
The file spans one app run, has no trial or condition column, and says nothing about who played.

## 6. Calibration file `<playerId>-<model>.json` (conditions 2 and 4, game)

`implementationVersion` (`runtime-calibration-v1`), `faceModel` (the backend id that made it: the model name or
`jev:...`), `savedUtc`, `classNames` (7), `scoreWidth` (7) and `scoreCentroids` (7 x 7 flattened, 49 numbers),
`movementWidth` (63) and `movementCentroids` (7 x 63 flattened, 441 numbers). It describes one person's face:
keep it on the device, never commit it. A file made with one model is refused by the other.

## 7. The four pipelines at a glance

| | 1 `sparsevr_nocal` | 2 `sparsevr_cal` | 3 `jev_nocal` | 4 `jev_cal` |
|---|---|---|---|---|
| `models.face_backend` / `backend_id` | `SparseVr` / model name | `SparseVr` / model name | `Jev` / `jev:jev-1.13.0:p1` | `Jev` / `jev:jev-1.13.0:p1` |
| `models.use_calibration` | `false` | `true` | `false` | `true` |
| `models.jev_*` keys | absent | absent | present | present |
| `score_*` | model probabilities (sum 1) | same | Jev's answers / their sum | same |
| `raw_*` | blank | blank | Jev's answers | Jev's answers |
| `backend_error` rows possible | rarely (model errors) | rarely | yes (network, key) | yes |
| Study `calibration_mode` | `raw` | `movement` | `raw` | `movement` |
| `calibrated_predicted_emotion` (study) | blank | filled (ticks 1-9 of the 14 test trials) | blank | filled |
| `calibrated_predicted_emotion` (game) | blank | filled on every tick (if the player calibrated; blank if skipped) | blank | filled on every tick (same) |
| Study rows in `predictions.csv` | 140 | 210 | 140 | 210 |
| Game rows in `predictions.csv` | 60 per round | 60 | 60 | 60 |
| Study `trials.csv` rows | 14 | 21 | 14 | 21 |
| Study `calibration.calibration_ticks_per_class` | all 0 | 9 each | all 0 | 9 each |
| Library results `calibrated` | `no` | `yes` | `no` | `yes` |
| Library results `confidence` | SparseVR probability | same | Jev's raw answer | same |
| Calibration file | none | `<id>-sparsevr.json` | none | `<id>-jev.json` |
| Jev requests per study session | 0 | 0 | 140 + 1 | 210 + 1 (the calibration block is included) |
| Jev requests per game round | 0 | 0 | 60 + 1 | 60 + 1 + about 63 to 70 for the calibration |

(The extra request is the start-up check.)

## 8. The game's score can be rebuilt from its own file

For every emotion, the game's score out of 10 is the number of ticks where
`calibrated_predicted_emotion` (or, when it is blank, `predicted_emotion`) equals `prompted_emotion`. This is
checked end to end in all four conditions. Two things to remember:

* If the player pressed **B** (skip) in a calibrated condition, the four columns are blank for the whole
  session: it ran uncalibrated, and `score_conditions.py` says so instead of reporting a comparison.
* The library results file (section 5) still records the calibrated emotion too (`emotion`, `calibrated` = `yes`),
  but it has no trial id, so use the game's `predictions.csv` for any analysis by emotion or by trial.

Files recorded before this change have only 382 columns and no calibrated decision; they can be scored for the
model but not for calibration.

## 9. Scoring a run

`python Archive/Analysis/score_conditions.py <sessions folder>` turns these files into accuracy and macro-F1
per condition (raw and calibrated, test ticks only, with the share of ticks answered). The label is
`prompted_emotion` and the prediction is `predicted_emotion`; see `Archive/Analysis/README.md`,
"Comparing the four conditions".

## 10. Examples (test data)

One study tick, condition 4 (`jev_cal`), frames omitted:

```
timestamp_utc=2026-10-17T05:20:17.810Z  session_id=20261017_052016_630ff0  trial_id=7  block_role=evaluation
repeat_index=1  prompted_emotion=Neutral  prediction_index=3  predicted_emotion=Sadness  correct=0  status=ok
generation=1624  score_Sadness=0.7798  (others 0.0367)
calibration_mode=movement  movement_calibration_enabled=true  raw_predicted_emotion=Sadness
calibrated_predicted_emotion=Disgust  calibrated_winner_similarity=1.4949  calibrated_runner_up_similarity=1.4460
calibrated_margin=0.0489  calibration_valid=true  calibration_error=
backend_id=jev:jev-1.13.0:p1  error_code=  raw_Sadness=0.85  (other raw_* 0.04)
```

The same tick in condition 1 (`sparsevr_nocal`): `calibration_mode=raw`, `calibrated_*` blank, `backend_id` is the
model name and every `raw_*` is blank.

Library results file, condition 4: `2026-10-17T05:20:17.781Z,0.000,Surprise,0.0400,yes,Fear` (the calibrated
emotion is Surprise with Jev's own answer 0.04 for it; the model alone said Fear).

Study `trials.csv`, condition 1 (first rows):
`20261017_051930_16ab9c,0,evaluation,1,Happiness,raw,,Happiness,1,9,false,,,0,,true,`
