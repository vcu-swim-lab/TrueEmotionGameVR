# What a game writes: the four conditions and their files

Every game runs one condition, chosen by `ConditionSetup.Selected` in `ConditionSetup.cs`:

| # | Condition id | Model | Calibration |
|---|---|---|---|
| 1 | `sparsevr_nocal` | SparseVR (on the headset) | off |
| 2 | `sparsevr_cal` | SparseVR | on |
| 3 | `jev_nocal` | Jev (Typesafe's cloud model) | off |
| 4 | `jev_cal` | Jev | on |

The files have the **same columns in all four conditions**. What changes is which columns are filled in
(section 8). Recordings before 2026-10-19 (the study flow, `session-local-v1` to `v3`) used the same
`predictions.csv` columns but a different trial structure; see `CALIBRATION_STUDY_PROTOCOL.md`.

## 1. Where the files are

On the headset, under `/sdcard/Android/data/com.DefaultCompany.EmotionGatheringTool/files/`
(in the Editor: `%USERPROFILE%\AppData\LocalLow\DefaultCompany\Emotion Gathering Tool\`):

| File | Written by | Conditions |
|---|---|---|
| `sessions/session_<utc>_<6 hex>/predictions.csv` | `SessionLogger` | all |
| `sessions/session_<utc>_<6 hex>/trials.csv` | `SessionLogger` | all |
| `sessions/session_<utc>_<6 hex>/session_metadata.json` | `SessionLogger` | all |
| `sessions/session_<utc>_<6 hex>/face_tracking.csv` | `FaceTrackingRecorder` | all, when **Save Face Tracking** is on |
| `VREmotionDetection/results/session_<utc>.csv` | the library (`NewFaceAuModel`, **Save Results**) | all |

* A session folder exists only after the condition passed its start-up check. A game that stops with
  "Cannot start" (no Jev key, key rejected, no connection) writes no session folder.
* The game does not save a calibration file: it calibrates again in every session that uses it.
* Copy the session folders with the Unity menu **VR Emotion > Get Sessions From Headset**
  (to `HeadsetResults/sessions/`), and the library files with **VR Emotion > Get Results From Headset**.
* All times are UTC.

## 2. `predictions.csv` (one row per second of acting)

**394 columns.** 70 rows without calibration (Part 2: 7 trials × 10 seconds), 140 with it (Part 1 and
Part 2: 14 trials × 10). Columns in this order:

| # | Columns | Meaning |
|---|---|---|
| 1 | `timestamp_utc` | `yyyy-MM-ddTHH:mm:ss.fffZ` when the row was written |
| 2 | `session_id` | `<yyyyMMdd_HHmmss>_<6 hex>`; same on every row of the folder |
| 3 | `trial_id` | 0-based trial number across the whole game |
| 4 | `block_role` | `calibration` (Part 1) or `evaluation` (Part 2) |
| 5 | `repeat_index` | 1-based presentation of that emotion in its part (always 1) |
| 6 | `prompted_emotion` | the emotion the player was asked to act |
| 7 | `prediction_index` | 0-based second inside the trial (0 to 9) |
| 8 | `predicted_emotion` | the **model's own** answer (argmax of `score_*`); blank when `status` is not `ok` |
| 9 | `correct` | `1` if `predicted_emotion` equals `prompted_emotion`, else `0` (also `0` on a failed second) |
| 10 | `status` | `ok`, `no_tracking` (no valid face window) or `backend_error` (the model failed, see `error_code`) |
| 11 | `generation` | count of valid face frames so far; a small jump means tracking dropped out |
| 12-18 | `score_Anger` ... `score_Surprise` | seven scores in this order: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise |
| 19-23 | `frame_0_valid` ... `frame_4_valid` | always `1` for logged windows |
| 24-373 | `frame_<f>_au_<a>` | the 5 × 70 face-tracking values the model saw (frame 0 oldest, 63 face + 7 tongue values) |
| 374-385 | calibration columns | see below |
| 386-394 | `backend_id`, `error_code`, `raw_Anger` ... `raw_Surprise` | which model answered; why it failed; Jev's own answers |

The calibration columns:

| Column | Value |
|---|---|
| `calibration_mode` | `raw` (conditions 1, 3) or `movement` (2, 4); fixed at session start |
| `personal_/population_calibration_enabled`, `hybrid_alpha` | kept for old readers: always `false`, `false`, blank |
| `movement_calibration_enabled` | `true` in conditions 2, 4 |
| `raw_predicted_emotion` | the model's argmax (equal to `predicted_emotion`) |
| `calibrated_predicted_emotion` | the calibration's decision; **blank** in conditions 1, 3, in Part 1, on second 0 and on a failed second |
| `calibrated_winner_similarity`, `calibrated_runner_up_similarity`, `calibrated_margin` | the calibration's scores (0 to 2) for the winner and runner-up and their difference; blank when there is no decision |
| `calibration_valid`, `calibration_error` | `true`/`false` and the reason; blank on Part 1 rows (known only once Part 1 ends). `true` for raw conditions too: there is nothing to invalidate |

`status` and `error_code`:

| `status` | `error_code` | When |
|---|---|---|
| `ok` | blank | a prediction was made |
| `no_tracking` | blank or `NoTracking` | no valid face window in time |
| `backend_error` | `NotConfigured`, `NotReady`, `Busy`, `InvalidInput`, `Unauthorized`, `InvalidRequest`, `RateLimited`, `Timeout`, `Unreachable`, `ServerError`, `InvalidResponse`, `Internal`, `Canceled` | the model could not answer. For Jev: `Unauthorized` = key, `Unreachable`/`Timeout` = network, `RateLimited` = too many requests, `InvalidResponse` = Jev's answer was unusable, `Canceled` = the model was switched or the app paused mid-request |

On a failed row `predicted_emotion`, the scores, the frames and `raw_*` are blank; `backend_id` is still filled.

## 3. `trials.csv` (one row per trial)

17 columns: `session_id`, `trial_id`, `block_role`, `repeat_index`, `prompted_emotion`, `calibration_mode`,
`hybrid_alpha` (blank), `raw_trial_prediction`, `raw_trial_correct`, `raw_scored_ticks`, `raw_trial_tie`,
`calibrated_trial_prediction`, `calibrated_trial_correct`, `calibrated_scored_ticks`, `calibrated_trial_tie`,
`calibration_valid`, `calibration_error`. 7 rows without calibration, 14 with it.

A trial's prediction is the majority vote over seconds 1 to 9 (second 0 is dropped). The result screen shows
`calibrated_trial_prediction` when it is filled and `raw_trial_prediction` otherwise. `*_scored_ticks` says
how many seconds voted, `*_tie` whether the vote was tied. The `calibrated_*` columns are blank in conditions
1, 3 and for Part 1 trials. The file has no backend column: take the condition from `session_metadata.json`.

## 4. `session_metadata.json`

Top level: `session_id`, `utc_start`, `app_version`, `application_identifier`, `unity_version`, `platform`,
`device_model`, `window_frames` (5), `features_per_frame` (70); how the game was run: `protocol_version`
(`demo-v2`), `randomization_seed`, `trial_count` (7 or 14), `blocks`, `block_count`, `calibration_trials`,
`evaluation_trials`, `calibration_passes`, `evaluation_passes`, `evaluation_repeats`, `trial_counter_shown`,
`ticks_per_trial`, `fast_rehearsal`, `self_report`, `demo`, `face_tracking_saved`; then `class_order`,
`models`, `columns`, `notes` and a `calibration` block.

`models` (every key is a string; **the API key is never written**):

| Key | Conditions | Value |
|---|---|---|
| `condition` | all | `sparsevr_nocal`, `sparsevr_cal`, `jev_nocal`, `jev_cal` |
| `condition_label` | all | e.g. `Condition 3: Jev, no calibration` |
| `face_backend` | all | `SparseVr` or `Jev` |
| `backend_id` | all | the model asset's name (SparseVR), or `jev:jev-1.13.0:p1` |
| `use_calibration` | all | `true` / `false` |
| `jev_model`, `jev_base_url`, `jev_timeout_s`, `jev_max_retries`, `jev_send_own_faces`, `jev_note` | 3, 4 | the Jev settings of the game |
| `face_model_asset`, `face_model_append_softmax`, `face_model_softmax_reason` | all | the face model assigned in the scene (also present in Jev games; use `backend_id` for what actually answered) |

`calibration`: `mode` (`raw`/`movement`), the calibration method (`feature_space`, `transform`, `metric`,
`estimator`, `movement_rule`, ...), `runtime_calibration_version`, and the outcome: `calibration_valid`,
`calibration_error`, `personal_centroid_built`, `movement_centroid_built`, `calibration_ticks_per_class`
(for example `Anger=9, ..., Surprise=9`; each emotion needs 9).

## 5. `face_tracking.csv` (with Save Face Tracking; one row per frame)

About 72 rows a second, from the first get-ready screen to the end screen. 77 columns:

| Column | Meaning |
|---|---|
| `timestamp_utc` | the UTC time of the frame, in the same format as `predictions.csv` |
| `time_s` | seconds since the first get-ready screen (time 0) |
| `frame` | Unity's frame number |
| `screen` | `get_ready`, `act`, `result` or `part` |
| `trial_id`, `emotion` | the trial the screen belongs to (blank on a part screen) |
| `valid` | `true` when face tracking held this frame |
| `BrowLowererL` ... `TongueRetreat` | the 70 face-tracking values, 0 to 1, in the order the model reads them; blank when `valid` is `false` |

Used for the avatar video (README, *Video with avatar*).

## 6. Library results file `VREmotionDetection/results/session_<utc>.csv`

Six columns, one row per second of Part 2 (Part 1 seconds are not saved):

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

## 7. Jev requests per game

| | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Requests | 0 | 0 | 70 + 1 | 140 + 1 |

(The extra request is the start-up check. Part 1 counts too: calibration is collected from Jev's scores.)

## 8. The four conditions at a glance

| | 1 `sparsevr_nocal` | 2 `sparsevr_cal` | 3 `jev_nocal` | 4 `jev_cal` |
|---|---|---|---|---|
| `models.face_backend` / `backend_id` | `SparseVr` / model name | `SparseVr` / model name | `Jev` / `jev:jev-1.13.0:p1` | `Jev` / `jev:jev-1.13.0:p1` |
| `models.use_calibration` | `false` | `true` | `false` | `true` |
| `models.jev_*` keys | absent | absent | present | present |
| `score_*` | model probabilities (sum 1) | same | Jev's answers / their sum | same |
| `raw_*` | blank | blank | Jev's answers | Jev's answers |
| `backend_error` rows possible | rarely (model errors) | rarely | yes (network, key) | yes |
| `calibration_mode` | `raw` | `movement` | `raw` | `movement` |
| `calibrated_predicted_emotion` | blank | filled (seconds 1-9 of Part 2) | blank | filled |
| Rows in `predictions.csv` / `trials.csv` | 70 / 7 | 140 / 14 | 70 / 7 | 140 / 14 |
| `calibration.calibration_ticks_per_class` | all 0 | 9 each | all 0 | 9 each |
| Library results `calibrated` | `no` | `yes` | `no` | `yes` |
| Library results `confidence` | SparseVR probability | same | Jev's raw answer | same |

## 9. Scoring a game

`python Archive/Analysis/score_conditions.py <sessions folder>` turns these files into accuracy and macro-F1
per condition (raw and calibrated, Part 2 seconds only, with the share of seconds answered). The label is
`prompted_emotion` and the prediction is `predicted_emotion` (or `calibrated_predicted_emotion`); see
`Archive/Analysis/README.md`, "Comparing the four conditions".

## 10. Example (test data)

One Part 2 second, condition 4 (`jev_cal`), frames omitted:

```
timestamp_utc=2026-10-17T05:20:17.810Z  session_id=20261017_052016_630ff0  trial_id=7  block_role=evaluation
repeat_index=1  prompted_emotion=Neutral  prediction_index=3  predicted_emotion=Sadness  correct=0  status=ok
generation=1624  score_Sadness=0.7798  (others 0.0367)
calibration_mode=movement  movement_calibration_enabled=true  raw_predicted_emotion=Sadness
calibrated_predicted_emotion=Disgust  calibrated_winner_similarity=1.4949  calibrated_runner_up_similarity=1.4460
calibrated_margin=0.0489  calibration_valid=true  calibration_error=
backend_id=jev:jev-1.13.0:p1  error_code=  raw_Sadness=0.85  (other raw_* 0.04)
```

The same second in condition 1 (`sparsevr_nocal`): `calibration_mode=raw`, `calibrated_*` blank,
`backend_id` is the model name and every `raw_*` is blank.
