# Session-Local Calibration (`session-local-v1`): First Experiment

**Protocol:** `session-local-v1`
**Session reported:** `20260916_201632_85c128` (Oculus Quest Pro)
**Audience:** general machine-learning background, no prior knowledge of this project

For the full protocol specification, the frozen analysis rules, and the earlier results that
led here, see [`CALIBRATION_STUDY_PROTOCOL.md`](CALIBRATION_STUDY_PROTOCOL.md).

---

## 1. Purpose

Our emotion-recognition model works well for some people and poorly for others. The same
expression that reads correctly on one face is misread on another. Rather than retrain the
model, we ask whether a short calibration at the start of a session lets us reinterpret the
model's own output for that particular person, for the rest of that sitting.

The model itself is never retrained, fine-tuned, or modified. Only the interpretation of its
output changes.

## 2. Protocol

One headset sitting, 21 trials, no score or feedback shown at any point.

- **Calibration block (K): 7 trials.** Each of Anger, Disgust, Fear, Happiness, Neutral,
  Sadness and Surprise is performed once. Neutral is prompted as "let your face relax
  naturally" rather than as an expression to act.
- **Evaluation block (D): 14 new trials.** The same seven states, each performed twice.

K and D trials are entirely separate performances. Evaluation trials are never used to create
or update the calibration. The headset stays on between the two blocks.

Each trial lasts about 10 seconds and yields 10 predictions ("ticks"). Tick 0 is discarded as a
transition sample, caught while the participant is still forming the expression; ticks 1–9 are
used.

## 3. Model input and output

For each prediction the headset supplies a window of 5 consecutive frames, each holding 70
facial action-unit values — a `[5, 70]` window. The existing model consumes all five frames
together as one input; we do not make five separate predictions and average them.

The model returns a single seven-value score vector, one score per state:
`[Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise]`.

## 4. How calibration works

For each calibration trial:

1. Ticks 1–9 produce nine separate seven-value score vectors.
2. Each vector is transformed with `log(clip(score, epsilon))`.
3. The nine transformed vectors are averaged column by column.
4. The result is one seven-value **centroid** for that state.

Repeating this for all seven states yields seven personal centroids.

For example, when this particular person performs Disgust, the model might typically respond
with a moderate Disgust score, a substantial Neutral score, and a small amount of Anger. The
centroid records that whole pattern. A centroid therefore represents: *how the existing model
typically responds when this specific person performs this state.*

Note that the centroid is built from the model's seven output scores, not from the raw
action-unit frames.

## 5. How evaluation works

Each evaluation window goes through the same unchanged model and receives the same log
transformation. The transformed vector is then compared to all seven personal centroids using
cosine similarity, and the most similar centroid gives the **calibrated prediction**.

The baseline is the model's ordinary raw argmax — its highest-scoring class. Both predictions
are scored on exactly the same D trials.

The distinction matters because the calibrated decision uses the whole score pattern, not just
the winner. The model may output `Surprise` as its top class for a genuine Disgust expression;
if that full seven-score pattern nonetheless most closely resembles this person's Disgust
centroid, the calibrated result is `Disgust`.

## 6. Results

First real session: **210 of 210 ticks tracked**, replay fidelity agreement **1.000** (the
offline analysis reproduces the headset's scores exactly).

| Metric (on D trials) | Raw model | Calibrated |
|---|---|---|
| Tick accuracy | 51.6% | **67.5%** |
| Trial-majority accuracy | 64.3% (9/14) | **71.4% (10/14)** |
| Macro-F1 | 0.595 | **0.681** |

Per state, in ticks: Anger improved from 5/18 to 15/18, Disgust from 8/18 to 14/18, Surprise
from 7/18 to 11/18. Fear did not improve. Happiness decreased slightly, from 12/18 to 10/18.

## 7. Interpretation and limitations

This is the first **positive trial-disjoint** result for session-local calibration. It matters
because it removes an earlier flaw: previous measurements calibrated on the early ticks of an
expression and tested on the late ticks of that *same continuous performance*, which inflated
the apparent gain. Here calibration and testing come from separate trials.

The result supports continuing the experiment. It does not show that the method generalizes.

- This is **one participant**.
- At the trial level the gain is **one additional correct trial** out of 14.
- One class (Happiness) got worse.
- Fear remains unfixed.

Before the adapter goes into the runtime library, more participants must complete the identical
protocol — especially a previously low-performing user, since that is the case the method is
meant to address.

---

**Calibration**

```
5×70 window → ONNX → 7 scores → log transform → average 9 ticks → personal centroid
```

**Evaluation**

```
5×70 window → ONNX → 7 scores → log transform → cosine similarity to 7 centroids → calibrated emotion
```
