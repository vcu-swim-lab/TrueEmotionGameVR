# VR Emotion Detection — Validation Report

**Project:** TrueEmotionGameVR / `com.vcu-swim-lab.vr-emotion-detection`
**Scope:** runtime data-path defects, their validation, and the first on-device measurements
**Status:** runtime fixes and instrumentation applied; automated tests passed; first logged
headset session analysed; Fear and Anger remain open diagnostic targets

---

## 1. Executive summary

Real-time emotion classification in `TrueEmotionGameVR` was producing usable results for three of
seven classes and near-zero results for three others. Investigation identified a defect in the
runtime data path that destroyed the temporal structure of every model input: a five-frame sliding
window was collapsing so that all five timesteps held a copy of the newest frame. The classifier is
a dilated causal temporal convolutional network whose entire design depends on motion across the
window, so it was being asked to read movement from a still image.

The defect was corrected and the correction was validated by automated test rather than by
inspection: the regression test fails against the original algorithm and passes against the fixed
one. On-device measurement afterwards showed one large, statistically significant improvement
(Disgust, 0/10 → 9/10, Fisher's exact p = 0.0001). The aggregate change across all classes
(26/60 → 32/60) is **not** statistically significant at this sample size, and two classes — Fear and
Anger — remain at or near zero.

Two further defects were found and corrected during the same review:

- A model-preparation helper appended a softmax to every classifier it loaded. Three of the four
  classifiers already ended in a softmax, so their reported confidence values were not
  probabilities. The fourth genuinely required one, so the correct fix was to make the choice
  explicit per model rather than to remove it.
- Face-tracking dropouts were being recorded as real data. When tracking was unavailable the
  device produced an all-zero frame that entered the model as though it were a genuine neutral
  expression.

Finally, per-prediction CSV logging was added. Previous sessions produced only six integers and
were not reanalysable. The first logged session now answers the priority question — what the model
predicts *instead* during Fear and Anger trials — in section 5.4. It also preserves the exact
five-frame input window for future offline replay.

**The action-unit analysis in section 8 changes the working assumption of the project.** Across the
six prompted classes, how far the face actually moved predicts the outcome without exception: every
emotion whose peak action unit exceeded 0.40 scored 8–9/10, and both emotions below 0.25 scored
0–2/10. Fear's recorded profile is Surprise's profile at roughly half amplitude, with both of the
units that distinguish the two under FACS entirely absent; Anger activates only the brow, and does
so less strongly than Disgust does. On these trials the classifier is describing its input
accurately. The remaining failure therefore lies in what reaches the model rather than in the model
itself — which rules out probability calibration as a remedy and bounds what an activity gate can
achieve. One confirmatory experiment, re-acting the two classes while deliberately driving the
silent units, separates a protocol limitation from a hardware one and is the recommended next step.

---

## 2. The RingBuffer aliasing defect

### 2.1 Mechanism

The sliding window was held in a `RingBuffer<float[]>` — a `float[5][]`, five references to five
`float[70]` arrays. Each poll shifted the window left:

```csharp
// original
for (int i = 0; i < ring.buffer.Length - 1; ++i)
{
    ring.buffer[i] = ring.buffer[i + 1];
}

var tmp = ring.buffer[ring.buffer.Length - 1];
Array.Copy(data, tmp, data.Length);
```

In C# an array of arrays stores **references**. `ring.buffer[i] = ring.buffer[i + 1]` therefore does
not move 70 floats from one slot to another; it makes two slots point at the same array. Repeating
that in a loop makes successive slots converge on a single object, and the subsequent `Array.Copy`
writes the new frame through every one of those aliases at once.

Beginning from five distinct arrays `A B C D E`:

| Poll | Slots after the shift loop | Window contents after the copy | Distinct arrays |
|---|---|---|---|
| 1 | `B C D E E` | `B C D f1 f1` | 4 |
| 2 | `C D E E E` | `C D f2 f2 f2` | 3 |
| 3 | `D E E E E` | `D f3 f3 f3 f3` | 2 |
| 4 | `E E E E E` | `f4 f4 f4 f4 f4` | **1** |

From the fourth poll onward all five timesteps are the same object and therefore always hold the
newest frame.

### 2.2 Why every inference was affected

`RingBuffer.Clear()` resets a fill counter only; the arrays are allocated once in the constructor
and are already aliased by the time it is called. `PredictRaw()` called `Clear()` and then waited for
five polls before running inference — one more than the four needed for total collapse. Every
inference therefore ran against a fully collapsed window. This was not an intermittent fault.

Measured against the shipped ONNX graph with simulated onset windows, feeding five copies of the
newest frame instead of five real frames changed the predicted label on **48% of windows**.

### 2.3 The fix

Preserve the array being displaced and recycle it to the tail, so the five arrays rotate rather than
collapse:

```csharp
var tmp = ring.buffer[0];                       // keep the oldest array
for (int i = 0; i < ring.buffer.Length - 1; ++i)
{
    ring.buffer[i] = ring.buffer[i + 1];
}
ring.buffer[ring.buffer.Length - 1] = tmp;      // recycle it to the end
Array.Copy(data, tmp, frameSize);
```

Two added lines. No allocation per push, no change to any call site.

The rotation was additionally extracted from the event subscription into a public
`RingBuffer.Push(float[])` extension. Previously the logic was reachable only through a device
callback, and neither `Device` nor the buffer internals could be constructed from outside the
assembly, which is why the defect had no test. `Listen` is now a single delegating line. `Push` also
validates frame length and throws `ArgumentException` on a mismatch.

---

## 3. Automated validation

### 3.1 Setup

EditMode tests under `Tests/Editor/`, assembly `VREmotionDetection.Editor.Tests`, executed by the
Unity Test Framework against Unity 6000.0.33f1. The tests exercise the real `Push`, the real
`ToTensor()`, and the real tensor readback — not a reimplementation.

The decisive test pushes seven distinguishable frames into a five-frame buffer and asserts that the
resulting tensor contains frames 3, 4, 5, 6, 7 in arrival order.

### 3.2 Result against the original algorithm

With the original rotation restored, the test failed:

```
Expected and actual are both <System.Single[10]>
  Values differ at index [0]
  Expected: 3.0f
  But was:  7.0f
```

Index 0 holding the value of frame 7 rather than frame 3 is the collapse signature: every slot
carrying the newest frame. Run recorded in `VREmotionDetection-OldAlgorithm-TestResults.xml`
(`total="1" passed="0" failed="1"`).

### 3.3 Result against the fixed algorithm

```
result="Passed"  total="2"  passed="2"  failed="0"
```

Recorded in `VREmotionDetection-Final-TestResults.xml`. This is the verified state of the suite at
the time the headset measurements in section 5 were taken.

### 3.4 Extended suite

The suite was subsequently extended to cover the other defects described in this report. Full run,
recorded in `VREmotionDetection-Phase2-TestResults.xml`:

```
result="Passed"  total="14"  passed="14"  failed="0"  skipped="0"
```

| Test | Covers |
|---|---|
| `Push_KeepsLastFiveFramesInArrivalOrder` | window ordering (§2) |
| `Push_KeepsEveryFrameBackedByADistinctArray` | reference distinctness (§2) |
| `LegacyShift_CollapsesEverySlotOntoOneArray` | documents the original defect (§2) |
| `Push_RejectsFramesWithTheWrongLength` | frame-size validation |
| `Snapshot_IsIndependentOfLaterPushes` | window integrity for logging (§7.2) |
| `Snapshot_ToTensor_MatchesTheLiveWindow` | snapshot equals the live window |
| `Generation_IsMonotonicAcrossClear` | dropout signal survives `Clear()` (§7.3) |
| `PollAndDispatch_ForwardsValidReadings` | valid frames still reach the buffer (§7.5) |
| `PollAndDispatch_DropsInvalidReadings` | invalid frames do not (§7.5) |
| `PollAndDispatch_NeverStoresAnAllZeroTrackingDropout` | dropouts never become data (§7.5) |
| `PollAndDispatch_ReturnsTheDeviceInterval` | polling cadence preserved |
| `FaceModel_WithoutAppendedSoftmax_OutputsProbabilities` | face softmax decision (§6) |
| `SoundModel_EmitsLogits_SoItNeedsTheAppendedSoftmax` | audio emits logits (§6) |
| `SoundModel_WithAppendedSoftmax_OutputsProbabilities` | audio softmax decision (§6) |

The three model-preparation tests load the shipped ONNX assets and run real inference on the CPU
backend, so they verify the softmax finding against the actual graphs rather than against a
recorded assumption.

All project assemblies — including the game's `Assembly-CSharp`, which holds the logging code —
compiled with no errors and no warnings attributable to these changes.

---

## 4. Local test setup

The game resolves the library from a GitHub URL and compiles a read-only clone in
`Library/PackageCache/`. Local edits to the library working tree were therefore invisible to the
game, which made every iteration appear to require a push.

This was resolved with a Windows directory junction that makes the library an *embedded* package:

```
TrueEmotionGameVR\Packages\com.vcu-swim-lab.vr-emotion-detection
    --> C:\Users\alime\vcu-swim-lab\VREmotionDetection
```

Unity treats any folder under `Packages/` as an embedded package, and embedded resolution takes
precedence over the `manifest.json` dependency. `manifest.json` therefore needs no modification. The
Package Manager window shows the package badged **Custom**, and the previously cached GitHub clone
is removed automatically.

**No push to GitHub was required at any point** in this work.

The junction is excluded from version control via `.git/info/exclude`, which is a local,
untracked file — no tracked project file was modified to achieve this.

> **Warning — `packages-lock.json`.** Unity rewrites this file automatically when it resolves the
> embedded package, changing the entry from `"source": "git"` to `"source": "embedded"`. That
> modification is a side effect of the local-only setup and **must not be committed**: doing so
> would redirect every other clone of the project to a path that exists only on this machine.
> Either revert it before committing, or mark it with
> `git update-index --skip-worktree Packages/packages-lock.json`.

---

## 5. Headset experiment results

Single subject, acted expressions, in-headset, ten trials per emotion. One point is scored per
one-second tick whose argmax class equals the prompted class.

| Emotion | Original pipeline | Fixed five-frame pipeline | Change |
|---|---|---|---|
| Surprise | 9/10 | 8/10 | −1 |
| Fear | 0/10 | 0/10 | 0 |
| Sadness | 9/10 | 10/10 | +1 |
| Anger | 1/10 | 0/10 | −1 |
| Happiness | 7/10 | 5/10 | −2 |
| Disgust | 0/10 | 9/10 | **+9** |
| **Total** | **26/60** | **32/60** | **+6** |

### 5.1 Statistical assessment

Fisher's exact test, two-tailed, per class:

| Emotion | p | Interpretation |
|---|---|---|
| **Disgust** | **0.0001** | significant |
| Happiness | 0.65 | not significant |
| Surprise | 1.00 | not significant |
| Sadness | 1.00 | not significant |
| Anger | 1.00 | not significant |
| **Total (26/60 vs 32/60)** | **0.36** | **not significant** |

### 5.2 Conclusions to draw, and not to draw

- **Disgust improving from 0/10 to 9/10 is the strong observed change**, and it is consistent with
  the mechanism: disgust is expressed through rapid movement (nose wrinkle, upper-lip raise), which
  is precisely the information the collapsed window destroyed.
- **The aggregate change from 26/60 to 32/60 is not statistically significant at this sample size.**
  It should not be reported as an overall accuracy improvement.
- **The apparent declines in Happiness (−2) and Anger (−1) are not significant** and should not be
  treated as regressions caused by the fix.
- **Fear remains 0/20 across both sessions.** Two independent sessions at zero is not attributable
  to sampling variation, and this makes Fear the priority class for diagnosis. It is also the
  weakest class in the teacher's offline evaluation (f1 = 0.35).
- **The fix did not solve all emotion classes**, and no claim to that effect is supported by these
  data.

### 5.3 Sample size

Ten binary trials per class is a weak instrument: it cannot separate a real effect from noise
unless the effect is very large, as Disgust's was. The logging described in section 7 replaces six
integers per session with several hundred labelled windows, which is the main reason to record the
next session before attempting any further change.

### 5.4 First logged session: confusion matrix and score analysis

**Session:** `20260912_213907_eec228`  
**Device:** Oculus Quest Pro  
**Runtime:** Android / Unity 6000.0.33f1  
**Face model:** `sparseVR_teacher_model` with no appended softmax  
**Rows:** 60 predictions, 10 per prompted non-Neutral class  
**Tracking status:** 60 `ok`, 0 `no_tracking`

This is a later session than the two sessions in section 5. It scored **37/60**, rather than 32/60.
It is not a paired A/B comparison, so the difference must not be attributed to one particular code
change. Its value is the detailed error pattern, not an additional claim of overall accuracy.

| Prompted class | Anger | Disgust | Fear | Happiness | Neutral | Sadness | Surprise | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Anger | 2 | 0 | 0 | 0 | 8 | 0 | 0 | 2/10 |
| Disgust | 0 | 9 | 0 | 0 | 1 | 0 | 0 | 9/10 |
| Fear | 0 | 0 | 0 | 0 | 4 | 0 | 6 | 0/10 |
| Happiness | 0 | 1 | 0 | 8 | 1 | 0 | 0 | 8/10 |
| Sadness | 0 | 0 | 0 | 0 | 1 | 9 | 0 | 9/10 |
| Surprise | 0 | 0 | 0 | 0 | 1 | 0 | 9 | 9/10 |

The recorded output distribution was: Neutral 16, Surprise 15, Disgust 10, Sadness 9,
Happiness 8, Anger 2, and Fear 0. Neutral is never prompted in this protocol, so every Neutral
prediction is an error.

#### Fear

Fear is not merely low-confidence noise. It is never the winning class, but it is usually the
second-ranked output. Across the ten Fear ticks, the mean class scores were:

| Anger | Disgust | Fear | Happiness | Neutral | Sadness | Surprise |
|---:|---:|---:|---:|---:|---:|---:|
| 0.031 | 0.017 | **0.140** | 0.016 | 0.270 | 0.050 | **0.476** |

Six Fear ticks selected Surprise and four selected Neutral. During the middle of the Fear trial,
the model strongly preferred Surprise (approximately 0.68–0.83) while Fear remained second
(approximately 0.13–0.20). At the beginning and end of the trial it preferred Neutral. The model
therefore detects a non-neutral facial event but maps it to Surprise rather than Fear.

This is not safe to correct by remapping Surprise to Fear: genuine Surprise was 9/10 correct in the
same session. The next high-value investigation is the training label mapping, especially
EmojiHeroVR, followed by offline replay and class-specific analysis using additional recorded
sessions.

#### Anger

The mean Anger-trial scores were Neutral 0.495 and Anger 0.222. The model selected Anger on two
middle ticks, then returned to Neutral. This is consistent with a short, recognisable onset or
peak surrounded by windows that the model treats as resting face. It is evidence to investigate
an activity/selection rule with real recordings; it is not yet evidence for a threshold value.

#### Probability validation

In all 60 logged rows, the seven recorded face scores summed to 1.0 within floating-point error
(minimum 0.99999984, maximum 1.00000012). This validates that the face output is now recorded on a
probability scale after removal of the accidental second softmax.

---

## 6. The double-softmax finding

`Utils.PrepareModel` loaded a model asset and appended a `Softmax` to its output unconditionally.
All five shipped ONNX graphs were re-examined, structurally and by running them:

| Model | Final node | Output sums to 1 | Minimum value | Interpretation |
|---|---|---|---|---|
| `sparseVR_teacher_model.onnx` | `Softmax` | 1.000000 | +0.0000 | already probabilities |
| `act_trial_92.onnx` | `Softmax` | 1.000000 | +0.0000 | already probabilities |
| `natural_trial_96.onnx` | `Softmax` | 1.000000 | +0.0000 | already probabilities |
| `sound_model.onnx` | `Gemm` | 0.385098 | −0.0426 | **raw logits** |
| `logmel_spectrogram.onnx` | `Div` | n/a | n/a | preprocessing transform |

The three face classifiers were being softmaxed twice. The audio classifier was not: it emits raw
logits, including negative values, and genuinely requires the appended softmax.

Removing the softmax globally — the obvious reading of "the models already have one" — would have
silently broken audio classification. The correct fix is therefore a required per-call decision:

```csharp
internal static Model PrepareModel(ModelAsset asset, bool appendSoftmax)
```

The parameter has no default, so a future model cannot be added without the author making the
choice deliberately. Face call sites pass `false`; the voice call site passes `true`;
`logmel_spectrogram.onnx` continues to load through `ModelLoader.Load` directly and is unaffected.

### 6.1 Why this matters

Softmax is monotone, so applying it twice does not change which class wins. **No score reported in
section 5 is affected by this defect**, and both the 26/60 and 32/60 figures stand.

What it destroys is the magnitude. Applying a second softmax to the teacher's output compresses a
genuine peak probability of 0.8426 to 0.2736, and collapses the seven-class spread from roughly
0.84 to roughly 0.16. The consequences are entirely downstream of measurement:

- Any confidence threshold tuned against those values would be tuned against a distorted scale.
- Any probability calibration fitted on logged confidences would be fitted on numbers that are not
  probabilities.
- Comparing in-game confidence against notebook confidence would show a discrepancy with no cause
  in the model.

This is why the correction had to land together with logging rather than after it: logging
distorted confidences would have produced a dataset that looks usable and is not.

---

## 7. Logging design

### 7.1 Why all five frames are recorded

The face model consumes a `[1, 5, 70]` tensor. Recording only the newest 70-value frame once per
prediction would capture one fifth of the input and none of the temporal structure, making offline
replay impossible — and temporal structure is precisely what this investigation showed to be
decisive.

Each CSV row therefore carries the complete window, `frame_0_au_0` through `frame_4_au_69`, frame 0
oldest. With the full window, a recorded session can be replayed through the same ONNX graph in
Python and the in-game predictions reproduced exactly. That turns every future change — the
activity gate, tick aggregation, calibration — into an offline experiment rather than another
headset session.

### 7.2 Window integrity

The recorded window is an independent deep copy taken at the moment the input tensor is built, and
the tensor is built from that copy. This is a correctness requirement, not a convenience: the ring
buffer recycles its arrays on every poll, so a reference retained across even a single poll would
describe a window the model never saw. The `RingBuffer.Snapshot()` extension exists for this
purpose, and a test asserts that a snapshot is unaffected by subsequent pushes.

### 7.3 Row contents

One row per prediction tick: UTC timestamp, session id, trial id, prompted emotion, prediction
index within the trial, predicted emotion, correctness flag, status, ring-buffer generation, all
seven class scores in `Emotion` enum order (Anger, Disgust, Fear, Happiness, Neutral, Sadness,
Surprise), per-frame validity flags, and the 350 action-unit values of the window.

Ticks where no window of valid tracking frames could be collected are written with
`status = no_tracking` rather than omitted, so tracking loss is visible in the data instead of
appearing as missing rows.

`generation` is a monotonic count of frames actually accepted into the buffer. It identifies the
specific input window used by a prediction and confirms that the window contains accepted frames.
It is **not** a direct counter of short tracking dropouts: invalid polls do not increment it, and a
prediction waits until it has collected five valid frames. The `frame_<f>_valid` fields are
therefore always `1` for a successful logged window by design.

`status = no_tracking` identifies the important failure case in which five valid frames could not
be collected within the three-second timeout. To quantify shorter tracking dropouts, add an
explicit invalid-poll counter or a separate poll-level tracking log in a later change. Do not infer
an exact dropout count from `generation` alone.

### 7.4 Output location and format

Files are written beneath `Application.persistentDataPath` into a timestamped per-session folder,
so no recording can overwrite another. Each session folder holds `predictions.csv` and a
`session_metadata.json` sidecar recording UTC start time, application and Unity versions, device
model, bound model asset names, the softmax decision and its justification for each model, the
class order, and a description of every CSV column.

All numeric formatting uses the invariant culture. This is not cosmetic: under a comma-decimal
Windows locale, default .NET formatting emits `0,4213`, which corrupts a comma-separated file.
String fields are quote-escaped. The writer flushes every row and is closed on disable, on quit,
and on application pause — the last because the Quest pauses rather than quits when the headset is
removed, so a quit-only handler would lose the tail of a recording.

### 7.5 Tracking-validity gating

`AUDevice.Poll` previously left its output buffer untouched when `OVRFaceExpressions.ValidExpressions`
was false, and `DeviceManager` forwarded the resulting all-zero frame regardless. A real face never
produces 70 exact zeros, so a tracking dropout entered the model as a fabricated neutral expression.

`Device.Poll` now returns `(float interval, bool valid)` and `DeviceManager.PollAndDispatch`
forwards a reading only when it is valid. Invalid readings are dropped before reaching any ring
buffer. `SoundDevice` was given the same treatment, which additionally removes a latent null
dereference when polled before its asynchronous microphone start completed.

Because a prediction now waits for five *valid* frames, it can no longer complete while tracking is
unavailable. A timeout was added so the game reports "no reading" instead of stalling indefinitely.

---

## 8. Action-unit analysis of the first logged session

Section 5.4 establishes *what* the model predicted during session `20260912_213907_eec228`. This
section asks *why*, using the 350 action-unit values that each row preserves. All figures below
derive from that one file.

Two conventions: an emotion's "profile" is the per-unit peak across a window, averaged over that
emotion's ten windows; AU names are the `OVRPlugin.FaceExpression2` channels the Quest Pro reports.

### 8.1 The ring-buffer correction, confirmed on device

**None of the 60 recorded windows contains five identical frames**, and the mean largest
frame-to-frame action-unit change within a window is 0.159. The defect described in section 2 is
demonstrably absent on hardware, not merely in unit tests.

### 8.2 Expression intensity predicts the score

Peak action-unit value per emotion, averaged over that emotion's ten windows, against its score:

| Emotion | Peak AU | Score |
|---|---|---|
| Sadness | 0.783 | 9/10 |
| Happiness | 0.634 | 8/10 |
| Disgust | 0.427 | 9/10 |
| Surprise | 0.422 | 9/10 |
| **Anger** | **0.242** | **2/10** |
| **Fear** | **0.204** | **0/10** |

Every emotion above 0.40 scores 8–9/10; both emotions below 0.25 score 0–2/10. There is no
exception and no overlap. Across these six classes, how far the face moved is a better predictor of
the outcome than any property of the classifier.

### 8.3 Fear is a low-intensity Surprise

The two action-unit profiles have the same shape at different amplitude, and both of the features
that distinguish them under FACS are absent from the recording:

| AU group | Fear | Surprise | Ratio |
|---|---|---|---|
| Brow raise (shared) | 0.189 | 0.379 | 0.50× |
| Upper lid raiser (shared) | 0.080 | 0.179 | 0.44× |
| Jaw drop (shared) | 0.192 | 0.206 | 0.93× |
| Brow lowerer (**discriminator**) | 0.068 | 0.038 | both ≈ 0 |
| Lip stretcher (**discriminator**) | **0.000** | **0.000** | both absent |

Fear is distinguished from surprise by brows that are raised *and* drawn together, and by
horizontally stretched lips. Neither appears in the data.

**The classifier is therefore not making an error on these trials.** The recorded input is a
low-intensity surprise, and the model reports it as such. There is no fear signal present to detect.

This is not a sensor limitation: `Lip_Stretcher` reaches 0.13 elsewhere in the same session, so the
channel is live. It simply does not activate during fear trials.

### 8.4 Anger is confined to the brow

| AU | Anger | Highest elsewhere in session |
|---|---|---|
| Brow lowerer (AU4) | 0.211 | **Disgust 0.218** |
| Lid tightener (AU7) | 0.086 | Disgust 0.312 |
| Lip tightener (AU23) | 0.000 | Sadness 0.030 |
| Lip pressor (AU24) | 0.000 | Disgust 0.002 |
| Chin raiser (AU17) | 0.000 | Sadness 0.341 |

Only the brow lowerer activates, and it activates *more* strongly during disgust than during anger,
so it carries no discriminative weight. Anger's lower-face component — tightened lips, pressed lips,
raised chin — is entirely absent. The model's Neutral verdict is an accurate description of the
input.

### 8.5 Interpretation

The two failing classes fail because of what reaches the model, not because of what the model does
with it. This inverts the assumption the project has operated under, and it changes the value of two
planned interventions:

- **Probability calibration is ruled out rather than deferred.** Raising Fear's prior would cause
  low-intensity surprise to be labelled fear. That would gain points on fear trials and lose them on
  surprise trials, without improving detection of anything. It optimises the score, not the system.
- **The activity gate would suppress poor predictions without producing correct ones.** Six of ten
  anger windows and four of ten fear windows already satisfy the training gate (three consecutive
  frames with any unit above 0.25), so gating would remove some output without changing what the
  remaining output says.

### 8.6 Limits of this analysis

One session, one subject, ten trials per class; each action-unit profile is a mean over ten windows.
The direction of the findings is consistent and large, but the sample is small and not independent
of the subject's acting.

One alternative explanation cannot yet be excluded. `Lip_Pressor` never exceeds 0.0057 anywhere in
the session, across all six emotions. That may mean the headset, which rests against the face,
physically prevents the movement or prevents its detection. If so, anger may not be expressible
in-headset regardless of effort, and the conclusion would shift from "acting" to "hardware".
Distinguishing the two requires the experiment in section 9.

---

## 9. Remaining risks and next steps

**Immediate, in order:**

1. **Run the protocol re-test.** This is now the highest-value experiment, because it discriminates
   between the two remaining explanations in section 8.6. Record one session in which Fear is acted
   with the brows raised *and* drawn together and the lips stretched horizontally, and Anger with a
   clenched jaw and pressed lips — that is, deliberately driving the specific action units the
   current recording shows to be silent. Then compare the AU profiles against sections 8.3 and 8.4.

   - If the amplitudes rise and the scores follow, the limitation was the acting protocol, and the
     fix is to specify the protocol rather than to change any code.
   - If the amplitudes do not rise, the headset is preventing those movements or their detection,
     and Fear and Anger may not be detectable in-headset at any effort. That would be a hardware
     finding, and it would bound what this system can be expected to do.

   Both outcomes are informative, and neither requires a code change.

2. **Verify the EmojiHeroVR label mapping.** Fear resolves primarily to Surprise in the first
   logged session, and the fear output itself is never first. The unresolved mapping of about 35%
   of training windows is therefore a high-value integrity check. Note that section 8.3 offers a
   complete explanation of the Fear result without invoking permuted labels, so this is no longer
   the leading hypothesis for that specific symptom — but it remains worth settling on its own
   merits.
3. **Record at least two more logged sessions.** Reproduce the Fear-to-Surprise and
   Anger-to-Neutral patterns before fitting any rule. Keep the same protocol and preserve the CSV
   folders unchanged. Note from section 5.4 that two runs of identical code differed by five points,
   so single-session comparisons cannot support conclusions on their own.
4. **Replay logged five-frame windows offline through ONNX Runtime.** Confirm that the Python
   output reproduces the Unity output before testing any activity gate, aggregation, or calibration
   idea.
5. **Add explicit tracking-dropout telemetry if needed.** `no_tracking` captures timeouts, but the
   current successful-window CSV cannot count short invalid polls.

**Deferred until more logged data exists:**

6. **The activity gate.** Training retained a window only if it contained three consecutive frames
   with at least one action unit above 0.25, and required all units below 0.25 for Neutral. The
   runtime applies no such gate, so onsets, offsets and rest are all presented to the model as if
   they were expressions. This should be implemented against recorded action-unit distributions
   rather than against the training threshold assumed to transfer. Note the limit established in
   section 8.5: most failing windows already pass this gate, so it will suppress weak output rather
   than correct it.

**Ruled out:**

7. **Probability calibration and per-class prior correction.** This was previously listed as
   deferred; section 8.5 removes it from consideration. The original proposal was derived from
   *simulated* input streams that predicted Disgust would be the worst-performing class, whereas it
   is now among the best — the simulation did not resemble real action-unit data. More decisively,
   the recorded data shows Fear failing because the input is a genuine low-intensity surprise. A
   prior correction that promoted Fear would relabel that input, gaining points on fear trials at
   the direct expense of surprise trials without improving detection. It would optimise the score
   rather than the system, and should not be implemented.

**Why the EmojiHeroVR check is first:** Every other training source passes through a
`LabelEncoder`, which orders classes alphabetically. EmojiHeroVR labels bypass it: the integer
emotion id is parsed directly out of the sequence identifier and used as the class index. Nothing
confirms that this dataset's native integer order matches the alphabetical order used everywhere
else. EmojiHeroVR contributes approximately 5,784 of 16,742 training windows (~35%). If the orders
differ, that share of the training set carries permuted labels. Confirming this against the dataset
documentation requires no code and gates the value of any further modelling work — though
correcting it would require retraining the teacher, which is outside the current scope.

**Context on the achievable ceiling.** The 0.58 accuracy figure from the teacher's evaluation is the
acted-only subset. The same checkpoint scores 0.30 accuracy and 0.27 macro-F1 on naturally elicited
data, and 0.47 combined. Since this game asks the subject to act, 0.58 is the appropriate reference
— but it is a ceiling, not an expectation. Anger and Fear are the weakest classes in every model in
both notebooks, so gains in those two classes should be expected to be real but bounded.
