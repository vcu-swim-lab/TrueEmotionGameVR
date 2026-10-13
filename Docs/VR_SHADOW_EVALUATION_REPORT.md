# VR Shadow Evaluation — Case Study (n = 1)

**Phase:** C, offline shadow evaluation of `session-local-v1` recordings
**Primary session:** `20260916_201632_85c128` (Oculus Quest Pro)
**Diagnostic session:** `20260917_133343_759c71` — excluded from primary by the block-effect rule
**Frozen from Phase B:** alpha = 0.40, population artifact `ab1f0a1d…`, all classification rules
**Nothing was shown to any participant. Unity and the gameplay decision are unchanged.**
**Audience:** general machine-learning background, no prior knowledge of this project

This is an **n = 1 case study**. No inferential test, p value or confidence interval is
computed, and none may be quoted from it. See §2 for why, and §6 for what the number cannot
resolve.

---

## 1. What was run

Both `session-local-v1` recordings were replayed offline through the unchanged ONNX teacher.
The recordings themselves were not modified. For each session:

```
K block (7 trials) ─ ticks 1–9 ─ log-score ─► P_e   personal centroids
frozen artifact ────────────────────────────► G_e   population centroids (EmojiHero training)
                                              H_e = 0.40·P_e + 0.60·G_e
D block (14 trials) ─ ticks 1–9 ─► raw argmax │ nearest G │ nearest P │ nearest H
                                     └── trial-majority vote, one frozen tie rule ──┘
```

Everything is frozen from Phase B and was not re-tuned: alpha = 0.40, the population artifact
(semantic hash `ab1f0a1de18229374c104434562892ac3fc2b0993d4d925538152ffb045b5ffd`, 20 EmojiHero
training participants), log-score with epsilon 1e-12, cosine, the centroid estimator and the
majority-vote tie rule. All four methods score the identical 14 trials.

## 2. Identity linkage is unknown, and the unit is the participant

**The session logs carry no participant identifier.** `self_report: not collected`, and there is
no `participant_id` field. Whether `85c128` and `759c71` come from the same person or two people
**cannot be determined from the recordings**, and is not assumed either way in this report.

The statistical unit is the participant. With identity unknown and exactly one
quality-passing recording, the effective n for any primary claim is **1**. Trials, ticks and
sessions are not substituted for participants. Consequently:

- no sign test, p value, confidence interval or significance claim appears anywhere below;
- the two sessions are never pooled;
- nothing here counts toward the deployment gate.

## 3. Eligibility

| rule | `85c128` | `759c71` |
|---|---|---|
| `protocol_version == session-local-v1` | pass | pass |
| replay fidelity (argmax 1.000, max prob diff < 1e-4) | **pass** 1.000, 2.98e-07 | **pass** 1.000, 2.98e-07 |
| all seven centroids on nine tracked ticks | **pass** 9/9 each | **pass** 9/9 each |
| all seven classes have a scored D trial | **pass** | **pass** |
| no unscored D trial | **pass** 0 | **pass** 0 |
| **block-effect screen** | **pass** −10 pts, activity 0.91× | **FAIL** −52 pts, activity **0.48×** |

`759c71` fails the pre-existing screen in `block_effect.py`: accuracy falls 52 points from K to D
while the face moves **52% less**. Disgust and Happiness go from 100% in K to 0% in D and 70% of
D ticks are predicted Neutral. An adapter fitted on K is being asked to predict a materially
weaker performance than it was fitted on, so the session cannot separate a calibration effect
from a participant effect. It is reported in §5 as a labelled data-quality diagnostic, is never
pooled with §4, and does not count toward the gate.

## 4. Primary case study — `20260916_201632_85c128`

100% tracking, 210/210 ticks, 126 scored D ticks over 14 trials, no unscored trials, all seven
centroids complete.

### 4.1 Four methods, trial-majority vote

| method | accuracy | macro-F1 |
|---|---|---|
| raw (deployed argmax) | 0.6429 | 0.5952 |
| population-only | 0.5000 | 0.4340 |
| **personal-only** | **0.7143** | **0.6810** |
| hybrid (α = 0.40) | 0.6429 | 0.6293 |

### 4.2 Deltas

| comparison | macro-F1 | reading |
|---|---|---|
| **hybrid − personal** *(primary)* | **−0.0517** | the population term **hurt** |
| **hybrid − raw** *(gate, secondary)* | **+0.0340** | positive, but smaller than the personal term alone |
| population − raw | −0.1612 | the population profile alone is much worse than the model |
| personal − raw | +0.0857 | personal enrolment helped, as in the original single-participant report |

**The direction is opposite to EmojiHero on the comparison that matters.** On EmojiHero test,
`hybrid − personal` was +0.044 for 6 of 7 participants. Here it is **negative**: blending an
EmojiHero population profile into a personal profile that was working made it worse. Note also
that `hybrid − raw` is positive *entirely because of the personal term* — personal-only alone
gives +0.086, and the blend gives back more than half of it. That is exactly the confusion the
two separate deltas exist to prevent.

### 4.3 Per-class trial recall (hits out of 2)

| class | raw | population | personal | hybrid |
|---|---|---|---|---|
| Anger | 0.50 | 0.00 | **1.00** | 0.50 |
| Disgust | 0.50 | 0.50 | **1.00** | 0.50 |
| Fear | 0.00 | 0.00 | 0.00 | 0.00 |
| Happiness | **1.00** | **1.00** | 0.50 | **1.00** |
| Neutral | 1.00 | 1.00 | 1.00 | 1.00 |
| Sadness | 1.00 | 1.00 | 1.00 | 1.00 |
| Surprise | 0.50 | 0.00 | 0.50 | 0.50 |

Personal enrolment buys Anger and Disgust (0.50 → 1.00 each) at the cost of Happiness
(1.00 → 0.50). The hybrid gives Anger and Disgust back to 0.50 and recovers Happiness — it
lands between the two methods on every class, which is what a blend does, and on this session
that is worse than the personal end of the range.

**Fear is 0/2 under all four methods.** Consistent with the pre-registered ceiling
(`CALIBRATION_STUDY_PROTOCOL.md` §1.3): Anger and Fear fail on the model's *input*, and
reinterpreting an output vector cannot create information the input does not carry.

### 4.4 Confusion, trial-majority (rows = prompted, columns = predicted)

Full matrices are in `Analysis/results/vr_shadow_85c128.json`. Summary of where the 14 trials
land:

- **raw**: 9/14 correct. Both Fear trials read as other classes; one Anger, one Disgust and one
  Surprise miss.
- **personal**: 10/14. Anger and Disgust fully recovered; one Happiness trial lost.
- **hybrid**: 9/14. Same count as raw but a different distribution — Happiness back, Anger and
  Disgust partly given up.
- **population**: 7/14. The worst of the four.

### 4.5 Calibration coverage and missing data

| | |
|---|---|
| tracking | 210/210 ticks, 100% |
| K ticks used | 63 (9 per class × 7), tick 0 dropped throughout |
| ticks per centroid | 9 for all seven classes — no incomplete centroid |
| D trials scored | 14 of 14, **0 unscored** |
| `no_tracking` rows | 0 |
| `primary_macro_f1_valid` | **true** |
| replay fidelity | argmax agreement 1.000, max probability difference 2.98e-07 |

### 4.6 Centroid geometry

Personal centroid separation: mean pairwise cosine **0.87847** (min 0.76642, max 0.99567) —
the centroids sit close together, so the decision is fragile. K→D drift **0.97842** mean, and the
drift is lowest exactly where the adapter fails: **Fear 0.9460**, **Happiness 0.9612**.

## 5. Data-quality diagnostic — `20260917_133343_759c71`

**Excluded from every primary figure. Not pooled with §4. Does not count toward the gate.**
Shown because a rule that removes a recording should be visible in what it removed.

| method | accuracy | macro-F1 |
|---|---|---|
| raw | 0.2143 | 0.1011 |
| population-only | 0.2143 | 0.1154 |
| personal-only | 0.1429 | 0.1524 |
| hybrid | 0.2143 | 0.1392 |

Deltas: hybrid − personal **−0.0132**, hybrid − raw +0.0381, population − raw +0.0143,
personal − raw +0.0513.

Every method sits near the 14.3% chance level; raw macro-F1 is 0.101. Per-class trial recall is
0 for five of seven classes under all four methods. Coverage is technically complete — 210/210
ticks, nine ticks per centroid, no unscored trials — so the failure is not a tracking problem:
the participant simply performed the evaluation block far more weakly than the calibration
block, which is what the block-effect screen detected. K→D drift is **0.900** against 0.978 on
the primary session, lowest on Happiness (0.803) and Disgust (0.809) — the two classes that
collapsed from 100% to 0%.

The only thing worth carrying forward is that `hybrid − personal` is **negative here too**, in
the same direction as §4.

## 6. What 14 trials can resolve

Asked before any threshold is proposed. With 14 D trials, two per class over seven classes:

- From a perfect 14/14, **one trial flipping costs exactly −0.0762 macro-F1**.
- Around the operating range of this session (base macro-F1 0.6–0.7), the median absolute
  macro-F1 change from a single trial changing its prediction is **≈ 0.062**.

Against that resolution:

| observed | size in trials |
|---|---|
| hybrid − personal, −0.0517 | **less than one trial** |
| hybrid − raw, +0.0340 | about **half a trial** |
| personal − raw, +0.0857 | about **1.4 trials** (9/14 → 10/14 at the accuracy level) |

**Only `personal − raw` is larger than the measurement's own granularity.** The primary
comparison and the gate comparison are both below or near the size of a single trial changing
its mind. This is the concrete reason a harm threshold such as "no participant worse than
−0.05" must **not** be frozen now: it would sit below the resolution of the instrument that is
supposed to test it. The threshold belongs in the preregistration of the multi-participant VR
pilot, written before that data is seen, and should be set against the resolution its own trial
count provides.

## 7. Deployment gate

**The gate is not open, and Phase C cannot open it.** n = 1.

What a future pilot would have to show, stated now so it cannot be chosen later:

1. `hybrid − raw > 0` on trial-majority macro-F1 for at least **7 of 8** participants if eight
   are collected — not 6 of 8. With eight participants an exact two-sided sign test gives
   p = 0.070 at 7/8 and p = 0.289 at 6/8, so 6/8 is not consistency, it is noise with a
   direction. **7/8 is a minimum bar, not proof.**
2. Mean `hybrid − raw > 0` across participants.
3. No participant harmed beyond a threshold fixed in that pilot's preregistration, chosen
   against the macro-F1 resolution its own trial count supports (§6).
4. `hybrid ≥ personal`. **On the only usable live recording this already fails**: personal-only
   beats hybrid by 0.052. On that evidence the method to carry forward is personal-only, not
   hybrid.
5. Evidence from **live participants**. EmojiHero results do not count toward the gate.

**Recommendation on sample size.** Eight participants is the smallest defensible pilot and it
is genuinely marginal: the floor on a two-sided sign test at n = 8 is p = 0.008, reachable only
at 8/8, and a single dissenter costs an order of magnitude in p. **12–16 participants** would
let a 9/12 or 12/16 result mean something, would survive one or two block-effect exclusions
without collapsing the analysis, and would allow the harm threshold in (3) to be estimated
rather than guessed. `CALIBRATION_STUDY_PROTOCOL.md` §8 already notes that 15–20 participants
is worth more than more trials from fewer people; this analysis is consistent with that and
against treating 8 as sufficient.

## 8. Interpretation

**Supported by this case study:**

- On one quality-passing live recording, personal-only calibration beat the raw model
  (+0.086 macro-F1, 9/14 → 10/14 trials), reproducing the earlier single-participant result.
- On that same recording, blending the EmojiHero population profile in at α = 0.40 **reduced**
  performance relative to personal-only (−0.052), and the same sign appears on the excluded
  diagnostic session (−0.013).
- Both of those differences are within about one trial of measurement granularity.

**Not supported by anything here:**

- Any claim about the population term generalising from EmojiHero to the headset. The direction
  reversed, and the likely mechanism is domain shift: the population centroids describe how the
  teacher responds to **EmojiHero input** — 30-frame captures mean-pooled three at a time —
  while a live window is five instantaneous 10 Hz samples. The blend is well defined because
  both live in the model's seven-dimensional output space, but the population term is applied
  off the distribution it was estimated on.
- Any significance claim, from either session.
- Any statement about a second participant, since identity linkage is unknown.
- Anything that would justify wiring a calibrated prediction into gameplay.

**The multi-participant live pilot remains necessary and is now the only thing that can move
this forward.**

## 9. Next step

Collect the multi-participant `session-local-v1` pilot, preregistered before data collection,
with the harm threshold and the participant count fixed in advance against the resolution in
§6. Given §4.2 and §8, that pilot should treat **personal-only as the method under test** and
carry hybrid as a secondary arm, rather than the other way round.

Do not change alpha, the population artifact or any classification rule on the strength of this
case study. n = 1.

---

**Reproduce**

```bash
python Analysis/test_calibration_study.py            # 67 tests

python Analysis/calibration_study.py \
    "Logs/quest-sessions/sessions/session_20260916_201632_85c128" \
    --check --enrol-role calibration --test-role evaluation \
    --population-centroids Analysis/results/emojihero_population_centroids.json \
    --hybrid-alpha 0.40 --compare-all \
    --results-dir Analysis/results/vr_shadow \
    --json Analysis/results/vr_shadow_85c128.json
```

Without `--population-centroids` and `--hybrid-alpha` the script produces exactly the
personal-only result it produced before this phase, key set included; a test asserts it.
