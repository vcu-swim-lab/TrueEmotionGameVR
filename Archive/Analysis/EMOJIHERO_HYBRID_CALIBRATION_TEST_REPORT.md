# EmojiHero Hybrid Calibration — Frozen Test Phase

**Study:** `emojihero-personal-v1` (offline analogue of `session-local-v1`)
**Split reported:** EmojiHeroVR **test** — 8 participants, 378 sequences, **7 eligible**
**Preregistration snapshot:** `cc8ff2d959a6dca4b98b010b4540f835dd428bc6`
**Run at:** `86801565184bf3243cfe47a58782fe8852ccc272`
**Run once.** No rule, seed, alpha or threshold was changed before or after this result.
**Audience:** general machine-learning background, no prior knowledge of this project

The validation phase and its caveats are a separate document:
[`EMOJIHERO_HYBRID_CALIBRATION_VALIDATION_REPORT.md`](EMOJIHERO_HYBRID_CALIBRATION_VALIDATION_REPORT.md).
The two are reported side by side in §3 and are **not** pooled.

---

## 1. What was frozen before this ran

| pinned | value |
|---|---|
| alpha | **0.40**, from the frozen config, not the command line |
| primary metric | participant-level seven-class sequence macro-F1 |
| primary comparison | **hybrid − personal** |
| inference unit | participant |
| statistical test | exact two-sided paired sign test |
| seed / primary profile / robustness | 42 / 0 / 0–999 |
| window condition | `stride1` |
| ONNX model SHA-256 | `ef6cd4134f4d64ab5a7233ac719cc05de88963b33757eca4d92f1b625413343c` |
| population artifact (file SHA-256) | `a611246653af0580dff3ea305f0aa588502f87e3964be85b4c5c7cf093d0996a` |
| population artifact (semantic hash, the pin) | `ab1f0a1de18229374c104434562892ac3fc2b0993d4d925538152ffb045b5ffd` |

The run refused nothing, because nothing was overridden: `--split test` accepts no
`--hybrid-alpha`, `--profile-count` or `--windows`, and the parser has no `--epsilon`,
`--metric`, `--seed` or `--primary-profile` flag at all.

Between 996 and 1000 of the 1000 sampled calibration profiles were distinct per participant.

## 2. Primary result

> **hybrid − personal**, participant-level seven-class sequence macro-F1, averaged within each
> participant across the 1000 profiles, then tested across participants.

**Mean +0.0439. Six of seven participants improved, one did not. Ties 0, effective n = 7.
Exact two-sided sign test p = 0.125.**

At the single primary profile (index 0) the same comparison gives **+0.0556, 6/7, ties 0,
n = 7, p = 0.125** — the same direction, the same count, the same p.

**This does not reach significance, and with these data it could not have unless the result
were unanimous.** Seven participants put the smallest reachable two-sided p at 0.0156; one
dissenter raises it to 0.125. The design cannot distinguish 6/7 from chance, and that is a
property of n = 7, not a finding about the method.

### 2.1 The seven eligible participants, separately

Mean over the 1000 profiles, macro-F1:

| participant | raw | personal | hybrid | **hybrid − personal** |
|---|---|---|---|---|
| 1 | 0.4473 | 0.4349 | 0.4867 | **+0.0517** |
| 8 | 0.4093 | 0.5493 | 0.5077 | **−0.0416** |
| 10 | 0.5579 | 0.4232 | 0.5251 | **+0.1019** |
| 13 | 0.7008 | 0.5605 | 0.6340 | **+0.0735** |
| 15 | 0.4896 | 0.4677 | 0.4888 | **+0.0211** |
| 23 | 0.6213 | 0.6163 | 0.7070 | **+0.0906** |
| 27 | 0.6473 | 0.5943 | 0.6043 | **+0.0100** |

At the primary profile:

| participant | raw | personal | hybrid | **hybrid − personal** |
|---|---|---|---|---|
| 1 | 0.4350 | 0.3266 | 0.5163 | **+0.1897** |
| 8 | 0.3781 | 0.5938 | 0.4815 | **−0.1123** |
| 10 | 0.6235 | 0.4567 | 0.5144 | **+0.0577** |
| 13 | 0.7012 | 0.4569 | 0.5182 | **+0.0613** |
| 15 | 0.4447 | 0.3165 | 0.4405 | **+0.1240** |
| 23 | 0.6152 | 0.6578 | 0.7236 | **+0.0658** |
| 27 | 0.6127 | 0.7134 | 0.7164 | **+0.0030** |

Participant 8 is the dissenter in both views, and is the one participant for whom personal
enrolment worked well on its own (+0.140 over raw). For that person the population term is
dilution, not shrinkage. Participants 27 and 15 sit close to zero, so the mean rests largely on
1, 10, 13 and 23.

## 3. Validation and test, side by side

Reported together for comparison only. They are different participants and are never pooled.

| | validation (n = 8) | test (n = 7) |
|---|---|---|
| alpha | 0.40 *(chosen here)* | 0.40 *(frozen, not re-chosen)* |
| **hybrid − personal** | **+0.0607** | **+0.0439** |
| improved | 8/8 | 6/7 |
| ties | 0 | 0 |
| p | 0.008 | 0.125 |
| smallest reachable p | 0.008 | 0.016 |

The effect **shrank and lost unanimity**, which is what an in-sample optimum does when it
leaves the participants it was fitted on. The validation figure was the value the alpha search
was pointed at; this one is not. A +0.044 direction that survives the move to new participants
is worth something — it is simply not, on 7 people, statistically distinguishable from chance.

## 4. Secondary results

Not primary, and none of these may be promoted after the fact.

### 4.1 Other comparisons, macro-F1, mean over profiles

| comparison | mean | improved | ties | n | p |
|---|---|---|---|---|---|
| hybrid − raw | +0.0114 | 3/7 | 0 | 7 | 1.000 |
| personal − raw | −0.0325 | 1/7 | 0 | 7 | 0.125 |
| population − raw | −0.0595 | 0/7 | 0 | 7 | **0.016** |

**Personal-only enrolment lost to raw again**, on new participants: −0.033, helping 1 of 7. The
validation phase saw −0.043 helping 2 of 8. The two splits agree, and the direction is the
opposite of the single live participant's.

**Population-only lost to raw on every single participant**, 0/7, p = 0.016. That is the only
comparison in this study that clears its own significance floor, and it is a negative result:
a population profile used alone is worse than leaving the model's argmax untouched.

**Hybrid − raw is flat**, +0.011 at 3/7. Blending recovers what personal-only gives away; it
does not beat the deployed model.

### 4.2 Accuracy (secondary throughout)

Participant-equal means over profiles:

| method | macro-F1 | accuracy |
|---|---|---|
| raw | 0.5534 | 0.5863 |
| population | 0.4939 | 0.5555 |
| personal | 0.5209 | 0.5613 |
| hybrid | 0.5648 | 0.6103 |

| comparison | accuracy mean | improved | ties | n | p |
|---|---|---|---|---|---|
| hybrid − personal | +0.0490 | 5/7 | 0 | 7 | 0.453 |
| hybrid − raw | +0.0241 | 5/7 | 0 | 7 | 0.453 |
| personal − raw | −0.0250 | 1/7 | 0 | 7 | 0.125 |
| population − raw | −0.0307 | 1/7 | 0 | 7 | 0.125 |

At the primary profile the accuracy comparisons carry **one tie each** for `hybrid − personal`,
`hybrid − raw` and `personal − raw`, dropping the effective n to 6. The sign test drops ties
from the denominator, so those p values are computed on n = 6, not 7. The macro-F1 primary has
no ties at either view.

### 4.3 Per class

Mean sequence-recall change over profiles and participants:

| class | personal − raw | hybrid − personal |
|---|---|---|
| Anger | **+0.118** | −0.116 |
| Disgust | −0.209 | **+0.136** |
| Fear | +0.050 | −0.006 |
| Happiness | −0.092 | +0.125 |
| Neutral | −0.054 | **+0.192** |
| Sadness | +0.108 | −0.046 |
| Surprise | −0.206 | +0.079 |

The same redistribution the validation phase and the live study both show: personal enrolment
buys Anger and Sadness at the cost of Disgust and Surprise, and the population term buys them
back. No method lifts all seven.

### 4.4 Diagnostics

Calibration-minus-evaluation mean level gap ranges −0.044 to +0.011 across the eligible
participants on a 1–4 scale, so expression-intensity drift is not driving the result. Personal
centroid separation 0.838–0.880; personal-to-population cosine 0.943–0.970, again showing that a
person's profile and the population's are already very similar.

## 5. Participant 18 — missing-class diagnostic, not part of any primary figure

Participant 18 has a single Fear sequence. Enrolling it leaves nothing of that class to score,
and **the sequence was not split**. All seven classes were enrolled; only six carry a scored
sequence, so `primary_macro_f1_valid` is `false` and this participant is excluded from every
number in §2 and §4. Reported separately, on 36 evaluation sequences:

- hybrid − personal, macro-F1 over profiles: **+0.085**
- personal − raw, macro-F1 over profiles: **−0.158**
- raw macro-F1 0.942, the highest of any participant

These are **six-class** figures and are not comparable with the seven-class ones above. A
macro-F1 that omits a class is computed over a smaller mean and is not on the same scale; that
is precisely why the participant is excluded rather than folded in.

## 6. What this does and does not support

**Supported:**

- On EmojiHero test participants, blending a population profile into a one-sequence personal
  profile improved participant-level macro-F1 for 6 of 7 people, mean +0.044, p = 0.125 — a
  consistent direction that does not reach significance at n = 7.
- A one-sequence-per-emotion personal enrolment, on this corpus, is **worse than doing nothing**
  — on both splits.
- A population profile used alone is worse than the raw model on every test participant.

**Not supported by anything here:**

- That hybrid calibration helps on the live headset. This is the offline analogue of
  `session-local-v1`, not a repeat of it: EmojiHero stride-1 windows share 4 of 5 frames where
  live windows share none, a sequence is 30 frames where a live trial is ~10 seconds, and the
  prompt protocol differs. The result bounds the live effect in **neither** direction.
- That personal calibration fails in the live protocol. The enrolment budgets are not
  comparable — one 30-frame sequence against nine ticks over ~10 seconds — and the live
  single-participant result went the other way.
- That the live 6–8 participant pilot can be skipped. **It cannot.**
- Any claim resting on profile count. The 1000 profiles are re-draws of the same participants'
  sequences, averaged within each participant before any test. n is 7.

**Provenance, unchanged from validation:** the teacher was trained by gradient on the EmojiHero
training split, which is also where the population centroids come from. The test split entered
neither gradients nor model selection, but had been opened before when reporting teacher
performance; its metadata and label counts were also read during the pre-registered dry-run.
No test outcome was seen before this run.

## 7. Recommendation

Proceed to the **VR shadow evaluation** — compute all four methods offline on the existing
`session-local-v1` recordings, with the frozen population artifact and alpha = 0.40, showing the
participant nothing. That is the only measurement that speaks to the deployed system, and the
enrolment-budget difference makes it genuinely informative rather than a foregone conclusion.

Do **not** wire any calibrated prediction into gameplay. Nothing here justifies it: the primary
result is not significant, hybrid does not beat the raw model, and the live evidence is one
participant.

If more EmojiHero work is wanted later, the question this study raises is whether a **thicker**
personal enrolment — two or three sequences per emotion rather than one — changes the picture,
since the failure of personal-only on both splits is consistent with a noisy one-observation
centroid. That would be a new preregistration, not an edit to this one.

---

**Known cosmetic defect, not corrected here.** The result JSON records the population artifact
by its *file* hash while the frozen config pins its *semantic* hash. Enforcement used the
semantic hash, as designed; only the provenance echo in the output differs. It is left alone so
that no code changed between the frozen snapshot and this result, and should be fixed before the
next phase.

**Reproduce** (the command that was run, once):

```bash
python Analysis/emojihero_calibration_study.py --split test \
    --population-centroids Analysis/results/emojihero_population_centroids.json \
    --frozen-config Analysis/results/emojihero_hybrid_frozen_config.json \
    --output Analysis/results/emojihero_hybrid_test.json
```
