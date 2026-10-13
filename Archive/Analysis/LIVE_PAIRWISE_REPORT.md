# Within-Person Against Between-Person

**Scope:** live sessions only. No acted data enters this comparison.

**Reproduce:**
`python Analysis/live_pairwise_study.py --json results/live_pairwise.json`
`python Analysis/test_live_pairwise_study.py`   (21 tests)

**Status:** descriptive. Two people, and only one of them was recorded twice.

**The question:** for each emotion, is one person's variation between two evenings larger or
smaller than the difference between two people? If it is smaller, a per-person profile has
something stable to be fitted to. If it is larger, a profile captured on one visit will not
survive to the next, and per-person calibration is the wrong shape of fix.

---

## 1. The table

Cosine, mean over every cross pair. Reported in both representations, because the earlier
studies showed they can disagree: `peak` is the one 70-channel maximum per emotion that
[LIVE_VS_ACTED_REPORT.md](LIVE_VS_ACTED_REPORT.md) used, `window` is the `[5, 70]` blocks the
model actually consumed, from [WINDOW_LEVEL_REPORT.md](WINDOW_LEVEL_REPORT.md).

### peak — one 70-channel vector per emotion per session

| emotion | A1↔A2 | A1↔B | A2↔B | margin | |
| --- | --- | --- | --- | --- | --- |
| Anger | 0.607 | 0.419 | 0.535 | +0.071 | identity larger |
| Disgust | 0.921 | 0.737 | 0.614 | +0.184 | identity larger |
| **Fear** | **0.777** | **0.897** | **0.896** | **−0.119** | **session larger** |
| Happiness | 0.967 | 0.910 | 0.925 | +0.043 | identity larger |
| Sadness | 0.904 | 0.766 | 0.820 | +0.084 | identity larger |
| Surprise | 0.976 | 0.569 | 0.519 | +0.407 | identity larger |

Identity larger in 5 of 6. Mean within-person 0.859, mean between-person 0.747, **+0.112**.

### window — every scored 5×70 window (primary)

| emotion | A1↔A2 | A1↔B | A2↔B | margin | |
| --- | --- | --- | --- | --- | --- |
| Anger | 0.604 | 0.587 | 0.459 | +0.018 | identity larger, barely |
| Disgust | 0.794 | 0.295 | 0.229 | +0.499 | identity larger |
| **Fear** | **0.480** | **0.588** | **0.382** | **−0.108** | **session larger** |
| Happiness | 0.889 | 0.563 | 0.637 | +0.252 | identity larger |
| Sadness | 0.914 | 0.399 | 0.289 | +0.516 | identity larger |
| Surprise | 0.907 | 0.629 | 0.565 | +0.278 | identity larger |

Identity larger in 5 of 6. Mean within-person 0.765, mean between-person 0.522, **+0.242**.

**The two representations agree on the verdict and disagree on the size.** In the space the
model actually works in, the identity effect is more than twice as large: +0.242 against
+0.112. The peak profile understates it, which is consistent with what the window study found
about peak profiles generally.

### The forgiving statistic

`best` asks only whether the two sets overlap *anywhere* — each window's single closest match
on the other side, symmetrised — rather than how alike they are overall:

| emotion | A1↔A2 | A1↔B | A2↔B | margin |
| --- | --- | --- | --- | --- |
| Anger | 0.717 | 0.771 | 0.616 | **−0.054** |
| Disgust | 0.875 | 0.412 | 0.323 | +0.463 |
| Fear | 0.735 | 0.736 | 0.585 | **−0.001** |
| Happiness | 0.940 | 0.717 | 0.767 | +0.172 |
| Sadness | 0.934 | 0.684 | 0.607 | +0.249 |
| Surprise | 0.963 | 0.744 | 0.644 | +0.219 |

Under it, Anger flips and Fear is a dead heat. So **Anger and Fear are the two emotions where
A and B genuinely overlap**; the other four separate cleanly under either statistic.

## 2. What it says

**Identity is the larger effect, for four of six emotions, clearly.** Disgust, Happiness,
Sadness and Surprise all have the same person's two evenings far more alike than either is to
the other person, by 0.17 to 0.52 in window space. For those four, a per-person profile has a
stable thing to be fitted to.

**Fear is the exception, in both representations and under both statistics.** A's two
evenings of Fear are *less* alike than either is to B's Fear. That is not a stray result — it
matches everything else found about Fear: both players' Fear windows land nearer acted
Surprise than acted Fear, both score 0.00, and the acted corpus itself has Fear nearest
Surprise between its own training participants. Fear is not a personal signature. It is a
shared failure that no per-person calibration would touch.

**Anger is borderline.** +0.018 in window space on the primary statistic, and −0.054 on the
forgiving one. It is also A's least repeatable emotion in the peak space (0.607) and the one
where A scores only 0.28 despite being close to the acted training profiles.

## 3. One observation worth marking, not believing

Within-person consistency and A's own accuracy line up almost exactly:

| emotion | A's within-person consistency | A's accuracy |
| --- | --- | --- |
| Fear | 0.480 | 0.00 |
| Anger | 0.604 | 0.28 |
| Disgust | 0.794 | 1.00 |
| Happiness | 0.889 | 0.94 |
| Surprise | 0.907 | 1.00 |
| Sadness | 0.914 | 1.00 |

Spearman **+0.820**, Pearson **+0.968**, over six emotions from one person.

Set against the earlier null — distance to the acted training corpus versus accuracy, Spearman
+0.007 over twelve cells — this is the first quantity measured in this project that tracks
accuracy at all. But three things hold it down:

1. **Six emotions from one person.** This is a scatter to look at, not a result.
2. **Part of it is near-tautological.** An expression a person holds inconsistently between two
   evenings is, by construction, harder for a fixed classifier to hit twice.
3. **There is no within-person number for B**, so this cannot be checked on the person it
   would matter most for.

If it survives on new participants, it says something useful and unwelcome: what predicts
whether the model works on someone is how *repeatable* their expressions are, not how close
they are to the training set — and repeatability is a property of the person, which a
calibration profile cannot supply.

## 4. Limits

- **Only A was recorded twice.** There is no within-person number for B, so this table cannot
  say whether B is more or less variable than A — only how A's variation compares to the A–B
  gap.
- **Two people.** Six emotions from two people is not twelve observations, and no p-value is
  computed.
- **A's two sessions were hours apart** under the same conditions. They set a floor on one
  person's variation, not a general estimate of it; a visit a week later could differ by more.
  This is exactly what the two-visit design in
  [CALIBRATION_STUDY_PROTOCOL.md](../Docs/CALIBRATION_STUDY_PROTOCOL.md) exists to measure.
- **A within-person number larger than the between-person one is a necessary condition for a
  per-person calibration, not a design.** It says there is something stable to fit; it says
  nothing about whether fitting it helps.
