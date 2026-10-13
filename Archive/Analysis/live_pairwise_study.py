#!/usr/bin/env python3
"""Within-person against between-person, for each emotion, side by side.

Three numbers per emotion:

    A1 <-> A2    one person, two evenings
    A1 <-> B     that person's first evening against the other person
    A2 <-> B     that person's second evening against the other person

If the first is larger than both of the others, then the same person twice looks more alike
than two different people do, and identity is the bigger effect. If it is not, then one
person's variation between two evenings is as large as the difference between two people --
and a per-person calibration fitted on one visit would not survive to the next.

That is the whole question, and the two previous studies each answered a piece of it in
passing. `LIVE_VS_ACTED_REPORT.md` printed A's between-session cosine as a yardstick but never
put it next to the corresponding A-B numbers; `WINDOW_LEVEL_REPORT.md` measured the windows
but only against the acted corpus. This file puts the three in one table, and does it in both
representations, because they do not have to agree:

  **peak**     the 70-channel per-emotion maximum, the unit of the first report
  **window**   the [5, 70] blocks the model actually consumed, the unit of the second

There is no acted corpus in this comparison at all. It is live against live.

    python Analysis/live_pairwise_study.py --json results/live_pairwise.json

Descriptive only. Two people, and only one of them was recorded twice -- so there is a
within-person number for A and none for B, which is a limit stated rather than worked around.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import au_names                       # noqa: E402
import live_vs_acted_study as lva     # noqa: E402
import session_analysis as sa         # noqa: E402
import window_level_study as wl       # noqa: E402

COMMON_EMOTIONS = lva.COMMON_EMOTIONS
LIVE_SUBJECTS = lva.LIVE_SUBJECTS

REPRESENTATIONS = ('peak', 'window')

# Same two channel sets as the window study, and the same primary: the model is given all
# seventy, so that is what "the inputs differ" has to mean.
CHANNEL_SETS = wl.CHANNEL_SETS
PRIMARY_CHANNELS = wl.PRIMARY_CHANNELS


class StudyError(RuntimeError):
    """Raised when the inputs are not shaped the way this comparison requires."""


def session_vectors(session: sa.Session, representation: str,
                    channels: list[int]) -> dict[str, np.ndarray]:
    """One session's per-emotion representation, as a [k, d] matrix.

    `peak` gives one row per emotion -- the per-channel maximum over the session's scored
    frames, which is what `live_vs_acted_study` builds. `window` gives one row per scored
    window, flattened frame-major, which is what `window_level_study` builds. Both reuse those
    modules rather than recomputing, so this file cannot drift from either.
    """
    if representation == 'peak':
        profiles = lva.live_profiles(session)
        return {e: profiles[e]['peak'][None, channels] for e in COMMON_EMOTIONS}

    if representation == 'window':
        windows = wl.live_windows(session)
        columns = wl._channel_columns(channels)
        return {e: wl.flatten(windows[e]['windows'])[:, columns] for e in COMMON_EMOTIONS}

    raise StudyError(f'unknown representation {representation!r}; '
                     f'expected one of {REPRESENTATIONS}')


def compare(left: np.ndarray, right: np.ndarray) -> dict:
    """Similarity between two sets of vectors.

    `mean` is the average over every cross pair. It is symmetric, uses all the data, and is
    the primary: it asks how alike the two sets are overall.

    `best` is the average, over each side, of each vector's single closest match on the other
    -- symmetrised, since taking it in one direction only would give a different answer
    depending on which set was named first. It asks whether the two sets overlap *anywhere*,
    which is a weaker and more forgiving question, and it is reported second for that reason.

    With one vector per side (the `peak` representation) the two coincide, and that is correct
    rather than a degenerate case.
    """
    sims = wl.cosine_matrix(left.astype(np.float64), right.astype(np.float64))
    best = float(np.mean([sims.max(axis=1).mean(), sims.max(axis=0).mean()]))
    return {
        'mean': float(sims.mean()),
        'best': best,
        'min': float(sims.min()),
        'max': float(sims.max()),
        'n_pairs': int(sims.size),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Same helper as `live_vs_acted_study`, kept local so this
    file's numbers do not depend on a private name there."""
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def run(live_root: str | Path | None, representation: str, channels: str) -> dict:
    if channels not in CHANNEL_SETS:
        raise StudyError(f'unknown channel set {channels!r}')

    files = lva.resolve_live_sessions(live_root)

    # This comparison needs exactly one subject with two sessions and one with a single
    # session. If that ever changes, the table below stops meaning what it says, so it is
    # checked rather than assumed.
    repeated = [s for s, paths in files.items() if len(paths) == 2]
    single = [s for s, paths in files.items() if len(paths) == 1]
    if len(repeated) != 1 or len(single) != 1 or len(files) != 2:
        raise StudyError(
            'this comparison assumes exactly two subjects, one recorded twice and one once; '
            f'got { {s: len(p) for s, p in files.items()} }')
    a, b = repeated[0], single[0]

    columns = CHANNEL_SETS[channels]
    a1, a2 = [session_vectors(sa.load_session(p), representation, columns)
              for p in files[a]]
    b1 = session_vectors(sa.load_session(files[b][0]), representation, columns)

    emotions: dict[str, dict] = {}
    for emotion in COMMON_EMOTIONS:
        within = compare(a1[emotion], a2[emotion])
        cross_1 = compare(a1[emotion], b1[emotion])
        cross_2 = compare(a2[emotion], b1[emotion])

        # The verdict, on the primary statistic. "identity" means the same person twice is
        # more alike than either evening is to the other person -- what a per-person profile
        # needs in order to be worth fitting.
        cross_best = max(cross_1['mean'], cross_2['mean'])
        emotions[emotion] = {
            'within_person': within,
            'cross_1': cross_1,
            'cross_2': cross_2,
            'margin': within['mean'] - cross_best,
            'identity_dominates': within['mean'] > cross_best,
        }

    # Does the repeated subject's own consistency track their own accuracy? Six emotions from
    # one person, so this is a scatter to look at, not a result. It is worth printing because
    # the peak-profile study found no association at all between distance to the acted
    # training corpus and accuracy (Spearman +0.007), and this is a different quantity.
    model = lva.live_model_results([sa.load_session(p) for p in files[a]])
    consistency = np.array([emotions[e]['within_person']['mean'] for e in COMMON_EMOTIONS])
    accuracy = np.array([model[e]['acc_scored'] for e in COMMON_EMOTIONS])
    association = {
        'subject': a,
        'n_emotions': len(COMMON_EMOTIONS),
        'rows': [{'emotion': e,
                  'within_person': float(emotions[e]['within_person']['mean']),
                  'accuracy': float(model[e]['acc_scored'])}
                 for e in COMMON_EMOTIONS],
        'spearman': float(np.corrcoef(_rank(consistency), _rank(accuracy))[0, 1]),
        'pearson': float(np.corrcoef(consistency, accuracy)[0, 1]),
    }

    wins = sum(1 for e in emotions.values() if e['identity_dominates'])
    return {
        'association': association,
        'representation': representation,
        'channel_set': channels,
        'channel_count': len(columns),
        'repeated_subject': a,
        'single_subject': b,
        'sessions': {a: [p.parent.name for p in files[a]],
                     b: [p.parent.name for p in files[b]]},
        'emotions': emotions,
        'identity_dominates_count': wins,
        'n_emotions': len(COMMON_EMOTIONS),
        'mean_within': float(np.mean([e['within_person']['mean'] for e in emotions.values()])),
        'mean_cross': float(np.mean([max(e['cross_1']['mean'], e['cross_2']['mean'])
                                     for e in emotions.values()])),
    }


# -------------------------------------------------------------------- printing


def print_table(result: dict) -> None:
    a, b = result['repeated_subject'], result['single_subject']
    print()
    print('=' * 78)
    print(f'representation: {result["representation"]}   '
          f'channels: {result["channel_set"]} ({result["channel_count"]} of 70)')
    if result['representation'] == 'peak':
        print('  one 70-channel peak vector per emotion per session')
    else:
        print('  every scored [5, 70] window, flattened frame-major')
    print('=' * 78)
    print()
    print(f'  {"emotion":<11} {a}1<->{a}2  {a}1<->{b}   {a}2<->{b}   {"margin":>8}   verdict')
    for emotion in result['emotions']:
        e = result['emotions'][emotion]
        verdict = ('identity larger' if e['identity_dominates']
                   else 'SESSION larger')
        print(f'  {emotion:<11} {e["within_person"]["mean"]:>9.3f} '
              f'{e["cross_1"]["mean"]:>9.3f} {e["cross_2"]["mean"]:>10.3f} '
              f'{e["margin"]:>+9.3f}   {verdict}')

    print()
    print(f'  identity larger in {result["identity_dominates_count"]} of '
          f'{result["n_emotions"]} emotions')
    print(f'  mean within-person {result["mean_within"]:.3f}   '
          f'mean between-person {result["mean_cross"]:.3f}   '
          f'difference {result["mean_within"] - result["mean_cross"]:+.3f}')
    print()


def print_best_match(result: dict) -> None:
    """The forgiving statistic, for contrast with the primary."""
    a, b = result['repeated_subject'], result['single_subject']
    print('-- best-match instead of mean (does anything overlap at all?) ' + '-' * 16)
    print(f'  {"emotion":<11} {a}1<->{a}2  {a}1<->{b}   {a}2<->{b}   {"margin":>8}')
    for emotion in result['emotions']:
        e = result['emotions'][emotion]
        margin = e['within_person']['best'] - max(e['cross_1']['best'], e['cross_2']['best'])
        print(f'  {emotion:<11} {e["within_person"]["best"]:>9.3f} '
              f'{e["cross_1"]["best"]:>9.3f} {e["cross_2"]["best"]:>10.3f} {margin:>+9.3f}')
    print()


def print_association(result: dict) -> None:
    """One person's own repeatability against their own accuracy."""
    assoc = result['association']
    print(f'-- does {assoc["subject"]}\'s own consistency track {assoc["subject"]}\'s own '
          'accuracy? ' + '-' * 18)
    print('   EXPLORATORY. Six emotions from one person. Note also that part of this is')
    print('   near-tautological: an expression held inconsistently is harder to hit. It is')
    print('   printed because the peak-profile study found NO association between distance')
    print('   to the acted training corpus and accuracy, and this is a different quantity.')
    print()
    print(f'  {"emotion":<11} {"within-person":>14} {"accuracy":>10}')
    for row in sorted(assoc['rows'], key=lambda r: r['within_person']):
        print(f'  {row["emotion"]:<11} {row["within_person"]:>14.3f} '
              f'{row["accuracy"]:>10.2f}')
    print()
    print(f'  Spearman {assoc["spearman"]:+.3f}   Pearson {assoc["pearson"]:+.3f}   '
          f'n = {assoc["n_emotions"]} emotions, 1 person')
    print()


def print_limits(result: dict) -> None:
    a, b = result['repeated_subject'], result['single_subject']
    print('-- limits ' + '-' * 67)
    print(f'  Only {a} was recorded twice. There is no within-person number for {b}, so this')
    print(f'  table compares {a}\'s session-to-session variation against the {a}-{b} gap and')
    print(f'  cannot say whether {b} is more or less variable than {a}.')
    print()
    print('  Two people. Six emotions from two people is not twelve observations and no')
    print('  p-value is computed.')
    print()
    print(f'  The two {a} sessions were recorded hours apart under the same conditions. They')
    print('  set a floor on one person\'s variation, not a general estimate of it -- a visit')
    print('  a week later could differ by more.')
    print()
    print('  A within-person number larger than the between-person one does not by itself')
    print('  support a per-person calibration; it is a necessary condition, not a design.')
    print()


def to_json(obj):
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sessions', help='live session file or folder')
    parser.add_argument('--channels', choices=sorted(CHANNEL_SETS), default=PRIMARY_CHANNELS)
    parser.add_argument('--representation', action='append', choices=REPRESENTATIONS,
                        help='repeatable; default is both')
    parser.add_argument('--json', help='write the full result here')
    args = parser.parse_args(argv)

    representations = tuple(args.representation or REPRESENTATIONS)

    results = {}
    try:
        for representation in representations:
            results[representation] = run(args.sessions, representation, args.channels)
    except (StudyError, lva.StudyError, wl.StudyError, sa.SchemaError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    first = next(iter(results.values()))
    a, b = first['repeated_subject'], first['single_subject']
    print('=' * 78)
    print('within-person against between-person, live sessions only')
    print(f'  subject {a}   {", ".join(first["sessions"][a])}')
    print(f'  subject {b}   {", ".join(first["sessions"][b])}')
    print('  no acted data enters this comparison')
    print('=' * 78)

    for result in results.values():
        print_table(result)
        if result['representation'] == 'window':
            print_best_match(result)
            print_association(result)

    print_limits(first)

    if args.json:
        path = Path(args.json)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(results), indent=2), encoding='utf-8')
        print(f'wrote {path.relative_to(Path(__file__).resolve().parent)}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
