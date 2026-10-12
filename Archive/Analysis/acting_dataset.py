#!/usr/bin/env python3
"""Loader for the ActingRecordings corpus.

Layout:

    ActingRecordings/<participant>/<Emotion>/acting.csv
                     <participant>/emotion_flow.txt

One CSV per participant per emotion: a single ~10 s acted trial sampled at ~70 Hz, 115
columns of which the last 70 are the face channels named in `au_names.py`. Neutral is
prompted here, which the live game never did, so this is the only corpus on hand that can
support a seven-class analysis.

Three things this module refuses to do, each because getting them wrong would quietly
invalidate everything downstream:

**It does not pool data sources.** A `Corpus` carries one `domain`. EmojiHeroVR and the
older 2025 recordings are different devices, protocols and sampling rates; merging them into
one table would turn a protocol difference into an apparent person difference, which is the
exact effect this study exists to measure. Add them as separate corpora and compare the
results, never the rows.

**It does not treat one trial as several.** There is exactly one recording per participant
per emotion. Splitting a trial into segments produces correlated pieces, not repeats, so
nothing here offers a `repeats` view. Within-trial stability is available and is named as
such.

**It does not silently apply the self-report filter.** The filter removes 29 of 105
recordings and does so very unevenly -- one participant loses all seven. Callers ask for the
subset they want by name, and both are reported.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import au_names

DOMAIN = 'acted-local'

EMOTIONS: tuple[str, ...] = (
    'Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise',
)

DATA_DIR = Path(__file__).resolve().parent / 'data'

# De-identified exclusion decisions. The verbatim self-reports these were derived from are
# not version controlled -- see the header of the file itself. This file is OPTIONAL: it
# feeds only the clearly labelled sensitivity analysis, and the primary unfiltered acted
# analysis runs without it.
EXCLUSIONS_FILE = DATA_DIR / 'acted_exclusions_derived.csv'
SPLITS_FILE = DATA_DIR / 'acted_splits.csv'

# Head/tracking columns kept alongside the face channels. They are what makes a measurement
# explanation testable: a participant whose head moved throughout, or whose face tracking
# dropped, is a different story from one whose face simply did something unusual.
QUALITY_COLUMNS = (
    'HeadTracked', 'HeadX', 'HeadY', 'HeadZ', 'HeadPitch', 'HeadYaw', 'HeadRoll',
    'EyeGazeTracked', 'FaceValid',
)


class CorpusError(RuntimeError):
    """Raised when the corpus on disk is not shaped the way the analysis assumes."""


@dataclass
class Recording:
    """One participant acting one emotion once."""

    domain: str
    participant: str
    emotion: str
    path: Path

    split: str                  # train | validation | test
    split_source: str           # named | assumed
    excluded: bool              # self-report did not name the prompted emotion
    exclusion_reason: str | None  # category, never the participant's own words

    times: np.ndarray           # [n] seconds since recording start
    au: np.ndarray              # [n, 70] float32, channel order == au_names.NAMES
    face_valid: np.ndarray      # [n] bool
    quality: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return len(self.times)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if self.frames > 1 else 0.0

    @property
    def sample_rate(self) -> float:
        """Frames divided by elapsed time -- the rate that matters.

        The capture is nominally 60 Hz and the teacher pipeline averages every 6 rows to
        reach 10 Hz, which only works out because this average rate really is 60.

        Use this rather than the median inter-frame interval, which reads about 71 Hz on
        every recording here. The capture runs on Unity's update loop, so most frames arrive
        ~14 ms apart and a minority arrive much later; the median ignores the late ones and
        overstates the rate by 18%. Reporting it as "the sample rate" made the corpus look
        inconsistent with its own documentation when it is not.
        """
        return float(self.frames / self.duration) if self.duration > 0 else 0.0

    @property
    def median_interval_rate(self) -> float:
        """1 / median inter-frame interval. Kept as a jitter diagnostic, not a rate."""
        if self.frames < 2:
            return 0.0
        return float(1.0 / np.median(np.diff(self.times)))

    @property
    def valid_au(self) -> np.ndarray:
        """Face channels for frames the tracker vouched for.

        Frames with `FaceValid == 0` carry whatever was last written, not a reading, so they
        are dropped rather than analysed. How many were dropped is itself a measurement
        signal -- see `tracking_rate`.
        """
        return self.au[self.face_valid]

    @property
    def tracking_rate(self) -> float:
        return float(self.face_valid.mean()) if self.frames else 0.0

    def key(self) -> tuple[str, str]:
        return (self.participant, self.emotion)


@dataclass
class Corpus:
    """Every recording from one data source. Never more than one source."""

    domain: str
    root: Path
    recordings: list[Recording]
    warnings: list[str] = field(default_factory=list)

    @property
    def participants(self) -> list[str]:
        return sorted({r.participant for r in self.recordings})

    def subset(self, *, self_report_consistent: bool = False,
               splits: tuple[str, ...] | None = None) -> list[Recording]:
        """Recordings matching a filter.

        `self_report_consistent=True` is the E2 subset; the default is E1, everything the
        participant was prompted to act. These are different estimands and the report keeps
        them apart -- see SUBJECT_VARIABILITY_REPORT.md.
        """
        out = self.recordings
        if self_report_consistent:
            out = [r for r in out if not r.excluded]
        if splits is not None:
            out = [r for r in out if r.split in splits]
        return out

    def by_participant(self, **kwargs) -> dict[str, list[Recording]]:
        grouped: dict[str, list[Recording]] = {}
        for r in self.subset(**kwargs):
            grouped.setdefault(r.participant, []).append(r)
        return grouped

    def coverage(self, **kwargs) -> dict[str, set[str]]:
        """participant -> emotions present. The check E2 exists to make visible."""
        return {p: {r.emotion for r in rs}
                for p, rs in self.by_participant(**kwargs).items()}


# ------------------------------------------------------------------ metadata


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a data file, skipping the `#` comment block at the top."""
    if not path.is_file():
        raise CorpusError(f'missing data file: {path}')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        lines = [line for line in handle if not line.lstrip().startswith('#')]
    return list(csv.DictReader(lines))


def load_exclusions(path: Path = EXCLUSIONS_FILE) -> dict[tuple[str, str], str]:
    """(participant, emotion) -> why that recording is excluded.

    The value is a category, never the participant's own words: the verbatim self-reports
    are not version controlled and the analysis does not need them, only which recordings
    are excluded.

    Returns an empty mapping when the file is absent, rather than raising. The exclusions
    feed one clearly labelled sensitivity analysis; the primary acted analysis treats all
    recordings as its set, so a missing optional file must not stop it. A caller that
    needs the sensitivity analysis checks for an empty result and says so.
    """
    if not path.is_file():
        return {}

    return {
        (row['participant_id'].strip(), row['emotion'].strip()):
            row['reason_category'].strip()
        for row in _read_csv_rows(path)
        if row['excluded'].strip() not in ('', '0', 'false', 'False')
    }


def load_splits(path: Path = SPLITS_FILE) -> dict[str, tuple[str, str]]:
    """participant -> (split, how the split was established)."""
    return {
        row['participant'].strip(): (row['split'].strip(), row['source'].strip())
        for row in _read_csv_rows(path)
    }


# ------------------------------------------------------------------- loading


def _load_recording(path: Path, participant: str, emotion: str,
                    exclusions: dict, splits: dict) -> Recording:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row]

    if 'FaceValid' not in header:
        raise CorpusError(f'{path}: no FaceValid column')

    face_start = header.index('FaceValid') + 1
    channels = tuple(header[face_start:])
    if channels != au_names.NAMES:
        raise CorpusError(
            f'{path}: face channels do not match au_names.NAMES '
            f'({len(channels)} columns, first mismatch at '
            f'{next((i for i, (a, b) in enumerate(zip(channels, au_names.NAMES)) if a != b), "length")})'
        )

    index = {name: i for i, name in enumerate(header)}

    def column(name: str) -> np.ndarray:
        i = index[name]
        return np.array([float(row[i]) if row[i] else np.nan for row in rows], np.float64)

    times = column('RealtimeSeconds')
    times = times - times[0]
    au = np.array([[float(v) if v else 0.0 for v in row[face_start:]] for row in rows],
                  np.float32)
    face_valid = column('FaceValid').astype(bool)

    # The Emotion column repeats the folder name on every row. Disagreement means the
    # recording was filed under the wrong prompt, which would mislabel it in every result.
    emotion_col = index.get('Emotion')
    if emotion_col is not None:
        labelled = {row[emotion_col].strip() for row in rows if row[emotion_col].strip()}
        if labelled and labelled != {emotion}:
            raise CorpusError(
                f'{path}: folder says {emotion} but the Emotion column says '
                f'{sorted(labelled)}'
            )

    split, split_source = splits.get(participant, ('unknown', 'missing'))

    return Recording(
        domain=DOMAIN,
        participant=participant,
        emotion=emotion,
        path=path,
        split=split,
        split_source=split_source,
        excluded=(participant, emotion) in exclusions,
        exclusion_reason=exclusions.get((participant, emotion)),
        times=times,
        au=au,
        face_valid=face_valid,
        quality={name: column(name) for name in QUALITY_COLUMNS if name in index},
    )


def load_corpus(root: str | Path,
                exclusions_file: Path = EXCLUSIONS_FILE,
                splits_file: Path = SPLITS_FILE) -> Corpus:
    """Read every `<participant>/<Emotion>/acting.csv` under `root`."""
    root = Path(root)
    if not root.is_dir():
        raise CorpusError(f'not a directory: {root}')

    exclusions = load_exclusions(exclusions_file)
    splits = load_splits(splits_file)

    recordings: list[Recording] = []
    warnings: list[str] = []

    for participant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        participant = participant_dir.name
        if participant not in splits:
            warnings.append(
                f'{participant}: no split recorded in {splits_file.name}; '
                'its results cannot be labelled train/validation/test'
            )

        found = set()
        for emotion_dir in sorted(p for p in participant_dir.iterdir() if p.is_dir()):
            csv_path = emotion_dir / 'acting.csv'
            if not csv_path.is_file():
                warnings.append(f'{participant}/{emotion_dir.name}: no acting.csv')
                continue
            recordings.append(
                _load_recording(csv_path, participant, emotion_dir.name, exclusions, splits)
            )
            found.add(emotion_dir.name)

        missing = set(EMOTIONS) - found
        if missing:
            warnings.append(f'{participant}: no recording for {", ".join(sorted(missing))}')

    if not recordings:
        raise CorpusError(f'no acting.csv files under {root}')

    # An exclusion naming something that is not in the corpus means the list and the data
    # have drifted apart, and the filter is then not doing what the report claims.
    present = {r.key() for r in recordings}
    for key in exclusions:
        if key not in present:
            warnings.append(
                f'exclusion list names {key[0]}/{key[1]}, which is not in the corpus'
            )

    return Corpus(domain=DOMAIN, root=root, recordings=recordings, warnings=warnings)


def find_corpus_root(explicit: str | Path | None = None) -> Path:
    """Locate ActingRecordings without a path typed into the script."""
    tried: list[Path] = []

    def candidate(path: Path) -> Path | None:
        tried.append(path)
        return path if (path / '.').is_dir() and any(path.glob('*/*/acting.csv')) else None

    if explicit:
        path = Path(explicit).expanduser()
        if found := candidate(path):
            return found
        raise CorpusError(f'no ActingRecordings corpus at {path}')

    import os
    if env := os.environ.get('ACTING_RECORDINGS'):
        if found := candidate(Path(env).expanduser()):
            return found

    home = Path.home()
    for base in (home / 'multimodal-vr-emotion-study' / 'multimodal-vr-emotion-study-main',
                 home / 'multimodal-vr-emotion-study',
                 home):
        if base.is_dir():
            if found := candidate(base / 'ActingRecordings'):
                return found

    listing = '\n'.join(f'  {p}' for p in tried)
    raise CorpusError(
        'could not find the ActingRecordings corpus. Looked in:\n'
        f'{listing}\n'
        'Pass --data PATH, or set ACTING_RECORDINGS.'
    )
