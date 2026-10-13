#!/usr/bin/env python3
"""Names for the 70 face-tracking channels, and the groupings the analysis reads them by.

Every result in this folder must name its channels. "AU 22 moved" is not a finding;
"InnerBrowRaiserL moved" is. The live session logs store the channels as bare indices
(`frame_<f>_au_<a>`), so something has to supply the names, and that something has to be
verifiable rather than typed from memory.

Two independent sources agree on the order, which is what makes this safe:

  1. `OVRPlugin.FaceExpression2` in the Meta XR SDK -- the enum the runtime reads.
  2. The `acting.csv` header in the ActingRecordings corpus, which writes the channels out
     under their own names.

`test_subject_variability.py` asserts both against the list below whenever those files are
present on the machine. That check is what licenses applying these names to the live logs'
numeric indices; without it the index-to-name mapping would be an assumption.

The names here are the CSV spelling (no underscores), which is the form the acted corpus
uses. `from_enum_name` converts the SDK spelling.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Channel names in index order 0-69, taken from the acting.csv header.
NAMES: tuple[str, ...] = (
    'BrowLowererL', 'BrowLowererR', 'CheekPuffL', 'CheekPuffR', 'CheekRaiserL',
    'CheekRaiserR', 'CheekSuckL', 'CheekSuckR', 'ChinRaiserB', 'ChinRaiserT',
    'DimplerL', 'DimplerR', 'EyesClosedL', 'EyesClosedR', 'EyesLookDownL',
    'EyesLookDownR', 'EyesLookLeftL', 'EyesLookLeftR', 'EyesLookRightL', 'EyesLookRightR',
    'EyesLookUpL', 'EyesLookUpR', 'InnerBrowRaiserL', 'InnerBrowRaiserR', 'JawDrop',
    'JawSidewaysLeft', 'JawSidewaysRight', 'JawThrust', 'LidTightenerL', 'LidTightenerR',
    'LipCornerDepressorL', 'LipCornerDepressorR', 'LipCornerPullerL', 'LipCornerPullerR', 'LipFunnelerLB',
    'LipFunnelerLT', 'LipFunnelerRB', 'LipFunnelerRT', 'LipPressorL', 'LipPressorR',
    'LipPuckerL', 'LipPuckerR', 'LipStretcherL', 'LipStretcherR', 'LipSuckLB',
    'LipSuckLT', 'LipSuckRB', 'LipSuckRT', 'LipTightenerL', 'LipTightenerR',
    'LipsToward', 'LowerLipDepressorL', 'LowerLipDepressorR', 'MouthLeft', 'MouthRight',
    'NoseWrinklerL', 'NoseWrinklerR', 'OuterBrowRaiserL', 'OuterBrowRaiserR', 'UpperLidRaiserL',
    'UpperLidRaiserR', 'UpperLipRaiserL', 'UpperLipRaiserR', 'TongueTipInterdental', 'TongueTipAlveolar',
    'TongueFrontDorsalPalate', 'TongueMidDorsalPalate', 'TongueBackDorsalVelar', 'TongueOut', 'TongueRetreat',
)

CHANNEL_COUNT = len(NAMES)

INDEX: dict[str, int] = {name: i for i, name in enumerate(NAMES)}

# The older `OVRPlugin.FaceExpression` enum stops at 63 and is index-identical for 0-62.
# EmojiHeroVR reports those 63 channels, so when that corpus is added as a separate domain,
# its data occupies indices 0-62 and indices 63-69 -- the seven tongue channels -- have no
# measurement behind them. They must be treated as absent, not as zeros: a zero here would
# read as "the tongue did not move" when the truth is "the tongue was never observed".
V1_CHANNEL_COUNT = 63
TONGUE_INDICES: tuple[int, ...] = tuple(range(63, 70))


def from_enum_name(enum_name: str) -> str:
    """`Brow_Lowerer_L` -> `BrowLowererL`."""
    return enum_name.replace('_', '')


def name(index: int) -> str:
    return NAMES[index]


def indices_matching(pattern: str) -> tuple[int, ...]:
    """Channel indices whose name matches a regex, case-insensitively."""
    rx = re.compile(pattern, re.IGNORECASE)
    return tuple(i for i, n in enumerate(NAMES) if rx.search(n))


# ------------------------------------------------------------------ groupings

# Functional regions. A per-channel table of 70 rows is unreadable; the question "does this
# person use their brow where that one uses their mouth" is asked at this level.
#
# `eye_gaze` is separated from `eye` deliberately. Gaze direction is where someone is
# looking, not how their face is set, and pooling it into an expression summary would let
# looking around the scene register as expressiveness.
REGIONS: dict[str, tuple[int, ...]] = {
    'brow':     indices_matching(r'BrowLowerer|InnerBrowRaiser|OuterBrowRaiser'),
    'eye':      indices_matching(r'EyesClosed|LidTightener|UpperLidRaiser'),
    'eye_gaze': indices_matching(r'EyesLook'),
    'cheek':    indices_matching(r'Cheek'),
    'nose':     indices_matching(r'NoseWrinkler'),
    'mouth':    indices_matching(r'Lip|Mouth|Dimpler'),
    'jaw':      indices_matching(r'Jaw'),
    'chin':     indices_matching(r'ChinRaiser'),
    'tongue':   indices_matching(r'Tongue'),
}

# Channels that describe an expression, i.e. everything except gaze direction and the tongue
# channels (which the headset reports but which no emotion prompt in this study targets, and
# which EmojiHeroVR does not measure at all).
EXPRESSION_INDICES: tuple[int, ...] = tuple(
    i for i in range(CHANNEL_COUNT)
    if i not in set(REGIONS['eye_gaze']) | set(REGIONS['tongue'])
)


def _bilateral_pairs() -> tuple[tuple[int, int], ...]:
    """(left, right) index pairs for channels that come in mirrored halves.

    Used as a measurement diagnostic. A face is roughly symmetric, so a channel pair that
    is persistently lopsided across every emotion points at headset fit or camera coverage
    rather than at how the person expresses anything -- which is one of the few handles
    this data gives on separating sensor effects from expression style.
    """
    pairs = []
    for i, n in enumerate(NAMES):
        if n.endswith('L') and not n.endswith('LB') and not n.endswith('LT'):
            partner = n[:-1] + 'R'
            if partner in INDEX:
                pairs.append((i, INDEX[partner]))
    # The lip funneler and lip suck channels are quartered LB/LT/RB/RT rather than L/R.
    for stem in ('LipFunneler', 'LipSuck'):
        for half in ('B', 'T'):
            left, right = f'{stem}L{half}', f'{stem}R{half}'
            if left in INDEX and right in INDEX:
                pairs.append((INDEX[left], INDEX[right]))
    # MouthLeft/MouthRight and JawSidewaysLeft/Right spell the side out in full.
    for left, right in (('MouthLeft', 'MouthRight'),
                        ('JawSidewaysLeft', 'JawSidewaysRight')):
        pairs.append((INDEX[left], INDEX[right]))
    return tuple(sorted(pairs))


BILATERAL_PAIRS: tuple[tuple[int, int], ...] = _bilateral_pairs()


# -------------------------------------------------------------- verification

def names_from_csv_header(path: str | Path) -> tuple[str, ...]:
    """The channel names an `acting.csv` declares, in file order."""
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle))
    return tuple(header[header.index('FaceValid') + 1:])


def names_from_sdk_enum(path: str | Path) -> tuple[str, ...]:
    """Channel names parsed from `OVRPlugin.FaceExpression2`, in declared index order."""
    source = Path(path).read_text(encoding='utf-8', errors='replace')
    match = re.search(r'public enum FaceExpression2\s*\{(.*?)\n\s*\}', source, re.S)
    if not match:
        raise ValueError(f'no FaceExpression2 enum in {path}')

    by_index: dict[int, str] = {}
    for line in match.group(1).splitlines():
        entry = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+)', line)
        if entry and entry.group(1) not in ('Max', 'Invalid'):
            index = int(entry.group(2))
            if index >= 0:
                by_index[index] = from_enum_name(entry.group(1))

    return tuple(by_index[i] for i in sorted(by_index))
