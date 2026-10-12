#!/usr/bin/env python3
"""Convert the frozen EmojiHero population profile into the artifact Unity loads.

    python Analysis/export_population_artifact_for_unity.py           # write it
    python Analysis/export_population_artifact_for_unity.py --check   # verify, write nothing

Why a conversion exists at all
------------------------------
`Analysis/results/emojihero_population_centroids.json` is the analysis artifact: a dictionary
keyed by emotion name, wrapped in provenance, sized for a reader with pandas. Unity has no
dictionary-shaped deserialiser, and the runtime needs a shape it can validate field by field
rather than one it has to trust.

Nothing numeric changes. The seven seven-dimensional vectors are carried across as the
identical doubles -- Python writes them with `repr`, which round-trips exactly, and the C#
reader parses them back to the same bits. `--check` proves it: it re-reads what it wrote and
compares bit patterns, not decimal strings.

The two hashes, and why there are two
-------------------------------------
`semantic_sha256` is the preregistration pin, computed by `emojihero_calibration_study` over
the centroids and the whole feature configuration. It answers "is this the profile frozen
before the test split was opened". It is carried through unchanged and the C# compares it
against a compiled-in constant.

`conversion_sha256` answers a different question -- "does this converted file still hold what
the exporter put in it" -- and it exists because the first hash cannot be recomputed in C#.
Its canonical form serialises every double as an IEEE-754 bit pattern in hex, so Python's
`struct.pack` and .NET's `BitConverter.DoubleToInt64Bits` produce byte-identical input with no
float formatting, locale or exponent notation anywhere in the path.

`PopulationProfile.ConversionHash` in
`Assets/Scripts/Calibration/PopulationProfile.cs` builds the same string. If the two ever
drift, `Analysis/CsharpTests` fails on a golden vector rather than the headset failing on a
participant.

What is not exported
--------------------
No EmojiHero raw data. The profile is aggregate model *output* -- seven vectors averaged over
twenty training participants -- carrying no action units, no face windows and no individual's
centroid. The per-participant support breakdown is dropped too: it is provenance for a human
reading the analysis artifact, not something the runtime validates, and a build has no reason
to ship a per-person sequence count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import session_analysis as sa  # noqa: E402

UNITY_SCHEMA_VERSION = "emojihero-population-centroids-unity-v1"
CONVERSION_HASH_PREAMBLE = "emojihero-population-centroids-unity-conversion-v1"

DEFAULT_SOURCE = Path("Analysis/results/emojihero_population_centroids.json")
DEFAULT_DEST = Path("Assets/StreamingAssets/emojihero_population_centroids_unity.json")

METRIC = "cosine"


def bits(value: float) -> str:
    """The IEEE-754 bit pattern of a double, as sixteen lowercase hex digits.

    The cross-language contract. `BitConverter.DoubleToInt64Bits(v).ToString("x16")` in C#
    returns this same string for every finite double, including negative ones -- the centroid
    values are all negative, being logs of probabilities, so that case is the normal one here
    rather than an edge case.
    """
    return struct.pack(">d", float(value)).hex()


def conversion_hash(payload: dict) -> str:
    """The canonical form `PopulationProfile.ConversionHash` reproduces, field for field."""
    lines = [
        CONVERSION_HASH_PREAMBLE,
        f"schema_version={payload['schema_version']}",
        f"source_schema_version={payload['source_schema_version']}",
        f"semantic_sha256={payload['semantic_sha256']}",
        f"source_split={payload['source_split']}",
        "class_order=" + ",".join(payload["class_order"]),
        f"epsilon={bits(payload['epsilon'])}",
        f"feature_space={payload['feature_space']}",
        f"transform={payload['transform']}",
        f"estimator={payload['estimator']}",
        f"metric={payload['metric']}",
        f"model_sha256={payload['model_sha256']}",
    ]

    for row in payload["centroids"]:
        lines.append(
            f"centroid[{row['emotion']}]=" + ",".join(bits(v) for v in row["values"])
        )

    canonical = "\n".join(lines) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_payload(source: Path, model_path: Path) -> dict:
    """Read and fully validate the analysis artifact, then reshape it.

    Validation is `emojihero_calibration_study.load_population_centroids`, reused rather than
    reimplemented: schema version, model hash against the ONNX file actually on disk, source
    split, class order, epsilon, matrix shape, finiteness, and the semantic hash recomputed
    from the artifact's own contents. A file that fails any of those never reaches Unity.
    """
    import emojihero_calibration_study as eh

    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    matrix, artifact = eh.load_population_centroids(source, model_sha=model_sha)

    class_order = list(artifact["class_order"])
    centroids = [
        {"emotion": emotion, "values": [float(v) for v in matrix[index]]}
        for index, emotion in enumerate(class_order)
    ]

    payload = {
        "schema_version": UNITY_SCHEMA_VERSION,
        "source_schema_version": artifact["schema_version"],
        "semantic_sha256": artifact["semantic_sha256"],
        # Filled in below, once every hashed field is present.
        "conversion_sha256": "",
        "source_split": artifact["source_split"],
        "class_order": class_order,
        "epsilon": artifact["epsilon"],
        "feature_space": artifact["feature_space"],
        "transform": artifact["transform"],
        "estimator": artifact["estimator"],
        "metric": METRIC,
        "model_sha256": model_sha,
        "model_relative": artifact["model"]["relative"],
        "centroids": centroids,
        # Provenance for a human opening the file on the device. None of it is hashed and
        # none of it is validated, so editing it cannot change a prediction.
        "n_participants": artifact["n_participants"],
        "n_sequences": artifact["n_sequences"],
        "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "aggregate model-output representation over the EmojiHero training participants; "
            "carries no action units, no face windows and no per-person centroid. Estimated on "
            "EmojiHero captures, applied to live headset windows -- see "
            "Docs/VR_SHADOW_EVALUATION_REPORT.md for the measured direction of that mismatch."
        ),
    }

    payload["conversion_sha256"] = conversion_hash(payload)
    return payload


def verify_round_trip(payload: dict, text: str) -> None:
    """Re-read the serialised form and compare bit patterns.

    Decimal equality is not the question. The C# reader parses these literals with
    round-trip semantics, so what matters is that the written text recovers the identical
    double -- which is what makes the conversion hash agree across the two languages.
    """
    reloaded = json.loads(text)

    for written, read_back in zip(payload["centroids"], reloaded["centroids"]):
        if written["emotion"] != read_back["emotion"]:
            raise SystemExit(
                f"round-trip changed an emotion label: {written['emotion']} -> {read_back['emotion']}"
            )

        for index, (a, b) in enumerate(zip(written["values"], read_back["values"])):
            if bits(a) != bits(b):
                raise SystemExit(
                    f"round-trip changed {written['emotion']}[{index}]:\n"
                    f"  written: {bits(a)}\n"
                    f"  read:    {bits(b)}"
                )

    if conversion_hash(reloaded) != payload["conversion_sha256"]:
        raise SystemExit("the conversion hash does not survive a round trip through JSON")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None,
                        help=f"analysis artifact (default: {DEFAULT_SOURCE})")
    parser.add_argument("--dest", default=None,
                        help=f"Unity artifact to write (default: {DEFAULT_DEST})")
    parser.add_argument("--model", default=None,
                        help="the ONNX teacher, hashed as the provenance pin")
    parser.add_argument("--check", action="store_true",
                        help="verify the existing artifact and write nothing")
    args = parser.parse_args(argv)

    root = sa.find_repo_root() or Path.cwd()
    source = Path(args.source) if args.source else root / DEFAULT_SOURCE
    dest = Path(args.dest) if args.dest else root / DEFAULT_DEST
    model = Path(args.model) if args.model else root / "Packages" / Path(
        "com.vcu-swim-lab.vr-emotion-detection/Runtime/Models/sparseVR_teacher_model.onnx")

    if not source.exists():
        raise SystemExit(f"no population artifact at\n  {source}")
    if not model.exists():
        raise SystemExit(f"no ONNX teacher at\n  {model}")

    payload = build_payload(source, model)
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    verify_round_trip(payload, text)

    print(f"source            {source}")
    print(f"model sha256      {payload['model_sha256']}")
    print(f"semantic sha256   {payload['semantic_sha256']}")
    print(f"conversion sha256 {payload['conversion_sha256']}")
    print(f"centroids         {len(payload['centroids'])} x "
          f"{len(payload['centroids'][0]['values'])}, class order "
          f"{', '.join(payload['class_order'])}")

    if args.check:
        if not dest.exists():
            raise SystemExit(f"--check: nothing at\n  {dest}\nRun without --check to write it.")

        existing = json.loads(dest.read_text(encoding="utf-8"))
        problems = []
        if existing.get("semantic_sha256") != payload["semantic_sha256"]:
            problems.append(
                f"  semantic:   {existing.get('semantic_sha256')}\n"
                f"  expected:   {payload['semantic_sha256']}")
        if existing.get("conversion_sha256") != conversion_hash(existing):
            problems.append("  the file's conversion hash does not match its own contents")
        if existing.get("conversion_sha256") != payload["conversion_sha256"]:
            problems.append(
                f"  conversion: {existing.get('conversion_sha256')}\n"
                f"  expected:   {payload['conversion_sha256']}")

        if problems:
            raise SystemExit("the artifact in the build does not match the source:\n"
                             + "\n".join(problems))

        print(f"\nOK: {dest} matches the source artifact.")
        return 0

    if dest.exists():
        existing = json.loads(dest.read_text(encoding="utf-8"))
        if existing.get("conversion_sha256") == payload["conversion_sha256"]:
            print(f"\n{dest}\nalready holds this profile; rewriting it anyway so the "
                  "export timestamp reflects this run.")
        else:
            print(f"\nreplacing the artifact in the build:\n"
                  f"  was: {existing.get('conversion_sha256')}\n"
                  f"  now: {payload['conversion_sha256']}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Newlines pinned to LF rather than left to `os.linesep`. The artifact is checked into the
    # repository and read by a C# parser on a headset; an exporter that emits CRLF on the
    # operator's Windows machine and LF elsewhere would produce two byte-different files
    # holding the same profile, for no reason anybody could see in a diff.
    with open(dest, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)

    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
