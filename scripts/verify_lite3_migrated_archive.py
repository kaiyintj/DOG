#!/usr/bin/env python3
"""Validate the deliberately compact B-disk Lite3 migration archive."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ALGORITHM_SHA = "d27c1032f97d8e744c3ee2f2ef196c00ea6bac7e"
PASS_STATUS = "ALGORITHM_STATIC_PASS_NON_GEOMETRIC"
CALIBRATION_STATUS = "INTRINSICS_FROM_BAG_EXTRINSICS_UNVERIFIED"
EXPECTED_OMISSIONS = frozenset({
    "merged/merged_0.db3",
    "merged/metadata.yaml",
    "output_bag/metadata.yaml",
    "output_bag/output_bag_0.db3",
})
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class MigrationValidationError(RuntimeError):
    """Report an incomplete, changed, or unexpectedly expanded archive."""


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_relative_path(value, ledger_path):
    candidate = Path(value)
    if candidate.is_absolute():
        raise MigrationValidationError(
            "{} contains absolute path {!r}".format(ledger_path, value))
    parts = tuple(part for part in candidate.parts if part != ".")
    if not parts or ".." in parts:
        raise MigrationValidationError(
            "{} contains unsafe path {!r}".format(ledger_path, value))
    return Path(*parts).as_posix()


def _read_ledger(ledger_path, allow_absolute=False):
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MigrationValidationError(
            "cannot read {}: {}".format(ledger_path, error)) from error

    entries = {}
    for line_number, line in enumerate(lines, start=1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not DIGEST_PATTERN.fullmatch(fields[0]):
            raise MigrationValidationError(
                "{}:{} is not a SHA-256 ledger entry".format(
                    ledger_path, line_number))
        raw_path = fields[1]
        candidate = Path(raw_path)
        if candidate.is_absolute() and allow_absolute:
            key = str(candidate)
        else:
            key = _normal_relative_path(raw_path, ledger_path)
        if key in entries:
            raise MigrationValidationError(
                "{} contains duplicate path {!r}".format(
                    ledger_path, key))
        entries[key] = fields[0]
    if not entries:
        raise MigrationValidationError(
            "{} contains no entries".format(ledger_path))
    return entries


def _check_hashes(entries, base_directory, label, omit=frozenset()):
    for entry, expected_digest in sorted(entries.items()):
        if entry in omit:
            continue
        candidate = Path(entry)
        path = candidate if candidate.is_absolute() else base_directory / entry
        if not path.is_file():
            raise MigrationValidationError(
                "{} file is missing: {}".format(label, path))
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise MigrationValidationError(
                "{} hash differs for {}: expected={}, actual={}".format(
                    label, path, expected_digest, actual_digest))


def _check_manifest(run_directory):
    manifest_path = run_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MigrationValidationError(
            "cannot read {}: {}".format(manifest_path, error)) from error

    expected = {
        "git_sha": ALGORITHM_SHA,
        "git_dirty": False,
        "overall": PASS_STATUS,
        "calibration_status": CALIBRATION_STATUS,
        "motion_ready": False,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise MigrationValidationError(
                "manifest {} expected {!r}, got {!r}".format(
                    key, expected_value, manifest.get(key)))
    overall_path = run_directory / "OVERALL"
    try:
        overall = overall_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise MigrationValidationError(
            "cannot read {}: {}".format(overall_path, error)) from error
    if overall != PASS_STATUS:
        raise MigrationValidationError(
            "OVERALL expected {!r}, got {!r}".format(
                PASS_STATUS, overall))


def _check_artifact_set(run_directory, entries):
    ledger_paths = set(entries)
    missing_from_ledger = EXPECTED_OMISSIONS - ledger_paths
    if missing_from_ledger:
        raise MigrationValidationError(
            "artifact ledger lacks expected full-run paths: {}".format(
                sorted(missing_from_ledger)))

    actual_files = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file() and path.name != "artifact_sha256.txt"
    }
    expected_files = ledger_paths - EXPECTED_OMISSIONS
    missing = expected_files - actual_files
    unexpected = actual_files - expected_files
    if missing or unexpected:
        raise MigrationValidationError(
            "compact artifact set differs: missing={}, unexpected={}".format(
                sorted(missing), sorted(unexpected)))

    present_omissions = EXPECTED_OMISSIONS & actual_files
    if present_omissions:
        raise MigrationValidationError(
            "compact archive contains excluded payloads: {}".format(
                sorted(present_omissions)))


def validate_migration(source_directory, run_directory, project_root):
    """Validate the archive through its two-directory interface."""
    source_directory = Path(source_directory).expanduser().resolve()
    run_directory = Path(run_directory).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    for label, path in (
            ("source directory", source_directory),
            ("run directory", run_directory),
            ("project root", project_root)):
        if not path.is_dir():
            raise MigrationValidationError(
                "{} does not exist: {}".format(label, path))

    _check_manifest(run_directory)

    source_entries = _read_ledger(run_directory / "source_sha256.txt")
    _check_hashes(source_entries, source_directory, "source")

    artifact_entries = _read_ledger(
        run_directory / "artifact_sha256.txt")
    _check_artifact_set(run_directory, artifact_entries)
    _check_hashes(
        artifact_entries,
        run_directory,
        "retained artifact",
        omit=EXPECTED_OMISSIONS,
    )

    implementation_entries = _read_ledger(
        run_directory / "implementation_sha256.txt",
        allow_absolute=True,
    )
    _check_hashes(
        implementation_entries, project_root, "implementation")

    return {
        "source_hashes": "PASS",
        "retained_artifacts": "PASS",
        "expected_omissions": "PASS",
        "implementation_hashes": "PASS",
        "overall": "B_DISK_COMPACT_ARCHIVE_PASS",
    }


def parse_args(argv=None):
    """Parse the two archive paths exposed by the command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    """Validate one migrated archive and print stable status lines."""
    arguments = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        result = validate_migration(
            arguments.source_dir, arguments.run_dir, project_root)
    except MigrationValidationError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1

    print("SOURCE_HASHES={}".format(result["source_hashes"]))
    print("RETAINED_ARTIFACTS={}".format(
        result["retained_artifacts"]))
    print("EXPECTED_OMISSIONS={}".format(
        result["expected_omissions"]))
    print("IMPLEMENTATION_HASHES={}".format(
        result["implementation_hashes"]))
    print("OVERALL={}".format(result["overall"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
