"""Tests for the compact Lite3 migration validator."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_lite3_migrated_archive.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_lite3_migrated_archive", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ledger(path, entries):
    path.write_text(
        "".join("{}  {}\n".format(digest, name)
                for name, digest in sorted(entries.items())),
        encoding="utf-8",
    )


def _archive(tmp_path):
    source = tmp_path / "source"
    run = tmp_path / "run"
    project = tmp_path / "project"
    source.mkdir()
    run.mkdir()
    project.mkdir()

    source_file = source / "camera.db3"
    source_file.write_bytes(b"source")
    _write_ledger(
        run / "source_sha256.txt",
        {"camera.db3": _digest(source_file)},
    )

    implementation = project / "core.py"
    implementation.write_text("VALUE = 1\n", encoding="utf-8")
    _write_ledger(
        run / "implementation_sha256.txt",
        {"core.py": _digest(implementation)},
    )

    manifest = {
        "git_sha": MODULE.ALGORITHM_SHA,
        "git_dirty": False,
        "overall": MODULE.PASS_STATUS,
        "calibration_status": MODULE.CALIBRATION_STATUS,
        "motion_ready": False,
    }
    (run / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    (run / "OVERALL").write_text(
        MODULE.PASS_STATUS + "\n", encoding="utf-8")
    notes = run / "notes.txt"
    notes.write_text("retained\n", encoding="utf-8")

    artifact_entries = {
        "OVERALL": _digest(run / "OVERALL"),
        "implementation_sha256.txt": _digest(
            run / "implementation_sha256.txt"),
        "manifest.json": _digest(run / "manifest.json"),
        "notes.txt": _digest(notes),
        "source_sha256.txt": _digest(run / "source_sha256.txt"),
    }
    artifact_entries.update({
        omission: hashlib.sha256(omission.encode()).hexdigest()
        for omission in MODULE.EXPECTED_OMISSIONS
    })
    _write_ledger(run / "artifact_sha256.txt", artifact_entries)
    return source, run, project


def test_valid_compact_archive_passes(tmp_path):
    """Accept an archive containing every retained ledger entry."""
    source, run, project = _archive(tmp_path)

    result = MODULE.validate_migration(source, run, project)

    assert result["overall"] == "B_DISK_COMPACT_ARCHIVE_PASS"


def test_missing_retained_artifact_fails(tmp_path):
    """Reject a retained ledger entry that disappeared after migration."""
    source, run, project = _archive(tmp_path)
    (run / "notes.txt").unlink()

    with pytest.raises(
            MODULE.MigrationValidationError, match="missing="):
        MODULE.validate_migration(source, run, project)


def test_unexpected_artifact_fails(tmp_path):
    """Reject an unrecorded file in the supposedly exact compact archive."""
    source, run, project = _archive(tmp_path)
    (run / "extra.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(
            MODULE.MigrationValidationError, match="unexpected="):
        MODULE.validate_migration(source, run, project)


def test_changed_retained_artifact_fails(tmp_path):
    """Reject retained content that no longer matches its ledger."""
    source, run, project = _archive(tmp_path)
    (run / "notes.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(
            MODULE.MigrationValidationError, match="hash differs"):
        MODULE.validate_migration(source, run, project)
