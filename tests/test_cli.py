"""
Tests for CLI commands.
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from sqlalchemy.exc import ArgumentError

from garmin_health_data.cli import extract, verify
from garmin_health_data.db import create_tables, get_session
from garmin_health_data.lifecycle import acquire_lock, setup_lifecycle_dirs


def test_raw_sql_string_raises_error(tmp_path):
    """
    Test that passing a raw SQL string to session.execute() raises an error.

    SQLAlchemy 2.x requires raw SQL to be wrapped with text(). This test captures the
    original bug where PRAGMA integrity_check was passed as a plain string, causing
    ArgumentError.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    with get_session(str(db_path)) as session:
        with pytest.raises(
            ArgumentError, match="should be explicitly declared as text"
        ):
            session.execute("PRAGMA integrity_check")


def test_verify_runs_integrity_check(tmp_path):
    """
    Test that verify command executes PRAGMA integrity_check successfully.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    runner = CliRunner()
    result = runner.invoke(verify, ["--db-path", str(db_path)])

    assert result.exit_code == 0
    assert "Database integrity check passed" in result.output


@pytest.mark.parametrize("error_count", [1, 2])
def test_verify_reports_integrity_errors(tmp_path: Path, error_count: int) -> None:
    """
    Report every integrity error and fail without changing the database.

    :param tmp_path: Temporary directory for the database.
    :param error_count: Number of invalid rows to insert.
    :return: None.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE integrity_fixture (value INTEGER CHECK (value > 0))"
        )
        # Persist invalid rows so a new connection detects real SQLite errors.
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.executemany(
            "INSERT INTO integrity_fixture VALUES (?)", [(0,)] * error_count
        )
    original_bytes = db_path.read_bytes()

    result = CliRunner().invoke(verify, ["--db-path", str(db_path)])

    assert result.output.count("CHECK constraint failed in integrity_fixture") == (
        error_count
    )
    assert "Database integrity check passed" not in result.output
    assert result.exit_code == 1
    assert db_path.read_bytes() == original_bytes


def test_verify_nonexistent_db(tmp_path):
    """
    Test that verify command rejects non-existent database path with a friendly message
    (rather than Click's raw "does not exist" path-validator error).
    """
    db_path = tmp_path / "nonexistent.db"

    runner = CliRunner()
    result = runner.invoke(verify, ["--db-path", str(db_path)])

    assert result.exit_code == 1
    assert "Database not found" in result.output
    assert "garmin extract" in result.output


# --------------------------------------------------------------------------------------
# Lifecycle integration tests for the extract command
# --------------------------------------------------------------------------------------


def _stub_extract_no_files(*args, **kwargs):
    """
    Return an empty extraction result; ingest_dir untouched.
    """
    return {
        "garmin_files": 0,
        "activity_files": 0,
        "failures": [],
        "failed_accounts": [],
    }


def _common_invoke(runner, db_path, *extra_args):
    """
    Invoke the extract command with a default narrow date range.
    """
    return runner.invoke(
        extract,
        [
            "--db-path",
            str(db_path),
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
            *extra_args,
        ],
    )


def test_extract_creates_lifecycle_dirs_next_to_db(tmp_path):
    """
    Extract command creates garmin_files/{ingest,process,storage,quarantine} next to the
    database file before extraction runs.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch(
            "garmin_health_data.cli.extract_data",
            side_effect=_stub_extract_no_files,
        ),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    base = tmp_path / "garmin_files"
    for name in ("ingest", "process", "storage", "quarantine"):
        assert (base / name).is_dir()


def test_extract_recovers_stale_process_files(tmp_path):
    """
    Files left in process/ from a previously crashed run are moved back to ingest/ at
    the start of the next extract run.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    base = tmp_path / "garmin_files"
    setup_lifecycle_dirs(base)
    stale = base / "process" / "stale.json"
    stale.write_text('{"old": true}')

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch(
            "garmin_health_data.cli.extract_data",
            side_effect=_stub_extract_no_files,
        ),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    # Recovery message printed.
    assert "Recovered 1 file(s)" in result.output
    # The stale file's content is preserved. With no GARMIN_FILE_TYPES
    # pattern matching its name, the pre-routing step archived it directly
    # to storage/ as backup-only — no longer left in process/ to loop on
    # subsequent runs.
    assert (base / "storage" / "stale.json").read_text() == '{"old": true}'
    assert not (base / "process" / "stale.json").exists()


def test_extract_uses_ingest_dir_for_extract_data(tmp_path):
    """
    extract_data is invoked with ingest_dir = garmin_files/ingest.
    """

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    captured = {}

    def capture_ingest(*args, **kwargs):
        captured["ingest_dir"] = kwargs["ingest_dir"]
        return _stub_extract_no_files()

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", side_effect=capture_ingest),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    assert captured["ingest_dir"] == tmp_path / "garmin_files" / "ingest"


# --------------------------------------------------------------------------------------
# Per-FileSet processing routes to storage / quarantine
# --------------------------------------------------------------------------------------


def test_process_loop_routes_success_to_storage_failure_to_quarantine(tmp_path):
    """
    With two FileSets in ingest/ where one will succeed and one will raise, the
    successful FileSet's files end up in storage/ and the failed FileSet's files end up
    in quarantine/.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    base = tmp_path / "garmin_files"
    setup_lifecycle_dirs(base)
    good = base / "ingest" / "user1_SLEEP_2025-01-01T12-00-00+00-00.json"
    bad = base / "ingest" / "user1_SLEEP_2025-01-02T12-00-00+00-00.json"
    good.write_text('{"ok": true}')
    bad.write_text('{"corrupt": true}')

    def stub_extract(*args, **kwargs):
        return {
            "garmin_files": 2,
            "activity_files": 0,
            "failures": [],
            "failed_accounts": [],
        }

    def stub_process(self, file_set, session):
        names = [p.name for p in file_set.file_paths]
        if any("2025-01-02" in n for n in names):
            raise RuntimeError("boom")

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", side_effect=stub_extract),
        patch(
            "garmin_health_data.processor.GarminProcessor.process_file_set",
            new=stub_process,
        ),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    assert (base / "storage" / good.name).exists()
    assert (base / "quarantine" / bad.name).exists()
    assert not good.exists()
    assert not bad.exists()


# --------------------------------------------------------------------------------------
# Failure summary printing
# --------------------------------------------------------------------------------------


def test_extract_prints_failure_summary(tmp_path):
    """
    End-of-run summary lists per-data-type failures from the extractor.
    """
    from garmin_health_data.extractor import ExtractionFailure

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    def stub_extract(*args, **kwargs):
        return {
            "garmin_files": 0,
            "activity_files": 0,
            "failures": [
                ExtractionFailure(
                    data_type="SLEEP",
                    date="2025-01-02",
                    activity_id="",
                    error="RuntimeError: hiccup",
                ),
                ExtractionFailure(
                    data_type="SLEEP",
                    date="2025-01-03",
                    activity_id="",
                    error="RuntimeError: again",
                ),
            ],
            "failed_accounts": [],
        }

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", side_effect=stub_extract),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    assert "Extraction failures" in result.output
    assert "SLEEP" in result.output
    assert "2025-01-02" in result.output


# --------------------------------------------------------------------------------------
# --extract-only and --process-only flags
# --------------------------------------------------------------------------------------


def test_extract_only_skips_processing(tmp_path):
    """
    --extract-only writes files into ingest/ and stops; no move to process/
    or storage/.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    def stub_extract(*args, **kwargs):
        ingest = kwargs["ingest_dir"]
        (ingest / "user1_SLEEP_2025-01-01T12-00-00+00-00.json").write_text("{}")
        return {
            "garmin_files": 1,
            "activity_files": 0,
            "failures": [],
            "failed_accounts": [],
        }

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", side_effect=stub_extract),
    ):
        result = _common_invoke(runner, db_path, "--extract-only")

    assert result.exit_code == 0, result.output
    base = tmp_path / "garmin_files"
    assert list((base / "ingest").iterdir())
    assert list((base / "process").iterdir()) == []
    assert list((base / "storage").iterdir()) == []
    assert "Extraction-only mode" in result.output


def test_process_only_skips_extraction(tmp_path):
    """
    --process-only does not call the extract API; it only processes whatever
    is in ingest/.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    base = tmp_path / "garmin_files"
    setup_lifecycle_dirs(base)
    (base / "ingest" / "user1_SLEEP_2025-01-01T12-00-00+00-00.json").write_text("{}")

    runner = CliRunner()
    mock_extract = MagicMock()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", mock_extract),
        patch("garmin_health_data.processor.GarminProcessor.process_file_set"),
    ):
        result = runner.invoke(
            extract,
            ["--db-path", str(db_path), "--process-only"],
        )

    assert result.exit_code == 0, result.output
    mock_extract.assert_not_called()


def test_extract_only_and_process_only_are_mutually_exclusive(tmp_path):
    """
    Passing both flags exits non-zero with a clear error message.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    runner = CliRunner()
    result = _common_invoke(runner, db_path, "--extract-only", "--process-only")

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


# --------------------------------------------------------------------------------------
# Concurrency lock
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fcntl-based lock is a no-op on Windows; concurrency test does not apply.",
)
def test_extract_aborts_when_lock_held(tmp_path):
    """
    A second concurrent extract aborts immediately with a clear message when the
    lifecycle lock is already held.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    base = tmp_path / "garmin_files"
    setup_lifecycle_dirs(base)

    runner = CliRunner()
    with (
        acquire_lock(base),
        patch("garmin_health_data.cli.ensure_authenticated"),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code != 0
    assert "Another garmin extract run is in progress" in result.output


# --------------------------------------------------------------------------------------
# Process-only does not require authentication
# --------------------------------------------------------------------------------------


def test_process_only_does_not_require_authentication(tmp_path):
    """
    Authentication is only required when extracting; --process-only must work without
    invoking ensure_authenticated.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    base = tmp_path / "garmin_files"
    setup_lifecycle_dirs(base)

    runner = CliRunner()
    auth_mock = MagicMock()
    with (
        patch("garmin_health_data.cli.ensure_authenticated", auth_mock),
        patch("garmin_health_data.processor.GarminProcessor.process_file_set"),
    ):
        result = runner.invoke(
            extract,
            ["--db-path", str(db_path), "--process-only"],
        )

    assert result.exit_code == 0, result.output
    auth_mock.assert_not_called()


def test_unmatched_files_routed_to_storage_as_backup(tmp_path):
    """
    Files that don't match any GARMIN_FILE_TYPES processor pattern (e.g. an unknown .xyz
    extension, or TCX/GPX/KML activity formats) are real Garmin data the user wanted
    extracted, just not data we can load.

    They go directly to storage/ as backup-only — NOT to quarantine, which is for
    genuine processing failures. Mirrors openetl's `store_format` skip-to-storage
    behavior.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    base = tmp_path / "garmin_files"
    setup_lifecycle_dirs(base)
    # Filename with timestamp so it groups into a FileSet, but with an
    # extension that no GARMIN_FILE_TYPES pattern matches.
    extra = base / "ingest" / "user1_UNKNOWN_2025-01-01T12-00-00+00-00.xyz"
    extra.write_text("backup-only data")

    def stub_extract(*args, **kwargs):
        return {
            "garmin_files": 1,
            "activity_files": 0,
            "failures": [],
            "failed_accounts": [],
        }

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", side_effect=stub_extract),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    # Backup-only file landed in storage, not quarantine.
    assert (base / "storage" / extra.name).exists()
    assert not (base / "quarantine" / extra.name).exists()
    assert not extra.exists()
    assert list((base / "process").iterdir()) == []


def test_extract_suppresses_date_range_line_when_window_empty(tmp_path):
    """
    On an up-to-date database the auto-detected start (last stored day + 1) lands after
    the default end (today), an empty window.

    The CLI must not print its "Date range" line for that no-op, while still delegating
    to extract_data, which owns the authoritative short-circuit.
    """
    from datetime import date

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    runner = CliRunner()
    # A far-future "last stored day" guarantees start > today's end regardless of the
    # wall clock, reproducing the up-to-date-database inverted window deterministically.
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.get_latest_date", return_value=date(2999, 1, 1)),
        patch(
            "garmin_health_data.cli.extract_data",
            side_effect=_stub_extract_no_files,
        ) as mock_extract,
    ):
        result = runner.invoke(extract, ["--db-path", str(db_path)])

    assert result.exit_code == 0, result.output
    # Auto-detect path ran, so execution reached the range-printing region...
    assert "Auto-detected start date:" in result.output
    # ...and the misleading backwards range line was suppressed.
    assert "Date range:" not in result.output
    # Delegation still happens; extract_data performs the real no-op short-circuit.
    mock_extract.assert_called_once()


def test_extract_prints_date_range_line_for_normal_window(tmp_path):
    """
    A normal forward window (start before end) still prints the "Date range" line.

    Guards the empty-window suppression from regressing into always-on.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch(
            "garmin_health_data.cli.extract_data",
            side_effect=_stub_extract_no_files,
        ),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    assert "Date range:" in result.output


def test_extract_labels_same_day_window_inclusive(tmp_path):
    """
    A same-day window (start == end) is extracted inclusively.

    The CLI labels it "single day, inclusive" rather than the multi-day "(exclusive)"
    wording, matching the documented same-day behavior in extract().
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch(
            "garmin_health_data.cli.extract_data",
            side_effect=_stub_extract_no_files,
        ),
    ):
        result = runner.invoke(
            extract,
            [
                "--db-path",
                str(db_path),
                "--start-date",
                "2025-01-01",
                "--end-date",
                "2025-01-01",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "single day, inclusive" in result.output
    assert "(exclusive)" not in result.output


def test_extract_exclude_data_types_extracts_all_but_excluded(tmp_path):
    """
    --exclude-data-types resolves to every registry type except the excluded one(s),
    handed to extract_data as an explicit list (forward-compatible: types added later
    are still included).
    """
    from garmin_health_data.constants import GARMIN_DATA_REGISTRY

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    captured = {}

    def capture(*args, **kwargs):
        captured["data_types"] = kwargs.get("data_types")
        return _stub_extract_no_files()

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", side_effect=capture),
    ):
        result = runner.invoke(
            extract,
            ["--db-path", str(db_path), "--exclude-data-types", "MENSTRUAL_CYCLE_DAY"],
        )

    assert result.exit_code == 0, result.output
    expected = [
        d.name
        for d in GARMIN_DATA_REGISTRY.all_data_types
        if d.name != "MENSTRUAL_CYCLE_DAY"
    ]
    assert captured["data_types"] == expected


def test_extract_rejects_data_types_with_exclude_data_types(tmp_path):
    """
    --data-types and --exclude-data-types are mutually exclusive; supplying both aborts
    before any extraction.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch(
            "garmin_health_data.cli.extract_data",
            side_effect=_stub_extract_no_files,
        ) as mock_extract,
    ):
        result = runner.invoke(
            extract,
            [
                "--db-path",
                str(db_path),
                "--data-types",
                "SLEEP",
                "--exclude-data-types",
                "STRESS",
            ],
        )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    mock_extract.assert_not_called()


def test_extract_rejects_unknown_exclude_data_type(tmp_path):
    """
    An unknown --exclude-data-types name aborts with an error naming the offender.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch(
            "garmin_health_data.cli.extract_data",
            side_effect=_stub_extract_no_files,
        ) as mock_extract,
    ):
        result = runner.invoke(
            extract,
            ["--db-path", str(db_path), "--exclude-data-types", "NOTAREALTYPE"],
        )

    assert result.exit_code != 0
    assert "NOTAREALTYPE" in result.output
    mock_extract.assert_not_called()


def test_extract_rejects_excluding_all_data_types(tmp_path):
    """
    Excluding every registry type leaves nothing to extract, which aborts.
    """
    from garmin_health_data.constants import GARMIN_DATA_REGISTRY

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    exclude_args = []
    for data_type in GARMIN_DATA_REGISTRY.all_data_types:
        exclude_args += ["--exclude-data-types", data_type.name]

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch(
            "garmin_health_data.cli.extract_data",
            side_effect=_stub_extract_no_files,
        ) as mock_extract,
    ):
        result = runner.invoke(extract, ["--db-path", str(db_path), *exclude_args])

    assert result.exit_code != 0
    assert "nothing left to extract" in result.output
    mock_extract.assert_not_called()


def test_extract_exclude_data_types_prints_exclusion_summary(tmp_path):
    """
    In exclude mode the CLI names the excluded types rather than dumping the full
    resolved allowlist of every other type.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch(
            "garmin_health_data.cli.extract_data",
            side_effect=_stub_extract_no_files,
        ),
    ):
        result = runner.invoke(
            extract,
            ["--db-path", str(db_path), "--exclude-data-types", "MENSTRUAL_CYCLE_DAY"],
        )

    assert result.exit_code == 0, result.output
    assert "Extracting all data types except: MENSTRUAL_CYCLE_DAY" in result.output


def test_extract_validates_exclude_options_before_auth(tmp_path):
    """
    Invalid exclude options abort before ensure_authenticated() (and before any DB or
    lock work), so a typo never triggers an auth prompt or database mutation.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated") as mock_auth,
        patch(
            "garmin_health_data.cli.extract_data",
            side_effect=_stub_extract_no_files,
        ),
    ):
        result = runner.invoke(
            extract,
            ["--db-path", str(db_path), "--exclude-data-types", "NOTAREALTYPE"],
        )

    assert result.exit_code != 0
    mock_auth.assert_not_called()


# --------------------------------------------------------------------------------------
# Gap ledger: record / resolve on extract, per-account backfill, gaps command
# --------------------------------------------------------------------------------------


def _make_failure(
    user_id, data_type, date_str, activity_id="", error="RuntimeError: boom"
):
    """
    Build an extractor ExtractionFailure record for ledger seeding.
    """
    from garmin_health_data.extractor import ExtractionFailure

    return ExtractionFailure(
        data_type=data_type,
        date=date_str,
        activity_id=activity_id,
        error=error,
        user_id=user_id,
    )


def _seed_ledger(db_path, failures):
    """
    Record failures directly into the gap ledger of a test database.
    """
    from garmin_health_data.db import record_extraction_failures

    record_extraction_failures(str(db_path), failures)


def _invoke_extract_auto_dates(runner, db_path):
    """
    Invoke extract without --start-date / --end-date (auto-detection path).
    """
    return runner.invoke(extract, ["--db-path", str(db_path)])


def test_extract_records_failures_to_ledger(tmp_path):
    """
    Failures returned by the extractor are persisted to the gap ledger.
    """
    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    def stub_extract(*args, **kwargs):
        return {
            "garmin_files": 0,
            "activity_files": 0,
            "failures": [
                _make_failure("11111111", "SLEEP", "2025-01-02"),
            ],
            "successes": [],
            "failed_accounts": [],
        }

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", side_effect=stub_extract),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    assert "gap ledger" in result.output

    from garmin_health_data.db import get_unresolved_failure_summary

    summary = get_unresolved_failure_summary(str(db_path))
    assert len(summary) == 1
    assert summary[0]["user_id"] == 11111111
    assert summary[0]["data_type"] == "SLEEP"
    assert summary[0]["entries"] == 1


def test_extract_resolves_ledger_on_success(tmp_path):
    """
    A success for the same (account, type, date) marks the ledger row resolved.
    """
    from garmin_health_data.db import get_unresolved_failure_summary
    from garmin_health_data.extractor import ExtractionSuccess

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))
    _seed_ledger(db_path, [_make_failure("11111111", "SLEEP", "2025-01-02")])

    def stub_extract(*args, **kwargs):
        return {
            "garmin_files": 1,
            "activity_files": 0,
            "failures": [],
            "successes": [
                ExtractionSuccess(
                    data_type="SLEEP",
                    date="2025-01-02",
                    activity_id="",
                    user_id="11111111",
                )
            ],
            "failed_accounts": [],
        }

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", side_effect=stub_extract),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    assert "Resolved 1" in result.output
    assert get_unresolved_failure_summary(str(db_path)) == []


def test_extract_backfills_unresolved_gaps_per_account(tmp_path):
    """
    The auto-detected window moves back only for the account with unresolved per-day
    failures; the other account keeps the global start.
    """
    from datetime import date, timedelta

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    gap_date = date.today() - timedelta(days=10)
    _seed_ledger(db_path, [_make_failure("11111111", "SLEEP", gap_date.isoformat())])

    captured = {}

    def capture_extract(*args, **kwargs):
        captured.update(kwargs)
        return _stub_extract_no_files()

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch(
            "garmin_health_data.cli.get_latest_date",
            return_value=date.today() - timedelta(days=1),
        ),
        patch(
            "garmin_health_data.cli.discover_accounts",
            return_value=[
                ("11111111", Path("/tokens/11111111")),
                ("22222222", Path("/tokens/22222222")),
            ],
        ),
        patch("garmin_health_data.cli.extract_data", side_effect=capture_extract),
    ):
        result = _invoke_extract_auto_dates(runner, db_path)

    assert result.exit_code == 0, result.output
    # Only the gap-affected account gets an earlier start.
    assert captured["account_start_dates"] == {"11111111": gap_date}
    assert "backfilling unresolved failures" in result.output


def test_extract_backfill_bounded_to_30_days(tmp_path):
    """
    Failures older than the 30-day horizon stay in the ledger but do not move the auto-
    detected window back.
    """
    from datetime import date, timedelta

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    old_gap = date.today() - timedelta(days=40)
    _seed_ledger(db_path, [_make_failure("11111111", "SLEEP", old_gap.isoformat())])

    captured = {}

    def capture_extract(*args, **kwargs):
        captured.update(kwargs)
        return _stub_extract_no_files()

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch(
            "garmin_health_data.cli.get_latest_date",
            return_value=date.today() - timedelta(days=1),
        ),
        patch(
            "garmin_health_data.cli.discover_accounts",
            return_value=[("11111111", Path("/tokens/11111111"))],
        ),
        patch("garmin_health_data.cli.extract_data", side_effect=capture_extract),
    ):
        result = _invoke_extract_auto_dates(runner, db_path)

    assert result.exit_code == 0, result.output
    assert captured["account_start_dates"] is None
    # The old failure is still ledgered for manual handling.
    from garmin_health_data.db import get_unresolved_failure_summary

    assert len(get_unresolved_failure_summary(str(db_path))) == 1


def test_extract_explicit_start_date_skips_backfill(tmp_path):
    """
    An explicit --start-date disables gap backfill entirely: the user's range is used
    as-is and no per-account starts are computed.
    """
    from datetime import date, timedelta

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    gap_date = date.today() - timedelta(days=10)
    _seed_ledger(db_path, [_make_failure("11111111", "SLEEP", gap_date.isoformat())])

    captured = {}

    def capture_extract(*args, **kwargs):
        captured.update(kwargs)
        return _stub_extract_no_files()

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", side_effect=capture_extract),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    assert captured["account_start_dates"] is None
    assert captured["data_interval_start"] == "2025-01-01"


def test_extract_creates_ledger_table_on_old_database(tmp_path):
    """
    A pre-upgrade database without the ledger table gains it automatically on the next
    extract run, and failures are recorded into it.
    """
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE user (user_id BIGINT PRIMARY KEY, full_name TEXT, "
        "birth_date DATE, create_ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute("INSERT INTO user (user_id, full_name) VALUES (11111111, 'Ada')")
    conn.commit()
    conn.close()

    def stub_extract(*args, **kwargs):
        return {
            "garmin_files": 0,
            "activity_files": 0,
            "failures": [_make_failure("11111111", "SLEEP", "2025-01-02")],
            "successes": [],
            "failed_accounts": [],
        }

    runner = CliRunner()
    with (
        patch("garmin_health_data.cli.ensure_authenticated"),
        patch("garmin_health_data.cli.extract_data", side_effect=stub_extract),
    ):
        result = _common_invoke(runner, db_path)

    assert result.exit_code == 0, result.output
    conn = sqlite3.connect(db_path)
    names = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    preserved = conn.execute(
        "SELECT full_name FROM user WHERE user_id = 11111111"
    ).fetchone()
    conn.close()
    assert "extraction_failure" in names
    assert preserved == ("Ada",)

    from garmin_health_data.db import get_unresolved_failure_summary

    assert len(get_unresolved_failure_summary(str(db_path))) == 1


def test_gaps_command_lists_unresolved_failures(tmp_path):
    """
    The gaps command groups unresolved failures by account and type with counts, date
    bounds, and the most recent error.
    """
    from garmin_health_data.cli import gaps

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))
    _seed_ledger(
        db_path,
        [
            _make_failure("11111111", "SLEEP", "2025-01-02"),
            _make_failure("11111111", "SLEEP", "2025-01-02"),
            _make_failure(
                "11111111", "SLEEP", "2025-01-05", error="RuntimeError: newer"
            ),
            _make_failure("22222222", "STRESS", "2025-01-03"),
        ],
    )

    result = CliRunner().invoke(gaps, ["--db-path", str(db_path)])

    assert result.exit_code == 0, result.output
    assert "Unresolved extraction failures" in result.output
    assert "Account 11111111 / SLEEP" in result.output
    assert "2 gap(s), 3 failure(s)" in result.output
    assert "2025-01-02" in result.output
    assert "2025-01-05" in result.output
    assert "RuntimeError: newer" in result.output
    assert "Account 22222222 / STRESS" in result.output


def test_gaps_command_scopes_by_accounts(tmp_path):
    """
    --accounts filters the gaps listing to the given user IDs.
    """
    from garmin_health_data.cli import gaps

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))
    _seed_ledger(
        db_path,
        [
            _make_failure("11111111", "SLEEP", "2025-01-02"),
            _make_failure("22222222", "STRESS", "2025-01-03"),
        ],
    )

    result = CliRunner().invoke(
        gaps, ["--db-path", str(db_path), "--accounts", "11111111"]
    )

    assert result.exit_code == 0, result.output
    assert "Account 11111111" in result.output
    assert "Account 22222222" not in result.output


def test_gaps_command_empty_ledger(tmp_path):
    """
    An empty ledger reports success rather than an empty table.
    """
    from garmin_health_data.cli import gaps

    db_path = tmp_path / "test.db"
    create_tables(str(db_path))

    result = CliRunner().invoke(gaps, ["--db-path", str(db_path)])

    assert result.exit_code == 0, result.output
    assert "No unresolved extraction failures" in result.output


def test_gaps_command_missing_database(tmp_path):
    """
    A missing database exits 1 with the standard hint.
    """
    from garmin_health_data.cli import gaps

    result = CliRunner().invoke(gaps, ["--db-path", str(tmp_path / "nope.db")])

    assert result.exit_code == 1
    assert "Database not found" in result.output
