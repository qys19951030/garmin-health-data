"""
Tests for the extraction-failure gap ledger (db.py read/write functions).

Covers recording (insert + merge), resolution on successful re-extraction, auto-start
candidate queries, the unresolved-failure summary used by the ``garmin gaps`` command,
and auto-creation of the ledger table on old databases. All tests are offline: they use
a temporary SQLite database and the extractor's record dataclasses directly.
"""

import sqlite3
from datetime import date, timedelta

from garmin_health_data.db import (
    create_tables,
    get_earliest_unresolved_failure_date,
    get_session,
    get_unresolved_failure_summary,
    record_extraction_failures,
    resolve_extraction_failures,
)
from garmin_health_data.extractor import ExtractionFailure, ExtractionSuccess
from garmin_health_data.models import ExtractionFailure as ExtractionFailureRow


def _failure(
    user_id="123",
    data_type="SLEEP",
    date_str="2025-01-02",
    activity_id="",
    error="RuntimeError: boom",
):
    """
    Build an extractor ExtractionFailure record with sensible defaults.

    :param user_id: Account user ID (string form).
    :param data_type: Garmin data type name.
    :param date_str: Date label (ISO date, range label, or "").
    :param activity_id: Activity ID string, or "".
    :param error: Error description.
    :return: ExtractionFailure dataclass instance.
    """
    return ExtractionFailure(
        data_type=data_type,
        date=date_str,
        activity_id=activity_id,
        error=error,
        user_id=user_id,
    )


def _success(user_id="123", data_type="SLEEP", date_str="2025-01-02", activity_id=""):
    """
    Build an extractor ExtractionSuccess record with sensible defaults.

    :param user_id: Account user ID (string form).
    :param data_type: Garmin data type name.
    :param date_str: Date label (ISO date or "").
    :param activity_id: Activity ID string, or "".
    :return: ExtractionSuccess dataclass instance.
    """
    return ExtractionSuccess(
        data_type=data_type,
        date=date_str,
        activity_id=activity_id,
        user_id=user_id,
    )


def _ledger_rows(db_path):
    """
    Read all gap-ledger rows from the test database.

    :param db_path: Path to the test database.
    :return: List of ExtractionFailure ORM rows.
    """
    with get_session(db_path) as session:
        rows = session.query(ExtractionFailureRow).all()
        # Detach-friendly plain tuples of the fields the tests assert on.
        return [
            (
                r.user_id,
                r.data_type,
                r.failure_date,
                r.activity_id,
                r.failure_count,
                r.last_error,
                r.resolved_ts,
            )
            for r in rows
        ]


def test_first_failure_is_recorded(temp_db_path):
    """
    A first failure inserts one ledger row with count 1 and no resolution.
    """
    create_tables(temp_db_path)

    recorded = record_extraction_failures(temp_db_path, [_failure()])

    assert recorded == 1
    rows = _ledger_rows(temp_db_path)
    assert len(rows) == 1
    user_id, data_type, failure_date, activity_id, count, error, resolved = rows[0]
    assert user_id == 123
    assert data_type == "SLEEP"
    assert failure_date == date(2025, 1, 2)
    assert activity_id is None
    assert count == 1
    assert error == "RuntimeError: boom"
    assert resolved is None


def test_failures_without_user_id_are_skipped(temp_db_path):
    """
    Failure records that were never stamped with an account are not ledgered.
    """
    create_tables(temp_db_path)

    recorded = record_extraction_failures(
        temp_db_path,
        [
            ExtractionFailure(
                data_type="SLEEP", date="2025-01-02", activity_id="", error="x"
            )
        ],
    )

    assert recorded == 0
    assert _ledger_rows(temp_db_path) == []


def test_repeated_failure_merges_into_one_row(temp_db_path):
    """
    The same failure recorded again bumps failure_count and last_error instead of
    inserting a second row; first_failure_ts is preserved.
    """
    create_tables(temp_db_path)

    record_extraction_failures(temp_db_path, [_failure(error="RuntimeError: first")])
    record_extraction_failures(temp_db_path, [_failure(error="RuntimeError: second")])
    record_extraction_failures(temp_db_path, [_failure(error="RuntimeError: third")])

    rows = _ledger_rows(temp_db_path)
    assert len(rows) == 1
    assert rows[0][4] == 3  # failure_count.
    assert rows[0][5] == "RuntimeError: third"  # last_error.

    with get_session(temp_db_path) as session:
        row = session.query(ExtractionFailureRow).one()
        assert row.first_failure_ts <= row.last_failure_ts


def test_distinct_contexts_get_distinct_rows(temp_db_path):
    """
    Per-day, per-activity, and context-free failures of the same data type are separate
    ledger rows.
    """
    create_tables(temp_db_path)

    record_extraction_failures(
        temp_db_path,
        [
            _failure(date_str="2025-01-02"),
            _failure(date_str="2025-01-03"),
            _failure(date_str="", activity_id="987", data_type="ACTIVITY"),
            _failure(date_str=""),  # Per-data-type failure, no context.
            _failure(date_str="2025-01-02..2025-01-05"),  # Range failure.
        ],
    )

    rows = _ledger_rows(temp_db_path)
    # The range-labelled and type-level failures share the context-free key
    # (NULL date, NULL activity) and merge into one row with count 2.
    assert len(rows) == 4
    null_date_rows = [r for r in rows if r[2] is None]
    assert len(null_date_rows) == 2  # Activity-level + merged context-free.
    merged = [r for r in null_date_rows if r[3] is None]
    assert merged[0][4] == 2  # failure_count.


def test_range_and_type_failures_merge_separately_per_key(temp_db_path):
    """
    Context-free failures (both NULL) merge on (user, type); a range failure and a type-
    level failure of the same type share one row.
    """
    create_tables(temp_db_path)

    record_extraction_failures(temp_db_path, [_failure(date_str="")])
    record_extraction_failures(
        temp_db_path, [_failure(date_str="2025-01-02..2025-01-05")]
    )

    rows = _ledger_rows(temp_db_path)
    assert len(rows) == 1
    assert rows[0][4] == 2  # failure_count merged.


def test_success_resolves_matching_date_failure(temp_db_path):
    """
    A per-day success for the same account, type, and date marks the ledger row resolved
    and stamps resolved_ts.
    """
    create_tables(temp_db_path)
    record_extraction_failures(temp_db_path, [_failure()])

    resolved = resolve_extraction_failures(temp_db_path, [_success()])

    assert resolved == 1
    rows = _ledger_rows(temp_db_path)
    assert rows[0][6] is not None  # resolved_ts set.


def test_success_does_not_resolve_other_accounts_types_or_dates(temp_db_path):
    """
    Resolution is scoped to the exact (account, type, date) key.
    """
    create_tables(temp_db_path)
    record_extraction_failures(temp_db_path, [_failure()])

    assert resolve_extraction_failures(temp_db_path, [_success(user_id="456")]) == 0
    assert (
        resolve_extraction_failures(temp_db_path, [_success(data_type="HEART_RATE")])
        == 0
    )
    assert (
        resolve_extraction_failures(temp_db_path, [_success(date_str="2025-01-03")])
        == 0
    )
    # Still unresolved.
    assert _ledger_rows(temp_db_path)[0][6] is None


def test_activity_success_resolves_activity_failure(temp_db_path):
    """
    A per-activity success resolves the per-activity failure with the same activity ID,
    without touching date-level rows.
    """
    create_tables(temp_db_path)
    record_extraction_failures(
        temp_db_path,
        [
            _failure(data_type="ACTIVITY", date_str="", activity_id="987"),
            _failure(data_type="ACTIVITY", date_str="", activity_id="654"),
        ],
    )

    resolved = resolve_extraction_failures(
        temp_db_path,
        [_success(data_type="ACTIVITY", date_str="", activity_id="987")],
    )

    assert resolved == 1
    rows = sorted(_ledger_rows(temp_db_path), key=lambda r: r[3])
    assert rows[0][3] == 654 and rows[0][6] is None
    assert rows[1][3] == 987 and rows[1][6] is not None


def test_context_free_success_resolves_type_level_failure(temp_db_path):
    """
    A range/no-date success resolves a context-free (type-level) failure of the same
    type.
    """
    create_tables(temp_db_path)
    record_extraction_failures(
        temp_db_path, [_failure(data_type="MENSTRUAL_CYCLE_SUMMARY", date_str="")]
    )

    resolved = resolve_extraction_failures(
        temp_db_path, [_success(data_type="MENSTRUAL_CYCLE_SUMMARY", date_str="")]
    )

    assert resolved == 1


def test_refailure_after_resolution_reopens_row(temp_db_path):
    """
    A failure recurring after resolution reopens the same row (resolved_ts back to NULL)
    and keeps accumulating failure_count.
    """
    create_tables(temp_db_path)
    record_extraction_failures(temp_db_path, [_failure()])
    resolve_extraction_failures(temp_db_path, [_success()])
    assert _ledger_rows(temp_db_path)[0][6] is not None

    record_extraction_failures(temp_db_path, [_failure()])

    rows = _ledger_rows(temp_db_path)
    assert len(rows) == 1
    assert rows[0][4] == 2  # failure_count.
    assert rows[0][6] is None  # Reopened.


def test_earliest_unresolved_failure_date(temp_db_path):
    """
    The auto-start candidate is the earliest unresolved per-day failure for the account;
    resolved and context-free rows are ignored.
    """
    create_tables(temp_db_path)
    record_extraction_failures(
        temp_db_path,
        [
            _failure(date_str="2025-01-10"),
            _failure(date_str="2025-01-05"),
            _failure(date_str=""),  # Type-level: never a candidate.
            _failure(user_id="456", date_str="2024-12-01"),  # Other account.
        ],
    )

    assert get_earliest_unresolved_failure_date(temp_db_path, "123") == date(2025, 1, 5)
    assert get_earliest_unresolved_failure_date(temp_db_path, "456") == date(
        2024, 12, 1
    )
    assert get_earliest_unresolved_failure_date(temp_db_path, "789") is None

    # Resolving the earliest row advances the candidate to the next one.
    resolve_extraction_failures(temp_db_path, [_success(date_str="2025-01-05")])
    assert get_earliest_unresolved_failure_date(temp_db_path, "123") == date(
        2025, 1, 10
    )


def test_earliest_unresolved_failure_date_respects_since_bound(temp_db_path):
    """
    Failures older than the look-back horizon stay in the ledger but do not become auto-
    start candidates.
    """
    create_tables(temp_db_path)
    record_extraction_failures(
        temp_db_path,
        [
            _failure(date_str="2024-11-01"),  # Older than the horizon.
            _failure(date_str="2025-01-05"),
        ],
    )

    since = date(2025, 1, 1)
    assert get_earliest_unresolved_failure_date(
        temp_db_path, "123", since=since
    ) == date(2025, 1, 5)

    # Once the in-horizon row is resolved, the old row does not take over.
    resolve_extraction_failures(temp_db_path, [_success(date_str="2025-01-05")])
    assert (
        get_earliest_unresolved_failure_date(temp_db_path, "123", since=since) is None
    )
    # ... but it is still visible without the bound.
    assert get_earliest_unresolved_failure_date(temp_db_path, "123") == date(
        2024, 11, 1
    )


def test_unresolved_failure_summary_groups_by_account_and_type(temp_db_path):
    """
    The gaps-command summary aggregates entries, occurrence counts, date bounds, and the
    most recent error per (account, data type).
    """
    create_tables(temp_db_path)
    record_extraction_failures(temp_db_path, [_failure(date_str="2025-01-02")])
    record_extraction_failures(temp_db_path, [_failure(date_str="2025-01-02")])
    record_extraction_failures(
        temp_db_path, [_failure(date_str="2025-01-05", error="RuntimeError: newer")]
    )
    record_extraction_failures(
        temp_db_path, [_failure(user_id="456", data_type="STRESS")]
    )
    # A resolved row must not appear in the summary.
    record_extraction_failures(
        temp_db_path, [_failure(data_type="HEART_RATE", date_str="2025-01-03")]
    )
    resolve_extraction_failures(
        temp_db_path, [_success(data_type="HEART_RATE", date_str="2025-01-03")]
    )

    summary = get_unresolved_failure_summary(temp_db_path)

    assert len(summary) == 2
    sleep_group = summary[0]
    assert sleep_group["user_id"] == 123
    assert sleep_group["data_type"] == "SLEEP"
    assert sleep_group["entries"] == 2
    assert sleep_group["failure_count"] == 3
    assert sleep_group["earliest_date"] == date(2025, 1, 2)
    assert sleep_group["latest_date"] == date(2025, 1, 5)
    assert sleep_group["last_error"] == "RuntimeError: newer"
    stress_group = summary[1]
    assert stress_group["user_id"] == 456
    assert stress_group["data_type"] == "STRESS"

    # Account scoping.
    scoped = get_unresolved_failure_summary(temp_db_path, user_ids=[456])
    assert [g["user_id"] for g in scoped] == [456]


def test_summary_and_earliest_date_tolerate_missing_ledger_table(tmp_path):
    """
    Pre-upgrade databases without the ledger table report no gaps instead of raising
    (read-only commands must work before the next extract run).
    """
    db_path = str(tmp_path / "old.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE user (user_id BIGINT PRIMARY KEY)")
    conn.commit()
    conn.close()

    assert get_unresolved_failure_summary(db_path) == []
    assert get_earliest_unresolved_failure_date(db_path, "123") is None


def test_create_tables_adds_ledger_to_old_database(tmp_path):
    """
    An old database missing the ledger table gains it on the next create_tables run
    (executed by every extract); existing data is untouched.
    """
    db_path = str(tmp_path / "old.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE user (user_id BIGINT PRIMARY KEY, full_name TEXT, "
        "birth_date DATE, create_ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute("INSERT INTO user (user_id, full_name) VALUES (123, 'Ada')")
    conn.commit()
    conn.close()

    create_tables(db_path)

    conn = sqlite3.connect(db_path)
    names = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    preserved = conn.execute(
        "SELECT full_name FROM user WHERE user_id = 123"
    ).fetchone()
    conn.close()

    assert "extraction_failure" in names
    assert preserved == ("Ada",)


def test_record_and_resolve_roundtrip_preserves_history(temp_db_path):
    """
    Resolved rows stay in the ledger (queryable history) while no longer counting as
    unresolved.
    """
    create_tables(temp_db_path)
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    record_extraction_failures(temp_db_path, [_failure(date_str=yesterday)])
    resolve_extraction_failures(temp_db_path, [_success(date_str=yesterday)])

    rows = _ledger_rows(temp_db_path)
    assert len(rows) == 1  # History preserved.
    assert rows[0][6] is not None
    assert get_unresolved_failure_summary(temp_db_path) == []
