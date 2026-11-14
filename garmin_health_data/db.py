"""
Database initialization and management for garmin-health-data.

Handles SQLite database creation, session management, and query utilities.
"""

import sqlite3
import sys
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import click
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import Session

# Handle importlib.resources for different Python versions
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

from garmin_health_data.models import (
    Activity,
    BodyBattery,
    ExtractionFailure,
    Floors,
    HeartRate,
    IntensityMinutes,
    Respiration,
    Sleep,
    Steps,
    Stress,
    TrainingReadiness,
    User,
)

# Minimum SQLite version. 3.35.0 (2021-03-12) added support for
# `INSERT ... ON CONFLICT ... RETURNING`, which the bulk upsert helper relies on
# to read back database-assigned columns (e.g. the auto-increment `sleep_id`).
_MIN_SQLITE_VERSION = (3, 35, 0)


def check_sqlite_version() -> None:
    """
    Validate that the linked SQLite library meets the minimum required version.

    :raises RuntimeError: If the installed SQLite version is too old.
    """
    if sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION:
        required = ".".join(str(v) for v in _MIN_SQLITE_VERSION)
        raise RuntimeError(
            f"garmin-health-data requires SQLite >= {required} "
            f"(found {sqlite3.sqlite_version}). "
            "Upgrade your Python build or system SQLite library."
        )


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    """
    Enable SQLite foreign-key enforcement on every new connection.

    SQLite defaults `PRAGMA foreign_keys` to OFF per connection, so `ON DELETE CASCADE`
    clauses in the schema are silently inert without this. Registered as a SQLAlchemy
    `connect` event listener on engines created by :func:`get_engine`.

    :param dbapi_connection: DB-API connection instance (sqlite3.Connection).
    :param _connection_record: SQLAlchemy connection record (unused).
    """
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def get_engine(db_path: str = "garmin_data.db"):
    """
    Create SQLAlchemy engine for SQLite database.

    Attaches a `connect` event listener that runs `PRAGMA foreign_keys=ON` on every new
    connection so `ON DELETE CASCADE` clauses in the schema actually fire.

    :param db_path: Path to SQLite database file.
    :return: SQLAlchemy engine.
    """
    check_sqlite_version()
    db_file = Path(db_path).expanduser().resolve()

    # Create database URL (use as_posix() for cross-platform compatibility).
    db_url = f"sqlite:///{db_file.as_posix()}"

    # Create engine with sensible defaults for SQLite.
    engine = create_engine(
        db_url,
        echo=False,  # Set to True for SQL debugging.
        connect_args={"check_same_thread": False},  # Allow multi-threading.
    )
    event.listen(engine, "connect", _enable_sqlite_foreign_keys)

    return engine


def create_tables(db_path: str = "garmin_data.db") -> None:
    """
    Create all tables in the database by executing the DDL file.

    The schema is defined in tables.ddl which includes inline comments preserved in the
    database.

    :param db_path: Path to SQLite database file.
    """
    # Fail fast if the linked SQLite library can't support the bulk upsert
    # helper's RETURNING-based path. Mirrors the gate in `get_engine` so
    # `garmin init` surfaces the same clear error as the read/write path.
    check_sqlite_version()

    # Execute DDL file to create all tables with inline comments.
    # Use importlib.resources to read the packaged resource file.
    try:
        ddl_sql = files("garmin_health_data").joinpath("tables.ddl").read_text()
    except (FileNotFoundError, TypeError):
        # Fallback for development/editable installs
        ddl_file = Path(__file__).parent / "tables.ddl"
        if not ddl_file.exists():
            raise FileNotFoundError(f"Schema DDL file not found: {ddl_file}")
        ddl_sql = ddl_file.read_text()

    # Use raw SQLite connection to execute the script (supports multiple statements).
    # This preserves inline comments in the database schema.
    db_file = Path(db_path).expanduser()
    conn = sqlite3.connect(str(db_file))
    try:
        conn.executescript(ddl_sql)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_session(db_path: str = "garmin_data.db"):
    """
    Context manager for database sessions.

    :param db_path: Path to SQLite database file.
    :yield: SQLAlchemy Session.
    """
    engine = get_engine(db_path)
    session = Session(engine)

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def initialize_database(db_path: str = "garmin_data.db") -> None:
    """
    Initialize a new database with all tables and indexes.

    :param db_path: Path to SQLite database file.
    """
    db_file = Path(db_path).expanduser()

    if db_file.exists():
        click.echo(f"Database already exists at: {db_file}")
    else:
        click.echo(f"Creating new database at: {db_file}")

    create_tables(db_path)
    click.secho("✅ Database initialized successfully", fg="green")
    click.echo(
        "\nSchema includes inline documentation. To view table definitions:\n"
        f"  sqlite3 {db_file} \"SELECT sql FROM sqlite_master WHERE type='table';\""
    )


def get_last_update_dates(db_path: str = "garmin_data.db") -> Dict[str, Optional[date]]:
    """
    Get the last update date for each data type.

    :param db_path: Path to SQLite database file.
    :return: Dictionary mapping data type name to last update date.
    """
    with get_session(db_path) as session:
        dates = {}

        # Sleep data.
        last_sleep = session.execute(select(func.max(Sleep.start_ts))).scalar()
        dates["sleep"] = last_sleep.date() if last_sleep else None

        # Heart rate.
        last_hr = session.execute(select(func.max(HeartRate.timestamp))).scalar()
        dates["heart_rate"] = last_hr.date() if last_hr else None

        # Activities.
        last_activity = session.execute(select(func.max(Activity.start_ts))).scalar()
        dates["activity"] = last_activity.date() if last_activity else None

        # Stress.
        last_stress = session.execute(select(func.max(Stress.timestamp))).scalar()
        dates["stress"] = last_stress.date() if last_stress else None

        # Body battery.
        last_bb = session.execute(select(func.max(BodyBattery.timestamp))).scalar()
        dates["body_battery"] = last_bb.date() if last_bb else None

        # Steps.
        last_steps = session.execute(select(func.max(Steps.timestamp))).scalar()
        dates["steps"] = last_steps.date() if last_steps else None

        # Respiration.
        last_resp = session.execute(select(func.max(Respiration.timestamp))).scalar()
        dates["respiration"] = last_resp.date() if last_resp else None

        # Floors.
        last_floors = session.execute(select(func.max(Floors.timestamp))).scalar()
        dates["floors"] = last_floors.date() if last_floors else None

        # Intensity minutes.
        last_im = session.execute(select(func.max(IntensityMinutes.timestamp))).scalar()
        dates["intensity_minutes"] = last_im.date() if last_im else None

        # Training readiness.
        last_tr = session.execute(
            select(func.max(TrainingReadiness.timestamp))
        ).scalar()
        dates["training_readiness"] = last_tr.date() if last_tr else None

        return dates


def get_latest_date(db_path: str = "garmin_data.db") -> Optional[date]:
    """
    Get the most recent date across all data types.

    :param db_path: Path to SQLite database file.
    :return: Most recent date or None if database is empty.
    """
    dates = get_last_update_dates(db_path)
    valid_dates = [d for d in dates.values() if d is not None]

    if not valid_dates:
        return None

    return max(valid_dates)


def get_record_counts(db_path: str = "garmin_data.db") -> Dict[str, int]:
    """
    Get record counts for all major tables.

    :param db_path: Path to SQLite database file.
    :return: Dictionary mapping table name to record count.
    """
    with get_session(db_path) as session:
        counts = {}

        counts["users"] = session.execute(select(func.count(User.user_id))).scalar()
        counts["activities"] = session.execute(
            select(func.count(Activity.activity_id))
        ).scalar()
        counts["sleep_sessions"] = session.execute(
            select(func.count(Sleep.sleep_id))
        ).scalar()
        counts["heart_rate_readings"] = session.execute(
            select(func.count(HeartRate.timestamp))
        ).scalar()
        counts["stress_readings"] = session.execute(
            select(func.count(Stress.timestamp))
        ).scalar()
        counts["body_battery_readings"] = session.execute(
            select(func.count(BodyBattery.timestamp))
        ).scalar()
        counts["step_readings"] = session.execute(
            select(func.count(Steps.timestamp))
        ).scalar()
        counts["respiration_readings"] = session.execute(
            select(func.count(Respiration.timestamp))
        ).scalar()

        return counts


def get_database_size(db_path: str = "garmin_data.db") -> int:
    """
    Get size of database file in bytes.

    :param db_path: Path to SQLite database file.
    :return: Size in bytes, or 0 if file doesn't exist.
    """
    db_file = Path(db_path).expanduser()

    if not db_file.exists():
        return 0

    return db_file.stat().st_size


def database_exists(db_path: str = "garmin_data.db") -> bool:
    """
    Check if database file exists.

    :param db_path: Path to SQLite database file.
    :return: True if database exists, False otherwise.
    """
    db_file = Path(db_path).expanduser()
    return db_file.exists()


def _parse_failure_date(date_str: str) -> Optional[date]:
    """
    Parse an ``ExtractionFailure.date`` label into a calendar date.

    Only a plain ISO date (``YYYY-MM-DD``) maps to a ledger ``failure_date``. Range
    labels (``"<start>..<end>"``) and empty labels (per-data-type failures) return
    ``None``: those failures are recorded for inspection but never influence the auto-
    detected start date.

    :param date_str: Date label from the extractor's failure/success record.
    :return: Parsed date, or None when the label carries no single-day context.
    """
    if not date_str or ".." in date_str:
        return None
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        return None


def _ledger_table_exists(session: Session) -> bool:
    """
    Report whether the ``extraction_failure`` table exists in the database.

    Pre-upgrade databases only gain the table on the next ``extract`` run (which
    executes ``tables.ddl``), so read-only commands must tolerate its absence.

    :param session: Active SQLAlchemy session.
    :return: True when the gap-ledger table is present.
    """
    result = session.execute(
        text(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'extraction_failure'"
        )
    ).scalar()
    return result is not None


def record_extraction_failures(db_path: str, failures: list) -> int:
    """
    Record extraction failures into the gap ledger, merging repeats into one row.

    Each failure is keyed by ``(user_id, data_type, failure_date, activity_id)``. A
    first occurrence inserts a row; a re-occurrence of the same key bumps
    ``failure_count`` and refreshes ``last_failure_ts`` / ``last_error``. A failure that
    recurs after being resolved reopens the row (``resolved_ts`` back to NULL) so the
    gap re-enters the auto-start window calculation.

    Failures without a ``user_id`` (e.g. constructed outside the per-account extraction
    loop) cannot be attributed to an account and are skipped.

    :param db_path: Path to SQLite database file.
    :param failures: List of extractor ``ExtractionFailure`` records (duck-typed:
        ``user_id``, ``data_type``, ``date``, ``activity_id``, ``error``).
    :return: Number of failures recorded.
    """
    attributable = [f for f in failures if getattr(f, "user_id", "")]
    if not attributable:
        return 0

    now = datetime.now()
    with get_session(db_path) as session:
        for failure in attributable:
            failure_date = _parse_failure_date(failure.date)
            activity_id = int(failure.activity_id) if failure.activity_id else None
            statement = select(ExtractionFailure).where(
                ExtractionFailure.user_id == int(failure.user_id),
                ExtractionFailure.data_type == failure.data_type,
                (
                    ExtractionFailure.failure_date.is_(None)
                    if failure_date is None
                    else ExtractionFailure.failure_date == failure_date
                ),
                (
                    ExtractionFailure.activity_id.is_(None)
                    if activity_id is None
                    else ExtractionFailure.activity_id == activity_id
                ),
            )
            existing = session.execute(statement).scalar_one_or_none()
            if existing is not None:
                existing.failure_count += 1
                existing.last_failure_ts = now
                existing.last_error = failure.error
                # A re-occurrence after resolution reopens the gap.
                existing.resolved_ts = None
            else:
                session.add(
                    ExtractionFailure(
                        user_id=int(failure.user_id),
                        data_type=failure.data_type,
                        failure_date=failure_date,
                        activity_id=activity_id,
                        first_failure_ts=now,
                        last_failure_ts=now,
                        failure_count=1,
                        last_error=failure.error,
                        resolved_ts=None,
                    )
                )
    return len(attributable)


def resolve_extraction_failures(db_path: str, successes: list) -> int:
    """
    Mark gap-ledger rows resolved when a run actually fetched and saved their data.

    A success matches unresolved rows with the same ``(user_id, data_type)`` and
    the same context: per-day successes resolve rows with that ``failure_date``,
    per-activity successes resolve rows with that ``activity_id``, and context-free
    successes (range / no-date types) resolve rows with neither. Resolved rows stay
    in the table for inspection but no longer extend the extraction window.

    :param db_path: Path to SQLite database file.
    :param successes: List of extractor ``ExtractionSuccess`` records (duck-typed:
        ``user_id``, ``data_type``, ``date``, ``activity_id``).
    :return: Number of ledger rows marked resolved.
    """
    attributable = [s for s in successes if getattr(s, "user_id", "")]
    if not attributable:
        return 0

    resolved = 0
    now = datetime.now()
    with get_session(db_path) as session:
        for success in attributable:
            statement = select(ExtractionFailure).where(
                ExtractionFailure.user_id == int(success.user_id),
                ExtractionFailure.data_type == success.data_type,
                ExtractionFailure.resolved_ts.is_(None),
            )
            activity_id = int(success.activity_id) if success.activity_id else None
            success_date = _parse_failure_date(success.date)
            if activity_id is not None:
                statement = statement.where(
                    ExtractionFailure.activity_id == activity_id
                )
            elif success_date is not None:
                statement = statement.where(
                    ExtractionFailure.failure_date == success_date,
                    ExtractionFailure.activity_id.is_(None),
                )
            else:
                statement = statement.where(
                    ExtractionFailure.failure_date.is_(None),
                    ExtractionFailure.activity_id.is_(None),
                )
            for row in session.execute(statement).scalars():
                row.resolved_ts = now
                resolved += 1
    return resolved


def get_earliest_unresolved_failure_date(
    db_path: str, user_id: str, since: Optional[date] = None
) -> Optional[date]:
    """
    Get the earliest unresolved per-day failure date for one account.

    Only per-day failures (``failure_date`` set) participate: per-activity, per-range,
    and per-data-type failures never influence the auto-detected start date. ``since``
    bounds how far back the gap backfill may reach; older unresolved failures stay in
    the ledger for manual handling.

    :param db_path: Path to SQLite database file.
    :param user_id: Garmin user ID (string form, as used by account discovery).
    :param since: Optional inclusive lower bound on ``failure_date``.
    :return: Earliest unresolved failure date, or None.
    """
    with get_session(db_path) as session:
        if not _ledger_table_exists(session):
            return None
        statement = select(func.min(ExtractionFailure.failure_date)).where(
            ExtractionFailure.user_id == int(user_id),
            ExtractionFailure.resolved_ts.is_(None),
            ExtractionFailure.failure_date.isnot(None),
        )
        if since is not None:
            statement = statement.where(ExtractionFailure.failure_date >= since)
        return session.execute(statement).scalar()


def get_unresolved_failure_summary(
    db_path: str, user_ids: Optional[List[int]] = None
) -> List[Dict]:
    """
    Summarize unresolved gap-ledger rows grouped by account and data type.

    :param db_path: Path to SQLite database file.
    :param user_ids: Optional list of integer user IDs to scope to. None means all
        accounts.
    :return: List of dicts (one per ``(user_id, data_type)`` group) with keys
        ``user_id``, ``data_type``, ``entries`` (distinct ledger rows),
        ``failure_count`` (total occurrences), ``earliest_date`` and ``latest_date``
        (over per-day failures; None when the group has no dated rows), and
        ``last_error`` (error of the most recent occurrence). Sorted by ``(user_id,
        data_type)``. Empty when the ledger table does not exist yet (pre-upgrade
        database).
    """
    with get_session(db_path) as session:
        if not _ledger_table_exists(session):
            return []
        statement = select(ExtractionFailure).where(
            ExtractionFailure.resolved_ts.is_(None)
        )
        if user_ids:
            statement = statement.where(ExtractionFailure.user_id.in_(list(user_ids)))
        rows = session.execute(
            statement.order_by(ExtractionFailure.user_id, ExtractionFailure.data_type)
        ).scalars()

        groups: Dict[tuple, Dict] = {}
        for row in rows:
            key = (row.user_id, row.data_type)
            group = groups.setdefault(
                key,
                {
                    "user_id": row.user_id,
                    "data_type": row.data_type,
                    "entries": 0,
                    "failure_count": 0,
                    "earliest_date": None,
                    "latest_date": None,
                    "last_error": "",
                    "_latest_ts": None,
                },
            )
            group["entries"] += 1
            group["failure_count"] += row.failure_count
            if row.failure_date is not None:
                if (
                    group["earliest_date"] is None
                    or row.failure_date < group["earliest_date"]
                ):
                    group["earliest_date"] = row.failure_date
                if (
                    group["latest_date"] is None
                    or row.failure_date > group["latest_date"]
                ):
                    group["latest_date"] = row.failure_date
            if (
                group["_latest_ts"] is None
                or row.last_failure_ts >= group["_latest_ts"]
            ):
                group["_latest_ts"] = row.last_failure_ts
                group["last_error"] = row.last_error

        summary = list(groups.values())
        for group in summary:
            del group["_latest_ts"]
        return summary
