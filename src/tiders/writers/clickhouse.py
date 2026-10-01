"""ClickHouse writer backend.

Automatically creates tables from Arrow schemas and inserts data using the
``clickhouse-connect`` async client.
"""

import logging
import re
import time
from typing import Dict, Optional, cast as type_cast
import pyarrow as pa
import pyarrow.compute as pc
from ..writers.base import DataWriter
from ..config import ClickHouseWriterConfig
import asyncio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resume floor (fork: tiders 0.1.9 + resume-floor-0.1.9).
#
# `push_data` sends the child INSERTs in parallel and `blocks` last. A crash
# between them leaves child rows above MAX(blocks); on restart the resume batch
# covers a different range, so the server's data-hash dedup never matches and
# the child rows are written twice. On the first batch after process start, for
# each child table, read its own max block at or above the batch start and drop
# the incoming rows at or below it. `blocks` is never filtered.
# ---------------------------------------------------------------------------

# Child tables whose INSERT runs before the anchor. Kept a constant (not derived
# from config) so a config edit cannot silently widen the write surface.
CHILD_TABLES = (
    "transactions",
    "logs",
    "erc20_transfers",
    "uniswap_v2_swaps",
    "uniswap_v3_swaps",
)

# Studio CLAUDE.md cap rule: every read carries these three settings.
READ_CAPS = {
    "max_memory_usage": 1_500_000_000,
    "max_threads": 4,
    "max_execution_time": 600,
}

# `default`'s Arrow part cut. Above either bound the server cuts more than one
# part and the single-part premise the floor relies on no longer holds, so a
# batch above either bound is refused rather than written (spec addendum B).
INSERT_BOUND_SETTINGS = ("max_insert_block_size", "min_insert_block_size_bytes")

# The insert must not be buffered by the server's async insert, or the part
# boundaries are no longer the batch's (Cyber Security, fork gate 0.1.9).
INSERT_SETTINGS = {"async_insert": 0}

# How long to wait for an in-flight INSERT on a child table before refusing.
INFLIGHT_WAIT_SECONDS = 60

# ---------------------------------------------------------------------------
# Materialized-view atomicity (spec addendum C). The four views on `logs` push
# synchronously (parallel_view_processing = 0, materialized_views_ignore_errors
# = 0), and their targets feed `dex_trades`. Measured on scratch 2026-10-01: an
# INSERT whose view SELECT throws still commits the `logs` source part and leaves
# the target empty. So before the floor drops `logs` rows in [start, floor], each
# view's own SELECT is re-derived over that range and compared with the target's
# key set; any mismatch is no write and exit nonzero. The view SELECT is read
# from `system.tables`, never retyped, so a view edit cannot drift from the check.
# ---------------------------------------------------------------------------

# (view name, target table) for the four views on `logs`. Constant, like the
# child-table allowlist, so a config edit cannot silently change the write surface.
LOG_MV_TARGETS = (
    ("mv_curve_swaps", "curve_swaps"),
    ("mv_fluid_swaps", "fluid_swaps"),
    ("mv_pancake_v3_swaps", "pancake_v3_swaps"),
    ("mv_v4_swaps", "uniswap_v4_swaps"),
)

# The target sort key, the distinctness key for the comparison. All four targets
# are keyed (block_number, log_index); `uniqExact` is safe on an exact engine too.
MV_KEY_COLUMNS = ("block_number", "log_index")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _assert_identifier(name: str, what: str) -> str:
    if not _IDENTIFIER.match(name):
        raise ValueError(f"invalid {what}: {name!r}")
    return name


def _assert_int(value: int, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid {what}: {value!r}")
    return value


def pyarrow_type_to_clickhouse(dt: pa.DataType) -> str:
    """Convert a PyArrow data type to its ClickHouse SQL type string.

    Supports scalar types (bool, integers, floats, strings, binary, dates,
    timestamps, times, decimals) and compound types (lists, structs, maps).

    Args:
        dt: A PyArrow ``DataType`` to convert.

    Returns:
        The equivalent ClickHouse type name (e.g. ``"UInt64"``, ``"Array(String)"``).

    Raises:
        Exception: If the PyArrow type has no known ClickHouse mapping.
    """
    if pa.types.is_boolean(dt):
        return "Bool"
    elif pa.types.is_int8(dt):
        return "Int8"
    elif pa.types.is_int16(dt):
        return "Int16"
    elif pa.types.is_int32(dt):
        return "Int32"
    elif pa.types.is_int64(dt):
        return "Int64"
    elif pa.types.is_uint8(dt):
        return "UInt8"
    elif pa.types.is_uint16(dt):
        return "UInt16"
    elif pa.types.is_uint32(dt):
        return "UInt32"
    elif pa.types.is_uint64(dt):
        return "UInt64"
    elif pa.types.is_float16(dt):
        return "Float32"  # ClickHouse doesn't support Float16
    elif pa.types.is_float32(dt):
        return "Float32"
    elif pa.types.is_float64(dt):
        return "Float64"
    elif pa.types.is_string(dt):
        return "String"
    elif pa.types.is_large_string(dt):
        return "String"
    elif pa.types.is_binary(dt):
        return "String"  # ClickHouse uses String for binary data too
    elif pa.types.is_large_binary(dt):
        return "String"  # ClickHouse uses String for binary data too
    elif pa.types.is_date32(dt):
        return "Date"  # Date32 in Arrow is the same as Date in ClickHouse
    elif pa.types.is_date64(dt):
        return "DateTime"  # Date64 maps to DateTime
    elif pa.types.is_timestamp(dt):
        return "DateTime"  # Timestamp maps to DateTime
    elif pa.types.is_time32(dt):
        return "Int32"  # Time32 in Arrow maps to Int32 in ClickHouse
    elif pa.types.is_time64(dt):
        return "Int64"  # Time64 in Arrow maps to Int64 in ClickHouse
    elif pa.types.is_list(dt):
        dt = type_cast(pa.ListType, dt)
        return f"Array({pyarrow_type_to_clickhouse(dt.value_type)})"
    elif pa.types.is_large_list(dt):
        dt = type_cast(pa.LargeListType, dt)
        return f"Array({pyarrow_type_to_clickhouse(dt.value_type)})"
    elif pa.types.is_struct(dt):
        dt = type_cast(pa.StructType, dt)
        fields = [
            f"{field.name} {pyarrow_type_to_clickhouse(field.type)}"
            for field in list(dt)
        ]
        return f"Tuple({', '.join(fields)})"
    elif pa.types.is_map(dt):
        dt = type_cast(pa.MapType, dt)
        key_type = pyarrow_type_to_clickhouse(dt.key_type)
        item_type = pyarrow_type_to_clickhouse(dt.item_type)
        return f"Map({key_type}, {item_type})"
    elif pa.types.is_decimal128(dt):
        dt = type_cast(pa.Decimal128Type, dt)
        # For Decimal128, we can get precision and scale
        precision = dt.precision
        scale = dt.scale
        return f"Decimal({precision}, {scale})"
    elif pa.types.is_decimal256(dt):
        dt = type_cast(pa.Decimal256Type, dt)
        # For Decimal256, we can get precision and scale
        precision = dt.precision
        scale = dt.scale
        return f"Decimal({precision}, {scale})"
    else:
        raise Exception(f"Unimplemented pyarrow type: {dt}")


def probe_sql(
    database: str, username: str, anchor_table: Optional[str] = None
) -> tuple[str, Dict[str, str]]:
    """The in-flight INSERT probe, with database and user as server parameters.

    An unqualified ``INSERT INTO logs`` resolves against the current database, so
    the probe filters ``current_database`` as well as ``user`` (Cyber Security
    refinement 1). ``query_kind = 'Insert'`` and ``query_id != queryID()`` keep
    the probe from matching itself (Cyber Security item 5). The table alternation
    is anchored at the statement head so a ``logs_archive`` INSERT does not match.

    The anchor table joins the alternation (spec addendum D): a dead process's
    ``blocks`` INSERT must be waited out too, or the anchor assert can run before
    it commits. Widening the probe only widens detection, never the write surface.
    """
    tables = list(CHILD_TABLES)
    if anchor_table is not None:
        anchor = _assert_identifier(anchor_table, "anchor table")
        if anchor not in tables:
            tables.append(anchor)
    alternation = "|".join(tables)
    sql = (
        "SELECT count() FROM system.processes "
        "WHERE query_kind = 'Insert' "
        "AND query_id != queryID() "
        "AND current_database = {db:String} "
        "AND user = {user:String} "
        f"AND match(query, '^INSERT INTO ({alternation}) [(]')"
    )
    return sql, {"db": database, "user": username}


def child_floor_sql(table: str, start: int) -> str:
    """`count(), max(block_number)` at or above ``start`` for one child table.

    ``count()`` separates "no rows at or above the start" from a real floor: a
    numeric ``max`` over an empty set is ``0`` in ClickHouse, so the count is the
    only thing that tells the two apart, and a failed read raises instead.
    """
    return (
        f"SELECT count(), max(block_number) FROM {_assert_identifier(table, 'table')} "
        f"WHERE block_number >= {_assert_int(start, 'start block')}"
    )


def anchor_assert_sql(anchor: str) -> str:
    """`max(number)` for the anchor table, to assert it is below the batch start."""
    return f"SELECT max(number) FROM {_assert_identifier(anchor, 'anchor table')}"


def mv_definition_sql(mv: str) -> str:
    """The `CREATE MATERIALIZED VIEW` text for one view, to read its SELECT."""
    return (
        "SELECT create_table_query FROM system.tables "
        "WHERE engine = 'MaterializedView' "
        "AND database = currentDatabase() "
        f"AND name = '{_assert_identifier(mv, 'view')}'"
    )


def _mv_key_uniq() -> str:
    return "uniqExact(" + ", ".join(MV_KEY_COLUMNS) + ")"


def mv_expected_sql(select_sql: str, start: int, floor: int) -> str:
    """Distinct keys the view's own SELECT derives from `[start, floor]`."""
    lo = _assert_int(start, "start block")
    hi = _assert_int(floor, "floor block")
    return (
        f"SELECT {_mv_key_uniq()} FROM ({select_sql}) "
        f"WHERE block_number >= {lo} AND block_number <= {hi}"
    )


def mv_actual_sql(target: str, start: int, floor: int) -> str:
    """Distinct keys the target holds over `[start, floor]`."""
    lo = _assert_int(start, "start block")
    hi = _assert_int(floor, "floor block")
    return (
        f"SELECT {_mv_key_uniq()} FROM {_assert_identifier(target, 'target table')} "
        f"WHERE block_number >= {lo} AND block_number <= {hi}"
    )


def mv_select_from_definition(create_query: str, mv: str) -> str:
    """The `SELECT` half of a view definition, or a refusal.

    ClickHouse stores the definition as ``CREATE MATERIALIZED VIEW ... AS SELECT
    ...``. Slicing on the first `` AS `` returns the view's real SQL, so the
    comparison never retypes a predicate that could drift. No `` AS `` means the
    definition was not what we expect, which is a refusal.
    """
    marker = " AS "
    idx = create_query.find(marker)
    if idx < 0:
        raise RuntimeError(
            f"cannot read the SELECT of view {mv} from its definition; no write"
        )
    return create_query[idx + len(marker):]


class Writer(DataWriter):
    """ClickHouse writer that auto-creates tables and inserts Arrow data.

    On the first :meth:`push_data` call (when ``create_tables`` is enabled),
    tables are created using the Arrow schema of the incoming data. Subsequent
    calls insert data directly. If an ``anchor_table`` is configured, it is
    always inserted last to provide ordering guarantees.

    The first batch after process start also applies the resume floor: for each
    child table, rows at or below that table's own leftover high-water mark are
    dropped before the INSERT. See the module docstring.
    """

    def __init__(self, config: ClickHouseWriterConfig):
        if config.client is not None:
            self.client = config.client
            self.database = config.client.database
            self._owns_client = False
        else:
            self.client = None
            self.database = config.database
            self._owns_client = True
            self._client_config = {
                "host": config.host,
                "port": config.port,
                "username": config.username,
                "password": config.password,
                "database": config.database,
                "secure": config.secure,
            }
        self.username = config.username
        self.order_by = config.order_by
        self.codec = config.codec
        self.skip_index = config.skip_index
        self.first_insert = True
        self._floor_applied = False
        self._insert_bounds: Optional[Dict[str, int]] = None
        self.anchor_table = config.anchor_table
        self.engine = config.engine
        self.create_tables = config.create_tables

    async def _ensure_client(self):
        """Create the async ClickHouse client if not already initialized."""
        if self.client is None:
            import clickhouse_connect

            self.client = await clickhouse_connect.get_async_client(
                **self._client_config
            )

    async def _create_table_if_not_exists(self, table_name: str, schema: pa.Schema):
        """Create a ClickHouse table from an Arrow schema if it does not already exist."""
        if not await self._check_table_exists(table_name):
            await self._create_table(table_name, schema)
        else:
            logger.debug(f"table {table_name} already exists so skipping creation")

    async def _check_table_exists(self, table_name: str) -> bool:
        """Return ``True`` if a table with the given name exists in the current database."""
        assert self.client is not None
        res = await self.client.query(
            f"SELECT count() > 0 as table_exists FROM system.tables WHERE database = '{self.database}' AND name = '{table_name}'"
        )

        return bool(res.result_rows[0][0])

    async def _create_table(self, table_name: str, schema: pa.Schema) -> None:
        """Issue a ``CREATE TABLE`` DDL statement followed by any skip-index definitions."""
        assert self.client is not None
        columns = []

        for field in schema:
            ch_type = pyarrow_type_to_clickhouse(field.type)
            col_def = f"`{field.name}` {ch_type}"

            if table_name in self.codec:
                table_codec = self.codec[table_name]

                if field.name in table_codec:
                    col_def += f" CODEC({table_codec[field.name]})"

            columns.append(col_def)

        order_by = "tuple()"
        if table_name in self.order_by:
            col_list = self.order_by[table_name]
            if len(col_list) > 0:
                order_by = f"({', '.join(col_list)})"

        create_table_query = f"""
        CREATE TABLE {table_name} (
            {", ".join(columns)}
        ) ENGINE = {self.engine} 
        ORDER BY {order_by}
        """

        logger.debug(f"creating table with: {create_table_query}")

        await self.client.command(create_table_query)

        if table_name in self.skip_index:
            for idx in self.skip_index[table_name]:
                skip_index = f"ALTER TABLE {table_name} ADD INDEX {idx.name} {idx.val} TYPE {idx.type_} GRANULARITY {idx.granularity}"
                logger.debug(f"creating index with {skip_index}")
                await self.client.command(skip_index)

    # -- resume floor -------------------------------------------------------

    @staticmethod
    def _batch_start(data: Dict[str, pa.Table]) -> Optional[int]:
        """The first batch's starting block, the minimum child ``block_number``.

        Every table in a batch shares one block range, so the minimum across the
        child tables present is the batch start. ``None`` when no child table is
        present or all are empty, in which case the floor drops nothing.
        """
        starts = []
        for name, table in data.items():
            if name not in CHILD_TABLES:
                continue
            if "block_number" not in table.column_names:
                continue
            column = table.column("block_number")
            if len(column) == 0:
                continue
            value = pc.min(column).as_py()
            if value is not None:
                starts.append(int(value))
        return min(starts) if starts else None

    @staticmethod
    def _drop_at_or_below(table: pa.Table, floor: int) -> pa.Table:
        """Rows whose ``block_number`` is strictly above ``floor``."""
        column = table.column("block_number")
        mask = pc.greater(column, pa.scalar(floor, type=column.type))
        return table.filter(mask)

    async def _read_child_floor(self, table: str, start: int) -> Optional[int]:
        """This child table's own leftover high-water mark at or above ``start``.

        ``None`` means the table has no rows at or above the start, so nothing is
        dropped. A failed read, a missing row, or a non-integer result raises, so
        the batch is never written on an unanswered floor.
        """
        assert self.client is not None
        res = await self.client.query(child_floor_sql(table, start), settings=READ_CAPS)
        row = res.result_rows[0] if res.result_rows else None
        if row is None:
            raise RuntimeError(f"resume floor read for {table} returned no row")
        row_count, value = row[0], row[1]
        if row_count == 0 or value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise RuntimeError(
                f"resume floor read for {table} returned a non-integer {value!r}"
            )
        return int(value)

    async def _wait_for_inflight(self, start: int) -> None:
        """Wait out any in-flight child INSERT, then assert the anchor is below start.

        A dead process's `blocks` INSERT can commit after the new start was
        computed; the anchor assert closes that case (spec addendum D). A wait
        that runs out, an empty probe or anchor result, or an anchor at or above
        start refuses the write. A ``NULL`` anchor (an empty anchor table) is the
        only value that does not: there is no committed row to violate the bound.
        """
        assert self.client is not None
        sql, parameters = probe_sql(self.database, self.username, self.anchor_table)
        deadline = time.monotonic() + INFLIGHT_WAIT_SECONDS
        while True:
            res = await self.client.query(sql, parameters=parameters, settings=READ_CAPS)
            if not res.result_rows or res.result_rows[0][0] is None:
                raise RuntimeError("in-flight probe returned no row; no write")
            inflight = int(res.result_rows[0][0])
            if inflight == 0:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{inflight} in-flight child INSERT(s) after "
                    f"{INFLIGHT_WAIT_SECONDS}s; no write"
                )
            await asyncio.sleep(1)

        if self.anchor_table is None:
            return
        res = await self.client.query(
            anchor_assert_sql(self.anchor_table), settings=READ_CAPS
        )
        if not res.result_rows:
            raise RuntimeError(
                f"anchor assert for {self.anchor_table} returned no row; no write"
            )
        value = res.result_rows[0][0]
        if value is not None and int(value) >= start:
            raise RuntimeError(
                f"MAX({self.anchor_table}) = {value} >= batch start {start}; no write"
            )

    async def _read_insert_bounds(self) -> Dict[str, int]:
        """The server's Arrow part-cut settings, read once before the first write."""
        assert self.client is not None
        names = ", ".join(f"'{name}'" for name in INSERT_BOUND_SETTINGS)
        res = await self.client.query(
            f"SELECT name, value FROM system.settings WHERE name IN ({names})",
            settings=READ_CAPS,
        )
        bounds: Dict[str, int] = {}
        for name, value in res.result_rows:
            bounds[name] = int(value)
        missing = [name for name in INSERT_BOUND_SETTINGS if name not in bounds]
        if missing:
            raise RuntimeError(f"server did not return insert bounds {missing}; no write")
        return bounds

    @staticmethod
    def _refuse(table: str, size: int, bound_name: str, bound_value: int) -> None:
        """Print a grep-able refusal line and raise, so the write never happens."""
        line = (
            f"TIDERS_RESUME_REFUSAL table={table} size={size} "
            f"bound={bound_name}={bound_value}"
        )
        logger.error(line)
        # Print as well as log: the systemd journal captures stdout even when the
        # handler is not configured, and the alert path greps for the prefix.
        print(line, flush=True)
        raise RuntimeError(line)

    def _refuse_oversized_batches(
        self, data: Dict[str, pa.Table], bounds: Dict[str, int]
    ) -> None:
        """Refuse any table whose batch crosses the server's part cut.

        Runs on every batch, not only the first: the batch that crashes can be
        any batch, and its one-part premise must hold (spec addendum B).
        """
        max_rows = bounds["max_insert_block_size"]
        max_bytes = bounds["min_insert_block_size_bytes"]
        for table_name, table in data.items():
            if table.num_rows > max_rows:
                self._refuse(table_name, table.num_rows, "max_insert_block_size", max_rows)
            if table.nbytes > max_bytes:
                self._refuse(
                    table_name, table.nbytes, "min_insert_block_size_bytes", max_bytes
                )

    async def _apply_resume_floor(self, data: Dict[str, pa.Table]) -> None:
        """Drop each child table's rows at or below its own leftover high-water mark.

        Runs once, on the first batch after process start. Any failed read raises
        through to the caller, so nothing is written.
        """
        start = self._batch_start(data)
        if start is None:
            return
        await self._wait_for_inflight(start)
        for table_name in list(data.keys()):
            if table_name not in CHILD_TABLES or table_name == self.anchor_table:
                continue
            floor = await self._read_child_floor(table_name, start)
            if floor is None:
                continue
            # Dropping `logs` rows the views never pushed would leave a silent gap
            # in their targets (spec addendum C, measured to commit on scratch).
            if table_name == "logs":
                await self._check_log_materialized_views(start, floor)
            before = data[table_name].num_rows
            data[table_name] = self._drop_at_or_below(data[table_name], floor)
            logger.info(
                "resume floor %s: dropped %d row(s) at or below block %d (start %d)",
                table_name,
                before - data[table_name].num_rows,
                floor,
                start,
            )

    async def _read_mv_select(self, mv: str) -> str:
        """The view's own SELECT, read from `system.tables` at start. Fail closed."""
        assert self.client is not None
        res = await self.client.query(mv_definition_sql(mv), settings=READ_CAPS)
        row = res.result_rows[0] if res.result_rows else None
        if row is None or not row[0]:
            raise RuntimeError(f"view {mv} has no definition on the server; no write")
        return mv_select_from_definition(str(row[0]), mv)

    async def _read_key_count(self, sql: str, what: str) -> int:
        """One `uniqExact` result. A missing or non-integer row raises, so no write."""
        assert self.client is not None
        res = await self.client.query(sql, settings=READ_CAPS)
        row = res.result_rows[0] if res.result_rows else None
        if row is None or row[0] is None:
            raise RuntimeError(f"{what} returned no value; no write")
        value = row[0]
        if not isinstance(value, int) or isinstance(value, bool):
            raise RuntimeError(f"{what} returned a non-integer {value!r}; no write")
        return int(value)

    async def _check_log_materialized_views(self, start: int, floor: int) -> None:
        """Refuse when a `logs` view's target is short over `[start, floor]`.

        The view SELECT is derived over the range and compared with the target's
        key set. A short target means a crashed push did not reach it, so dropping
        the source rows would lose them from published data.
        """
        for mv, target in LOG_MV_TARGETS:
            select_sql = await self._read_mv_select(mv)
            expected = await self._read_key_count(
                mv_expected_sql(select_sql, start, floor), f"view {mv} derivation"
            )
            actual = await self._read_key_count(
                mv_actual_sql(target, start, floor), f"target {target}"
            )
            if expected != actual:
                line = (
                    f"TIDERS_RESUME_MV_GAP view={mv} target={target} "
                    f"range={start}..{floor} expected={expected} actual={actual}"
                )
                logger.error(line)
                print(line, flush=True)
                raise RuntimeError(line + "; no write")

    async def push_data(self, data: Dict[str, pa.Table]) -> None:
        """Insert Arrow Tables into ClickHouse, creating tables on the first call if needed."""
        await self._ensure_client()
        assert self.client is not None
        # create tables if this is the first insert
        if self.create_tables and self.first_insert:
            tasks = []

            for table_name, table_data in data.items():
                task = asyncio.create_task(
                    self._create_table_if_not_exists(table_name, table_data.schema),
                    name=f"create table {table_name}",
                )
                tasks.append(task)

            for task in tasks:
                await task

            self.first_insert = False

        # The server's part-cut check runs on every batch before any INSERT: the
        # batch that crashes can be any batch, and its one-part premise must hold
        # (spec addendum B). The bounds are read once and cached for the process.
        if self._insert_bounds is None:
            self._insert_bounds = await self._read_insert_bounds()
        self._refuse_oversized_batches(data, self._insert_bounds)

        # resume floor: first batch after process start only
        if not self._floor_applied:
            self._floor_applied = True
            await self._apply_resume_floor(data)

        # insert into all tables except the anchor table in parallel
        tasks = []
        for table_name, table_data in data.items():
            if table_name == self.anchor_table:
                continue

            task = asyncio.create_task(
                self.client.insert_arrow(
                    table_name, table_data, settings=INSERT_SETTINGS
                ),
                name=f"write to {table_name}",
            )

            tasks.append(task)

        for task in tasks:
            await task

        # insert into anchor table after all other inserts are done
        if self.anchor_table is not None:
            table_data = data[self.anchor_table]
            await self.client.insert_arrow(
                self.anchor_table, table_data, settings=INSERT_SETTINGS
            )

    async def close(self) -> None:
        """Close the async ClickHouse client if this writer created it.

        A client supplied via ``config.client`` is owned by the caller and is
        left open. A client created lazily in :meth:`_ensure_client` is closed
        here so its aiohttp session and connection pool are released instead of
        leaking until event-loop teardown.
        """
        if self._owns_client and self.client is not None:
            await self.client.close()
            self.client = None

    async def read_max_block(self, table: str, column: str) -> Optional[int]:
        """Return MAX(column) from table, or None if the table is missing or empty."""
        await self._ensure_client()
        assert self.client is not None
        if not await self._check_table_exists(table):
            return None
        result = await self.client.query(f"SELECT MAX({column}) FROM {table}")
        value = result.result_rows[0][0] if result.result_rows else None
        if value is None:
            return None
        return int(value)
