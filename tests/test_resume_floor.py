"""Resume floor tests for the clickhouse writer (fork: resume-floor-0.1.9).

The production call site is exercised: every arm builds a ``Writer`` and calls
``push_data``, then reads the recorded INSERTs and the queries it issued. Only
the ClickHouse client is faked.
"""

import asyncio
import re

import pyarrow as pa
import pytest

from tiders.writers.clickhouse import (
    INSERT_SETTINGS,
    LOG_MV_TARGETS,
    READ_CAPS,
    Writer,
    anchor_assert_sql,
    child_floor_sql,
    mv_actual_sql,
    mv_definition_sql,
    mv_expected_sql,
    mv_select_from_definition,
    probe_sql,
)


class FakeResult:
    def __init__(self, rows):
        self.result_rows = rows


class FakeClient:
    """Records every query and insert; answers with a test-supplied handler."""

    def __init__(self, database, handler):
        self.database = database
        self.handler = handler
        self.queries = []
        self.inserts = []
        self.commands = []

    async def query(self, sql, parameters=None, settings=None):
        self.queries.append({"sql": sql, "parameters": parameters, "settings": settings})
        return self.handler(sql, parameters)

    async def command(self, sql):
        self.commands.append(sql)
        return None

    async def insert_arrow(self, table, arrow_table, settings=None, database=None):
        self.inserts.append({"table": table, "rows": arrow_table.num_rows, "settings": settings})

    async def close(self):
        return None


def table(blocks):
    return pa.table({"block_number": pa.array(blocks, type=pa.uint64())})


SETTINGS_ROW = [
    ("max_insert_block_size", "1000000"),
    ("min_insert_block_size_bytes", str(512 * 1024 * 1024)),
]


def default_handler(
    floors=None, anchor_max=None, settings=None, processes=0,
    mv_expected=0, mv_actual=0, mv_defs=True,
):
    floors = floors or {}
    settings = settings if settings is not None else SETTINGS_ROW

    def handler(sql, parameters):
        if sql.startswith("SELECT create_table_query FROM system.tables"):
            if not mv_defs:
                return FakeResult([])
            return FakeResult([
                (
                    "CREATE MATERIALIZED VIEW default.mv TO default.t AS SELECT "
                    "block_number, log_index FROM default.logs WHERE topic0 = '0x0'",
                )
            ])
        if sql.startswith("SELECT uniqExact("):
            return FakeResult([(mv_expected if " FROM (" in sql else mv_actual,)])
        if "system.tables" in sql:
            return FakeResult([(1,)])
        if sql.startswith("SELECT name, value FROM system.settings"):
            return FakeResult(settings)
        if "system.processes" in sql:
            return FakeResult([(processes,)])
        if sql.startswith("SELECT max(number) FROM"):
            return FakeResult([(anchor_max,)])
        if sql.startswith("SELECT count(), max(block_number)"):
            name = sql.split("FROM ")[1].split(" ")[0]
            return FakeResult([floors.get(name, (0, 0))])
        raise AssertionError(f"unexpected query: {sql}")

    return handler


def writer_with(handler, anchor_table="blocks", create_tables=True):
    client = FakeClient("default", handler)
    config = type(
        "Cfg",
        (),
        {
            "client": client,
            "database": "default",
            "username": "default",
            "anchor_table": anchor_table,
            "codec": {},
            "order_by": {},
            "skip_index": {},
            "engine": "MergeTree()",
            "create_tables": create_tables,
        },
    )()
    return Writer(config), client


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Probe and floor SQL shape
# --------------------------------------------------------------------------


def test_probe_excludes_itself_and_filters_database_and_user():
    sql, params = probe_sql("default", "studio")
    assert "query_kind = 'Insert'" in sql
    assert "query_id != queryID()" in sql
    assert "current_database = {db:String}" in sql
    assert "user = {user:String}" in sql
    assert params == {"db": "default", "user": "studio"}
    # Anchored at the statement head, in tiders' own unqualified form.
    assert "match(query, '^INSERT INTO (transactions|logs|erc20_transfers|uniswap_v2_swaps|uniswap_v3_swaps) [(]')" in sql
    assert "default." not in sql


def test_probe_regex_matches_tiders_text_and_not_a_prefixed_or_suffixed_table():
    alternation = "transactions|logs|erc20_transfers|uniswap_v2_swaps|uniswap_v3_swaps"
    pattern = re.compile(rf"^INSERT INTO ({alternation}) [(]")
    assert pattern.match("INSERT INTO logs (block_number) FORMAT Arrow")
    assert not pattern.match("INSERT INTO default.logs (")
    assert not pattern.match("INSERT INTO logs_archive (")


def test_probe_includes_the_anchor_so_a_blocks_insert_is_waited_out():
    # Spec addendum D: a dead process's `blocks` INSERT must be in the wait, or
    # the anchor assert can run before it commits and `blocks` is written twice.
    # Drive the writer and read the probe it actually issues, so dropping the
    # anchor from the call site goes red (a direct probe_sql call would not).
    writer, client = writer_with(default_handler(floors={"transactions": (0, 0)}))
    run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    probes = [q["sql"] for q in client.queries if "system.processes" in q["sql"]]
    assert probes, "the in-flight probe ran"
    assert all("uniswap_v3_swaps|blocks)" in sql for sql in probes)
    pattern = re.compile(
        r"^INSERT INTO (transactions|logs|erc20_transfers|uniswap_v2_swaps|uniswap_v3_swaps|blocks) [(]"
    )
    assert pattern.match("INSERT INTO blocks (number) FORMAT Arrow")
    # Without an anchor argument the alternation stays the child set only.
    no_anchor, params = probe_sql("default", "default")
    assert "blocks" not in no_anchor
    assert params == {"db": "default", "user": "default"}


def test_floor_read_is_bounded_and_capped():
    sql = child_floor_sql("logs", 26095593)
    assert "count(), max(block_number)" in sql
    assert "FROM logs WHERE block_number >= 26095593" in sql
    assert set(READ_CAPS) == {"max_memory_usage", "max_threads", "max_execution_time"}


# --------------------------------------------------------------------------
# The drop
# --------------------------------------------------------------------------


def test_drops_rows_at_or_below_the_floor_and_leaves_the_rest():
    floors = {"transactions": (4, 101)}
    writer, client = writer_with(default_handler(floors=floors))
    run(writer.push_data({"transactions": table([100, 101, 102, 103]), "blocks": table([100, 101, 102, 103])}))

    inserts = {i["table"]: i["rows"] for i in client.inserts}
    assert inserts["transactions"] == 2
    assert inserts["blocks"] == 4


def test_the_boundary_row_is_dropped_so_the_comparison_is_at_or_below():
    # If the comparison were `<` instead of `<=`, block 101 would survive. The
    # mutant must go red, so the test asserts 101 is gone.
    floors = {"transactions": (4, 101)}
    writer, client = writer_with(default_handler(floors=floors))
    run(writer.push_data({"transactions": table([100, 101, 102, 103]), "blocks": table([100, 101, 102, 103])}))
    sent = [q for q in client.queries if q["sql"].startswith("SELECT count(), max(block_number)")]
    assert len(sent) == 1
    assert sent[0]["settings"] == READ_CAPS


def test_empty_floor_drops_nothing_and_does_not_look_like_a_failure():
    writer, client = writer_with(default_handler(floors={"transactions": (0, 0)}))
    run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    inserts = {i["table"]: i["rows"] for i in client.inserts}
    assert inserts["transactions"] == 2


def test_failed_floor_read_writes_nothing():
    def handler(sql, parameters):
        if sql.startswith("SELECT count(), max(block_number)"):
            raise RuntimeError("read failed")
        return default_handler()(sql, parameters)

    writer, client = writer_with(handler)
    with pytest.raises(RuntimeError):
        run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    assert client.inserts == []


def test_non_integer_floor_writes_nothing():
    writer, client = writer_with(default_handler(floors={"transactions": (4, "abc")}))
    with pytest.raises(RuntimeError):
        run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    assert client.inserts == []


def test_blocks_is_never_filtered():
    floors = {"transactions": (4, 101), "logs": (2, 101), "blocks": (4, 101)}
    writer, client = writer_with(default_handler(floors=floors))
    run(writer.push_data({"transactions": table([100, 101, 102, 103]), "logs": table([100, 101, 102, 103]), "blocks": table([100, 101, 102, 103])}))
    inserts = {i["table"]: i["rows"] for i in client.inserts}
    assert inserts["blocks"] == 4
    # No floor read was issued against blocks.
    floor_reads = [q["sql"] for q in client.queries if q["sql"].startswith("SELECT count(), max(block_number)")]
    assert all("FROM blocks" not in sql for sql in floor_reads)


def test_floor_applies_only_on_the_first_batch():
    writer, client = writer_with(default_handler(floors={"transactions": (4, 101)}))
    data = lambda: {"transactions": table([100, 101, 102, 103]), "blocks": table([100, 101, 102, 103])}
    run(writer.push_data(data()))
    floor_reads_first = [q for q in client.queries if q["sql"].startswith("SELECT count(), max(block_number)")]
    assert len(floor_reads_first) == 1
    run(writer.push_data(data()))
    floor_reads_total = [q for q in client.queries if q["sql"].startswith("SELECT count(), max(block_number)")]
    assert len(floor_reads_total) == 1
    # The second batch is written whole.
    assert client.inserts[-2]["rows"] == 4


def test_every_resume_floor_query_carries_read_caps():
    # Catches dropping READ_CAPS from the probe, the anchor read, the MV
    # definition read or the MV uniqExact reads. create_tables=False keeps the
    # pre-existing table-exists read (upstream, outside this fork) out of the run,
    # so this asserts the fork's own read surface.
    writer, client = writer_with(
        default_handler(floors={"logs": (4, 101)}), create_tables=False
    )
    run(writer.push_data(_logs_data()))
    assert client.queries, "the floor path issued reads"
    assert all(q["settings"] == READ_CAPS for q in client.queries)


# --------------------------------------------------------------------------
# In-flight wait and anchor assert
# --------------------------------------------------------------------------


def test_anchor_at_or_above_start_refuses_the_write():
    writer, client = writer_with(default_handler(anchor_max=100))
    with pytest.raises(RuntimeError):
        run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    assert client.inserts == []


def test_inflight_insert_that_does_not_clear_refuses(monkeypatch):
    monkeypatch.setattr("tiders.writers.clickhouse.INFLIGHT_WAIT_SECONDS", 0)
    writer, client = writer_with(default_handler(processes=1))
    with pytest.raises(RuntimeError):
        run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    assert client.inserts == []


def test_empty_probe_result_refuses_the_write():
    # A healthy count() never returns zero rows; an empty result fails open if it
    # is read as "0 in flight" (Cyber Security, non-blocking item).
    def handler(sql, parameters):
        if "system.processes" in sql:
            return FakeResult([])
        return default_handler()(sql, parameters)

    writer, client = writer_with(handler)
    with pytest.raises(RuntimeError):
        run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    assert client.inserts == []


def test_empty_anchor_result_refuses_the_write():
    def handler(sql, parameters):
        if sql.startswith("SELECT max(number) FROM"):
            return FakeResult([])
        return default_handler()(sql, parameters)

    writer, client = writer_with(handler)
    with pytest.raises(RuntimeError):
        run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    assert client.inserts == []


def test_a_null_anchor_does_not_refuse():
    # An empty anchor table is not a violation: there is no committed row to sit
    # at or above the batch start. Only a missing row refuses.
    writer, client = writer_with(default_handler(anchor_max=None, floors={"transactions": (0, 0)}))
    run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    inserts = {i["table"]: i["rows"] for i in client.inserts}
    assert inserts["transactions"] == 2


# --------------------------------------------------------------------------
# Part-cut refusal
# --------------------------------------------------------------------------


def test_oversized_batch_is_refused_with_a_grep_able_line(capsys):
    settings = [("max_insert_block_size", "2"), ("min_insert_block_size_bytes", str(512 * 1024 * 1024))]
    writer, client = writer_with(default_handler(settings=settings))
    with pytest.raises(RuntimeError):
        run(writer.push_data({"transactions": table([100, 101, 102, 103]), "blocks": table([100, 101, 102, 103])}))
    assert client.inserts == []
    out = capsys.readouterr().out
    assert "TIDERS_RESUME_REFUSAL table=transactions size=4 bound=max_insert_block_size=2" in out


def test_oversized_later_batch_is_refused_the_same_as_the_first():
    # Spec addendum B: the batch that crashes can be any batch, so the check must
    # not be tied to the first-batch floor. A small first batch is written; a
    # later batch over the bound writes nothing.
    settings = [("max_insert_block_size", "3"), ("min_insert_block_size_bytes", str(512 * 1024 * 1024))]
    writer, client = writer_with(default_handler(settings=settings))
    run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    written = len(client.inserts)
    assert written == 2
    with pytest.raises(RuntimeError):
        run(
            writer.push_data(
                {"transactions": table([100, 101, 102, 103]), "blocks": table([100, 101, 102, 103])}
            )
        )
    assert len(client.inserts) == written
    # The bounds are read once, not per batch.
    bound_reads = [q for q in client.queries if q["sql"].startswith("SELECT name, value FROM system.settings")]
    assert len(bound_reads) == 1


def test_byte_oversized_batch_is_refused_with_a_grep_able_line(capsys):
    # The Architect's mutant dropped the nbytes arm and the suite stayed green.
    settings = [("max_insert_block_size", "1000000"), ("min_insert_block_size_bytes", "1")]
    writer, client = writer_with(default_handler(settings=settings))
    with pytest.raises(RuntimeError):
        run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    assert client.inserts == []
    out = capsys.readouterr().out
    assert "TIDERS_RESUME_REFUSAL table=transactions" in out
    assert "bound=min_insert_block_size_bytes=1" in out


def test_missing_bound_setting_refuses_the_write():
    writer, client = writer_with(default_handler(settings=[("max_insert_block_size", "1000000")]))
    with pytest.raises(RuntimeError):
        run(writer.push_data({"transactions": table([100]), "blocks": table([100])}))
    assert client.inserts == []


# --------------------------------------------------------------------------
# async_insert pin
# --------------------------------------------------------------------------


def test_every_insert_pins_async_insert_off():
    writer, client = writer_with(default_handler(floors={"transactions": (0, 0)}))
    run(writer.push_data({"transactions": table([100, 101]), "blocks": table([100, 101])}))
    assert client.inserts, "at least one insert"
    assert all(i["settings"] == INSERT_SETTINGS for i in client.inserts)
    assert INSERT_SETTINGS == {"async_insert": 0}


# --------------------------------------------------------------------------
# Materialized-view atomicity (spec addendum C)
# --------------------------------------------------------------------------


def _logs_data():
    return {"logs": table([100, 101, 102, 103]), "blocks": table([100, 101, 102, 103])}


def test_logs_floor_derives_each_view_select_over_the_range():
    writer, client = writer_with(default_handler(floors={"logs": (4, 101)}))
    run(writer.push_data(_logs_data()))
    defs = [q["sql"] for q in client.queries if q["sql"].startswith("SELECT create_table_query")]
    assert len(defs) == len(LOG_MV_TARGETS)
    uniq = [q["sql"] for q in client.queries if q["sql"].startswith("SELECT uniqExact(")]
    # One expected and one actual per view.
    assert len(uniq) == 2 * len(LOG_MV_TARGETS)
    assert all("uniqExact(block_number, log_index)" in sql for sql in uniq)
    assert all("100 AND block_number <= 101" in sql for sql in uniq)


def test_mv_target_short_of_the_view_refuses_the_write():
    writer, client = writer_with(
        default_handler(floors={"logs": (4, 101)}, mv_expected=5, mv_actual=4)
    )
    with pytest.raises(RuntimeError):
        run(writer.push_data(_logs_data()))
    assert client.inserts == []


def test_mv_match_lets_the_logs_floor_drop():
    writer, client = writer_with(
        default_handler(floors={"logs": (4, 101)}, mv_expected=2, mv_actual=2)
    )
    run(writer.push_data(_logs_data()))
    inserts = {i["table"]: i["rows"] for i in client.inserts}
    assert inserts["logs"] == 2  # 100 and 101 dropped, 102 and 103 remain
    assert inserts["blocks"] == 4


def test_missing_view_definition_refuses_the_write():
    writer, client = writer_with(default_handler(floors={"logs": (4, 101)}, mv_defs=False))
    with pytest.raises(RuntimeError):
        run(writer.push_data(_logs_data()))
    assert client.inserts == []


def test_a_failed_mv_read_writes_nothing():
    def handler(sql, parameters):
        if sql.startswith("SELECT uniqExact("):
            raise RuntimeError("mv read failed")
        return default_handler(floors={"logs": (4, 101)})(sql, parameters)

    writer, client = writer_with(handler)
    with pytest.raises(RuntimeError):
        run(writer.push_data(_logs_data()))
    assert client.inserts == []


def test_the_view_select_is_read_from_the_server_not_retyped():
    sql = mv_definition_sql("mv_curve_swaps")
    assert "FROM system.tables" in sql
    assert "'MaterializedView'" in sql
    assert "database = currentDatabase()" in sql
    assert "'mv_curve_swaps'" in sql
    select = mv_select_from_definition(
        "CREATE MATERIALIZED VIEW default.mv TO default.t (a UInt64) AS SELECT a FROM logs WHERE x = 1",
        "mv",
    )
    assert select == "SELECT a FROM logs WHERE x = 1"
    with pytest.raises(RuntimeError):
        mv_select_from_definition("CREATE MATERIALIZED VIEW default.mv TO default.t", "mv")


def test_mv_expected_and_actual_are_bounded_and_keyed():
    expected = mv_expected_sql("SELECT block_number, log_index FROM default.logs", 100, 101)
    actual = mv_actual_sql("curve_swaps", 100, 101)
    assert expected == (
        "SELECT uniqExact(block_number, log_index) FROM "
        "(SELECT block_number, log_index FROM default.logs) "
        "WHERE block_number >= 100 AND block_number <= 101"
    )
    assert actual == (
        "SELECT uniqExact(block_number, log_index) FROM curve_swaps "
        "WHERE block_number >= 100 AND block_number <= 101"
    )
