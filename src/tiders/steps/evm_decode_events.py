"""EVM event log decoding step."""

from typing import Dict
from copy import deepcopy

from tiders_core import evm_decode_events, evm_event_signature_to_arrow_schema
from ..config import EvmDecodeEventsConfig, OriginalColumns
import pyarrow as pa

# Raw, encoded input columns consumed by the EVM event decoder.
_RAW_COLUMNS = ("data", "topic0", "topic1", "topic2", "topic3")


def execute(
    data: Dict[str, pa.Table], config: EvmDecodeEventsConfig
) -> Dict[str, pa.Table]:
    """Decode raw EVM log entries into structured columns.

    Reads from ``config.input_table``, decodes each batch using the provided
    event signature, and writes the result to ``config.output_table``.
    ``config.original_columns`` controls how much of the original log row is
    kept alongside the decoded columns (see :class:`OriginalColumns`).

    Args:
        data: A dictionary mapping table names to PyArrow Tables.
        config: An :class:`EvmDecodeEventsConfig` with the event signature and
            decoding options.

    Returns:
        A new data dictionary containing the decoded output table.
    """
    data = deepcopy(data)

    input_table = data[config.input_table]
    input_batches = input_table.to_batches()

    mode = config.original_columns
    hstack = mode is not OriginalColumns.NONE

    output_batches = []

    for batch in input_batches:
        output_batches.append(
            evm_decode_events(
                config.event_signature,
                batch,
                config.allow_decode_fail,
                config.filter_by_topic0,
                hstack,
                config.large_int_as_binary,
            )
        )

    decoded_schema = evm_event_signature_to_arrow_schema(
        config.event_signature, config.large_int_as_binary
    )
    if hstack:
        schema = pa.schema(list(decoded_schema) + list(input_table.schema))
    else:
        schema = decoded_schema

    output_table = pa.Table.from_batches(output_batches, schema=schema)

    if mode in (OriginalColumns.DROP_ALL_RAW, OriginalColumns.DROP_EXCEPT_TOPIC0):
        raw = set(_RAW_COLUMNS)
        if mode is OriginalColumns.DROP_EXCEPT_TOPIC0:
            raw.discard("topic0")
        # Drop the raw columns only within the appended input region (index
        # >= len(decoded_schema)), by position, so a decoded column that shares a
        # name with a raw column is never removed. Absent columns are ignored.
        n_decoded = len(decoded_schema)
        input_names = [f.name for f in input_table.schema]
        drop_indices = {
            n_decoded + i for i, name in enumerate(input_names) if name in raw
        }
        if drop_indices:
            keep = [i for i in range(output_table.num_columns) if i not in drop_indices]
            output_table = output_table.select(keep)

    data[config.output_table] = output_table

    return data
