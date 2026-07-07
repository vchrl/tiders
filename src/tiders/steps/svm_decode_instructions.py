"""SVM (Solana) instruction decoding step."""

from typing import Dict
from copy import deepcopy

from tiders_core import svm_decode_instructions, instruction_signature_to_arrow_schema
from ..config import SvmDecodeInstructionsConfig, OriginalColumns
import pyarrow as pa

# Raw, encoded input columns consumed by the SVM instruction decoder: the
# instruction `data` blob, the account columns `a0`-`a9`, and the overflow
# `rest_of_accounts` list.
_RAW_COLUMNS = ("data", "rest_of_accounts", *(f"a{i}" for i in range(10)))


def execute(
    data: Dict[str, pa.Table], config: SvmDecodeInstructionsConfig
) -> Dict[str, pa.Table]:
    """Decode raw Solana instruction data into structured columns.

    Reads from ``config.input_table``, decodes each batch using the provided
    instruction signature, and writes the result to ``config.output_table``.
    ``config.original_columns`` controls how much of the original instruction
    row is kept alongside the decoded columns (see :class:`OriginalColumns`).

    Args:
        data: A dictionary mapping table names to PyArrow Tables.
        config: An :class:`SvmDecodeInstructionsConfig` with the instruction
            signature and decoding options.

    Returns:
        A new data dictionary containing the decoded output table.
    """
    data = deepcopy(data)

    mode = config.original_columns
    if mode is OriginalColumns.DROP_EXCEPT_TOPIC0:
        raise ValueError(
            "original_columns='drop_except_topic0' is only valid for EVM decode steps."
        )
    hstack = mode is not OriginalColumns.NONE

    input_table = data[config.input_table]
    input_batches = input_table.to_batches()

    output_batches = []

    for batch in input_batches:
        output_batches.append(
            svm_decode_instructions(
                config.instruction_signature,
                batch,
                config.allow_decode_fail,
                config.filter_by_discriminator,
                hstack,
            )
        )

    decoded_schema = instruction_signature_to_arrow_schema(config.instruction_signature)
    if hstack:
        schema = pa.schema(list(decoded_schema) + list(input_table.schema))
    else:
        schema = decoded_schema

    output_table = pa.Table.from_batches(output_batches, schema=schema)

    if mode is OriginalColumns.DROP_ALL_RAW:
        # Drop the raw columns only within the appended input region (index
        # >= len(decoded_schema)), by position, so a decoded column that shares a
        # name with a raw column is never removed. Absent columns are ignored.
        raw = set(_RAW_COLUMNS)
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
