"""SVM (Solana) log decoding step."""

from typing import Dict
from copy import deepcopy

from tiders_core import svm_decode_logs, instruction_signature_to_arrow_schema
from tiders_core.svm_decode import InstructionSignature
from ..config import SvmDecodeLogsConfig, OriginalColumns
import pyarrow as pa

# Raw, encoded input column consumed by the SVM log decoder: the base64 `message`
# blob.
_RAW_COLUMNS = ("message",)


def execute(
    data: Dict[str, pa.Table], config: SvmDecodeLogsConfig
) -> Dict[str, pa.Table]:
    """Decode raw Solana program log entries into structured columns.

    Reads from ``config.input_table``, decodes each batch using the provided
    log signature, and writes the result to ``config.output_table``.
    ``config.original_columns`` controls how much of the original log row is
    kept alongside the decoded columns (see :class:`OriginalColumns`).

    Note:
        The schema generation reuses ``instruction_signature_to_arrow_schema``
        by constructing a temporary ``InstructionSignature`` from the log
        signature's params, since a dedicated log-schema function is not yet
        available in ``tiders_core``.

    Args:
        data: A dictionary mapping table names to PyArrow Tables.
        config: An :class:`SvmDecodeLogsConfig` with the log signature and
            decoding options.

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
            svm_decode_logs(
                config.log_signature,
                batch,
                config.allow_decode_fail,
                hstack,
            )
        )

    decoded_schema = instruction_signature_to_arrow_schema(
        # Using a hack here because ...to_arrow_schema is only implemented for InstructionSignature and not for LogSignature, but the params are the same
        InstructionSignature(
            discriminator="",  # field is not used for schema generation
            params=config.log_signature.params,
            accounts_names=[],  # log signatures don't have accounts separated from params
        ),
    )
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
