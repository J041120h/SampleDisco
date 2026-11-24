"""Batch columns at the two correction levels.

``*_batch_col`` is the default batch column (or list of columns). Each level inherits it unless
set explicitly:

- cell level (``*_cell_level_batch_key``): corrects the cell embeddings ``Z_comp`` and ``Z_rmd``;
- sample level (``*_sample_level_batch_col``): defines the RMD reference group (crossed with
  modality for multi-omics), the sample-level Harmony covariates, the autotune batch term and
  the downstream DE batch. Every listed column is used.
"""
from typing import List, Optional, Sequence, Tuple, Union

BatchCols = Optional[Union[str, Sequence[str]]]


def batch_col_list(value: BatchCols) -> List[str]:
    """A column name, or a list of names, as a list; None and blank names are dropped."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(x) for x in value if x is not None and str(x).strip()]


def resolve_batch_levels(batch_col: BatchCols, cell_level: BatchCols,
                         sample_level: BatchCols) -> Tuple[List[str], List[str]]:
    """(cell-level, sample-level) batch columns. A level left unset (None) inherits
    ``batch_col``; an empty list means no batch correction at that level."""
    default = batch_col_list(batch_col)
    return (default if cell_level is None else batch_col_list(cell_level),
            default if sample_level is None else batch_col_list(sample_level))
