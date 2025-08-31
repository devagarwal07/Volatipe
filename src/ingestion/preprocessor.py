from __future__ import annotations
import pandas as pd
from pathlib import Path
from ..utils.logging import get_logger

logger = get_logger(__name__)


def merge_equities_indices(equities: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    if equities.empty or indices.empty:
        logger.warning("One of the input dataframes is empty")
    # Simple example: pivot equities by symbol and join index level close
    eq_close = equities.pivot_table(values='close', index=equities.index, columns='symbol')
    idx_close = indices.pivot_table(values='close', index=indices.index, columns='symbol')
    merged = eq_close.join(idx_close, how='outer', rsuffix='_idx')
    return merged.sort_index()
