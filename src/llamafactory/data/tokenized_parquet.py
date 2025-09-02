import itertools
from typing import Any, Dict, Iterable, List, Optional

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, IterableDataset, DatasetDict
from ..extras import logging

logger = logging.get_logger(__name__)


def _iter_parquet_rows(paths: List[str], ids_key: str, mask_key: Optional[str]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        opener = fsspec.open(path, "rb") if path.startswith("s3://") else open  # type: ignore
        with (opener(path).open() if path.startswith("s3://") else opener(path, "rb")) as f:
            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                table: pa.Table = pf.read_row_group(i)
                ids_col = table[ids_key]
                mask_col = table[mask_key] if mask_key and mask_key in table.column_names else None
                ids_py = ids_col.to_pylist()
                mask_py = mask_col.to_pylist() if mask_col is not None else itertools.repeat(None)
                for ids, mask in zip(ids_py, mask_py):
                    yield {
                        "input_ids": list(ids) if isinstance(ids, (list, tuple)) else ids,
                        **({"attention_mask": (list(mask) if isinstance(mask, (list, tuple)) else mask)} if mask is not None else {}),
                    }


def load_tokenized_parquet_dataset(
    data_files: List[str],
    ids_key: str = "input_ids",
    mask_key: Optional[str] = "attention_mask",
    streaming: bool = True,
) -> IterableDataset:
    """Create a streaming HF IterableDataset over pre-tokenized Parquet samples.

    - input: list of Parquet paths (S3 or local)
    - output: IterableDataset yielding dicts with `input_ids` and optionally `attention_mask`

    Note: Always streams row groups; avoids materializing large corpora in memory.
    """
    if not data_files:
        raise ValueError("data_files must be a non-empty list of Parquet paths")
    logger.info_rank0(f"Building streaming dataset from {len(data_files)} parquet file(s)")
    gen = lambda: _iter_parquet_rows(data_files, ids_key, mask_key)
    return IterableDataset.from_generator(gen)  # type: ignore


