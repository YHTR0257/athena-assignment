import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from typing import Union

from utils.config import load_config_section
from utils.logging import setup_logging

_log = setup_logging()

def save_dataframe_to_parquet(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """Save a DataFrame to a Parquet file.

    Args:
        df: DataFrame to save
        file_path: Path to save the Parquet file
    """
    data_type = kwargs.get("data_type", "experiment")
    data_cfg = load_config_section("config/config.yml", "data")
    save_folder = data_cfg.get(data_type, "./data")
    path = Path(save_folder) / file_path
    # File Suffix check
    if path.suffix.lower() != ".parquet":
        path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False,
                  coerce_timestamps='us',
                  engine='pyarrow',
                  allow_truncated_timestamps=True)
    _log.info(f"DataFrame saved to {path}")
