import pandas as pd
from typing import List, Tuple
from pathlib import Path
import numpy as np

from statsmodels.tsa.seasonal import MSTL

from utils.logging import setup_logging

_log = setup_logging()

class DataProcessor:
    """
    DataProcessor の Docstring
    """

    def __init__(self, data_fol:str):
        self.data_folder = Path(data_fol)
        _log.debug(f"DataProcessor initialized with data folder: {self.data_folder}")
        self.cols_to_use = ["OT", "HUFL", "HULL", "LUFL", "LULL", "MUFL", "MULL"]
    
    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        指定されたファイル名の CSV データを読み込み、DataFrame として返します。

        Args:
            file_name (str): 読み込む CSV ファイルの名前

        Returns:
            pd.DataFrame: 読み込んだデータの DataFrame
        """
        file_path = self.data_folder / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    def split_data(self, df: pd.DataFrame, date_col: str="date", **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        データを指定された日付でtrain, testに分割します。

        Args:
            df (pd.DataFrame): 分割する DataFrame
            date_col (str): 日付を示す列名（デフォルトは "date"）

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: トレーニングセットとテストセットのタプル
        """
        split_date = kwargs.get("split_date", {})
        df[date_col] = pd.to_datetime(df[date_col])
        train_df = df[(df[date_col] >= split_date.get("train_start")) & (df[date_col] <= split_date.get("train_end"))].reset_index(drop=True)
        test_df = df[(df[date_col] >= split_date.get("test_start")) & (df[date_col] <= split_date.get("test_end"))].reset_index(drop=True)
        _log.debug(f"Data split at {split_date}: {len(train_df)} training rows, {len(test_df)} testing rows")
        return train_df, test_df
    
    def preprocess_data(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        データの前処理を行います。ここでは、欠損値の除去と重複行の削除を行います。

        Args:
            df (pd.DataFrame): 前処理を行う DataFrame

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 前処理後のトレーニングセットとテストセットのタプル
        """
        method = kwargs.get("method", [])
        add_features = kwargs.get("add_features", [])

        _log.debug("Starting data preprocessing")
        _df = df.dropna()
        _log.debug(f"Dropped NA values, remaining rows: {_df.shape[0]}")
        _df = _df.drop_duplicates()
        _log.debug(f"Dropped duplicate rows, remaining rows: {_df.shape[0]}")

        _log.info("Data Cleaning completed, starting Data Splitting")
        _train_df, _test_df = self.split_data(_df, date_col="date", split_date=kwargs.get("split_date", {}))
        _log.info("Data Splitting completed")

        _train_df = self.add_features(_train_df, add_features)
        _test_df = self.add_features(_test_df, add_features)
        _log.info(f"Dropped {len(df) - len(_df)} rows, Added {len(add_features)} features")
        if "mstl" in method:
            mstl_cfg = kwargs.get("mstl", {})
            for col in self.cols_to_use:
                if col in _train_df.columns:
                    _train_df[col] = pd.to_numeric(_train_df[col], errors='coerce')
                    _log.debug(f"Converted column {col} to numeric")
                    _cfg = dict(**mstl_cfg.get("general", {}), **mstl_cfg.get(col, {}))
                    _train_df, _test_df = self.run_mstl_decomposition(_train_df, _test_df,
                                                            date_col="date", target_col=col,
                                                            prefix=f"stl_{col}_",
                                                            config=_cfg)
        if "lag" in method:
            lag_cfg = kwargs.get("lag", {})
            lag_periods = lag_cfg.get("periods", [1, 2, 3, 6, 12, 24])
            _train_df, _test_df = self.create_lags(_train_df, _test_df, lag_periods)
        _log.info("Data preprocessing completed")
        return _train_df, _test_df

    def create_lags(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    periods: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        指定された特徴量に対してラグ特徴量を作成します。

        Args:
            train_df (pd.DataFrame): トレーニングセットの DataFrame
            test_df (pd.DataFrame): テストセットの DataFrame
            periods (List[int]): 作成するラグの期間のリスト

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: ラグ特徴量が追加されたトレーニングセットとテストセットのタプル
        """
        max_period = max(periods)
        for feature in self.cols_to_use:
            train_values = train_df[feature].values
            test_values = test_df[feature].values
            bridge_values = train_values[-max_period:]
            combined_values = np.concatenate([bridge_values, test_values])

            for period in periods:
                lag_col_name = f"{feature}_lag_{period}"
                train_df[lag_col_name] = np.concatenate([
                    np.full(period, np.nan),
                    train_values[:-period]
                ])
                test_df[lag_col_name] = combined_values[max_period - period:-period]
            _log.debug(f"Created lag features for {feature}")
        return train_df, test_df

    def add_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        指定された特徴量を DataFrame に追加します。

        Args:
            df (pd.DataFrame): 特徴量を追加する DataFrame
            features (List[str]): 追加する特徴量のリスト

        Returns:
            pd.DataFrame: 特徴量が追加された DataFrame
        """
        _log.debug(f"Adding features: {features}")
        if 'date_features' in features and 'date' in df.columns:
            dt = pd.to_datetime(df['date'])
            df['year'] = dt.dt.year
            df['month'] = dt.dt.month
            df['day'] = dt.dt.day
            df['weekday'] = dt.dt.weekday
            df['yearmonth'] = dt.dt.to_period('M')
            _log.debug("Added 'year', 'month', 'day', 'weekday', 'yearmonth' features")
        if 'season' in features and 'month' in df.columns:
            df['season'] = df['month'] % 12 // 3 + 1
            _log.debug("Added 'season' feature")
        _log.debug("Feature addition completed")
        return df
    
    def run_mstl_decomposition(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                               target_col: str="OT", **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        MSTL分解を実行し、分解された成分を DataFrame に追加します。
        Y = Trend + S_1 + S_2 + S_3 + ... + Residual

        Args:
            train_df (pd.DataFrame): トレーニングセットの DataFrame
            test_df (pd.DataFrame): テストセットの DataFrame
            target_col (str): 分解対象の列名（デフォルトは "OT"）
            **kwargs: prefix (str), config (dict) などのオプション

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 分解成分が追加されたトレーニングセットとテストセットのタプル
        """
        prefix = kwargs.get("prefix", "")
        cfg = kwargs.get("config", {})
        periods = cfg.get("periods", [])

        if not periods:
            raise ValueError("MSTL decomposition requires 'periods' in config")

        _log.debug(f"Running MSTL decomposition on column: {target_col} with config: {cfg}")
        mstl = MSTL(train_df[target_col], **cfg)
        result = mstl.fit()
        seasonal_df = result.seasonal
        trend_df = result.trend

        # Validate seasonal_df structure
        if not isinstance(seasonal_df, pd.DataFrame):
            raise ValueError(
                f"MSTL returned Series instead of DataFrame. "
                f"Expected DataFrame with columns for periods {periods}"
            )

        # Validate all expected seasonal columns exist
        for period in periods:
            col_n = f'seasonal_{period}'
            if col_n not in seasonal_df.columns:
                raise KeyError(
                    f"Expected column '{col_n}' not found in MSTL result. "
                    f"Available columns: {list(seasonal_df.columns)}. "
                    f"Check if 'periods' config matches MSTL output."
                )

        # Add trend to train_df
        train_df[f'{prefix}trend'] = trend_df

        # Add seasonal components to train_df
        for seasonal_col in seasonal_df.columns:
            train_df[f'{prefix}{seasonal_col}'] = seasonal_df[seasonal_col]

        # Add residuals to train_df
        train_df[f'{prefix}resid'] = result.resid

        # Build seasonal lookup arrays for each period
        seasonal_lookup = {}
        for period in periods:
            col_n = f'seasonal_{period}'
            positions = np.arange(len(seasonal_df)) % period
            lookup = np.empty(period)
            for pos in range(period):
                mask = positions == pos
                if mask.any():
                    lookup[pos] = seasonal_df[col_n].values[mask][-1]
                else:
                    lookup[pos] = np.nan
            seasonal_lookup[period] = lookup

        # Apply seasonal patterns to test_df using numpy indexing
        start_idx = len(train_df)
        for period in periods:
            col_n = f'seasonal_{period}'
            test_positions = (np.arange(start_idx, start_idx + len(test_df)) % period)
            test_df[f'{prefix}{col_n}'] = seasonal_lookup[period][test_positions]

            # Validate mapping succeeded
            if np.isnan(test_df[f'{prefix}{col_n}'].values).any():
                raise ValueError(
                    f"Failed to map seasonal pattern for period {period} to test_df. "
                    f"Some positions could not be mapped."
                )

        _log.debug("MSTL decomposition completed and components added to DataFrame")
        return train_df, test_df
    
    def describe_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame の基本統計量を計算し、返します。

        Args:
            df (pd.DataFrame): 統計量を計算する DataFrame

        Returns:
            pd.DataFrame: 基本統計量の DataFrame
        """
        _log.debug("Calculating data description")
        description = df.describe()
        _log.debug("Data description calculated")
        return description
