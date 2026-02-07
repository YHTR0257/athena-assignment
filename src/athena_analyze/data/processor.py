import pandas as pd
from typing import List, Tuple
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from utils.array_backend import xp

from sklearn.discriminant_analysis import StandardScaler
from statsmodels.tsa.seasonal import MSTL

from utils.logging import setup_logging

_log = setup_logging()

class DataProcessor:
    """
    DataProcessor の Docstring
    """

    def __init__(self, data_fol:str):
        self.data_folder = Path(data_fol)
        self.target_col = "OT"
        _log.debug(f"DataProcessor initialized with data folder: {self.data_folder}")
        self.cols_to_use = ["OT", "HUFL", "HULL", "LUFL", "LULL", "MUFL", "MULL"]
    
    def load_data(self, file_name: str, **kwargs) -> pd.DataFrame:
        """
        指定されたファイル名の データを読み込み、DataFrame として返します。

        Args:
            file_name (str): 読み込む CSV ファイルの名前

        Returns:
            pd.DataFrame: 読み込んだデータの DataFrame
        """
        data_folder = kwargs.get("data_folder", self.data_folder)
        file_path = data_folder / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(file_path, engine='pyarrow')
        elif suffix == ".csv":
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

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
        if "moving_average" in method:
            ma_cfg = kwargs.get("moving_average", {})
            _train_df, _test_df = self.create_moving_averages(_train_df, _test_df, config=ma_cfg)
        if "sin_cos_features" in method:
            scf_cfg = kwargs.get("sin_cos_features", {})
            _train_df = self.sin_cos_features(_train_df, date_col="date", config=scf_cfg)
            _test_df = self.sin_cos_features(_test_df, date_col="date", config=scf_cfg)
        _log.info(f"Preprocessing methods applied: {method}")
        _log.info(f"Starting data normalization")

        norm_cfg = kwargs.get("normalization", {})
        # normコンフィグが存在する場合に正規化を実行
        if norm_cfg:
            _train_df, _test_df = self.normalize_data(_train_df, _test_df, config=norm_cfg)

        # データ型の変換（メモリ軽量化）
        dtype = kwargs.get("dtype", "float64")
        if dtype == "float32":
            _train_df = self._convert_float_dtype(_train_df, np.float32)
            _test_df = self._convert_float_dtype(_test_df, np.float32)
            _log.info("Converted numeric columns to float32 for memory optimization")

        _log.info("Data preprocessing completed")
        return _train_df, _test_df

    def create_moving_averages(self, train_df: pd.DataFrame, test_df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        指定された特徴量に対して移動平均特徴量を作成します。
        ターゲット変数（OT）は除外してデータリーケージを防ぎます。

        :param train_df: トレーニングデータ
        :type train_df: pd.DataFrame
        :param test_df: テストデータ
        :type test_df: pd.DataFrame
        :return: 移動平均特徴量が追加されたDataFrameのタプル
        :rtype: Tuple[DataFrame, DataFrame]
        """
        # ターゲット変数を除外してデータリーケージを防ぐ
        features = [col for col in self.cols_to_use if col != self.target_col]
        ma_cfg = kwargs.get("config", {})
        for feature in features:
            window_sizes = ma_cfg.get(feature, {}).get("windows", [3, 6, 12])
            for window in window_sizes:
                ma_col_name = f"{feature}_ma_{window}"
                train_df[ma_col_name] = train_df[feature].rolling(window=window, min_periods=1).mean()
                combined_values = pd.concat([train_df[feature].iloc[-window+1:], test_df[feature]], ignore_index=True)
                test_df[ma_col_name] = combined_values.rolling(window=window, min_periods=1).mean().iloc[window-1:].reset_index(drop=True)
            _log.debug(f"Created moving average features for {feature}")
        return train_df, test_df

    def create_lags(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    periods: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        指定された特徴量に対してラグ特徴量を作成します。
        ターゲット変数（OT）は除外してデータリーケージを防ぎます。

        Args:
            train_df (pd.DataFrame): トレーニングセットの DataFrame
            test_df (pd.DataFrame): テストセットの DataFrame
            periods (List[int]): 作成するラグの期間のリスト

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: ラグ特徴量が追加されたトレーニングセットとテストセットのタプル
        """
        max_period = max(periods)
        # ターゲット変数を除外してデータリーケージを防ぐ
        features = [col for col in self.cols_to_use if col != self.target_col]
        for feature in features:
            train_values = train_df[feature].to_numpy()
            test_values = test_df[feature].to_numpy()
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
            _log.debug("Added 'year', 'month', 'day', 'weekday' features")
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

        # Extrapolate trend to test_df using least squares linear fit
        # Fit on the last max(periods) points of the trend
        max_period = max(periods)
        trend_values = trend_df.values
        fit_segment = np.asarray(trend_values[-max_period:], dtype=np.float64)
        x_fit = np.arange(max_period, dtype=np.float64)
        slope, intercept = np.polyfit(x_fit, fit_segment, 1)
        x_test = np.arange(max_period, max_period + len(test_df))
        test_trend = slope * x_test + intercept
        test_df[f'{prefix}trend'] = test_trend

        # Fill residuals with 0 for test_df
        test_df[f'{prefix}resid'] = 0.0

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
    
    def sin_cos_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        指定された時刻に基づいてサイン・コサイン特徴量を追加します。

        :param self: 説明
        :param df: 時刻情報が含まれる DataFrame。すでに date_col が datetime 型であることを前提としています。
        :type df: pd.DataFrame
        :param date_col: 日付を示す列名（デフォルトは "date"）
        :return: サイン・コサイン特徴量が追加された DataFrame
        :rtype: pd.DataFrame
        """

        date_col = kwargs.get("date_col", "date")

        periods = kwargs.get("config", {}).get("periods", [24, 168, 2160])
        for p in periods:
            df[f'sin_{p}'] = np.sin(2 * np.pi * df[date_col].dt.hour / p)
            df[f'cos_{p}'] = np.cos(2 * np.pi * df[date_col].dt.hour / p)
            _log.debug(f"Added sin and cos features for period: {p}")
        return df

    def normalize_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        normalize_data の Docstring
        
        :param self: 説明
        :param train_df: 説明
        :type train_df: pd.DataFrame
        :param test_df: 説明
        :type test_df: pd.DataFrame
        :param kwargs: 説明
        :return: 説明
        :rtype: Tuple[DataFrame, DataFrame]
        """

        train_cols = train_df.select_dtypes(include=[np.number]).columns
        test_cols = test_df.select_dtypes(include=[np.number]).columns
        common_cols = train_cols.intersection(test_cols)

        method = kwargs.get("config", {}).get("method", "standard")
        if method == "standard":
            for col in common_cols:
                scaler = StandardScaler()
                scaler.fit(train_df[[col]])
                train_df[col] = scaler.transform(train_df[[col]])
                test_df[col] = scaler.transform(test_df[[col]])
            scaler = StandardScaler()
            scaler.fit(train_df[[self.target_col]])
            train_df[self.target_col] = scaler.transform(train_df[[self.target_col]])
            _log.debug("Standard normalization applied")
        else:
            _log.warning(f"Normalization method '{method}' not recognized. No normalization applied.")

        return train_df, test_df

    def _convert_float_dtype(self, df: pd.DataFrame, dtype: np.dtype) -> pd.DataFrame:
        """
        DataFrameの浮動小数点カラムを指定された型に変換します。

        Args:
            df (pd.DataFrame): 変換対象のDataFrame
            dtype (np.dtype): 変換先の型（np.float32など）

        Returns:
            pd.DataFrame: 型変換後のDataFrame
        """
        float_cols = df.select_dtypes(include=[np.float64]).columns
        for col in float_cols:
            df[col] = df[col].astype(dtype)
        return df


class TimeSeriesDataset(Dataset):
    """
    時系列データのPyTorchデータセットクラス
    """

    def __init__(self, X: xp.ndarray, y: xp.ndarray):
        """
        Docstring for __init__
        
        :param self: Description
        :param X: Description
        :type X: xp.ndarray
        :param y: Description
        :type y: xp.ndarray
        """
        self.X = X
        self.y = y
    
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[xp.ndarray, xp.ndarray]:
        return self.X[idx], self.y[idx]

def create_sliding_window_dataset(df: pd.DataFrame, windows_size: int,
                                  horizon: int, stride: int) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    スライディングウィンドウを使用して時系列データセットを作成します。
    
    :param df: Description
    :type df: pd.DataFrame
    :param windows_size: Description
    :type windows_size: int
    :param horizon: Description
    :type horizon: int
    :param stride: Description
    :type stride: int
    :return: Description
    :rtype: Tuple[Any, Any]
    """
    data = df.values
    n_samples = (len(data) - windows_size - horizon) // stride + 1

    X_list = []
    y_list = []

    for i in range(n_samples):
        start_idx = i * stride
        end_idx = start_idx + windows_size
        target_end_idx = end_idx + horizon

        if target_end_idx <= len(data):
            X_list.append(data[start_idx:end_idx])
            y_list.append(data[end_idx:target_end_idx])
    
    X = xp.array(X_list)
    y = xp.array(y_list)

    return X, y

def split_train_eval(
        X: xp.ndarray, y: xp.ndarray,
        train_ratio: float=0.8) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    トレーニングセットと評価セットにデータを分割します。
    """
    n_samples = len(X)
    split_idx = int(n_samples * train_ratio)

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_eval, y_eval = X[split_idx:], y[split_idx:]

    return X_train, y_train, X_eval, y_eval

def create_datasets(
    df: pd.DataFrame, window_size: int,
    horizon: int, stride: int,
    train_ratio: float=0.8
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset]:
    """
    Docstring for create_datasets
    
    :param df: Description
    :type df: pd.DataFrame
    :param window_size: Description
    :type window_size: int
    :param horizon: Description
    :type horizon: int
    :param stride: Description
    :type stride: int
    :param train_ratio: Description
    :type train_ratio: float
    :return: Description
    :rtype: Tuple[TimeSeriesDataset, TimeSeriesDataset]
    """
    X, y = create_sliding_window_dataset(
        df, windows_size=window_size,
        horizon=horizon, stride=stride
    )

    X_train, y_train, X_eval, y_eval = split_train_eval(
        X, y, train_ratio=train_ratio
    )

    train_dataset = TimeSeriesDataset(X_train, y_train)
    eval_dataset = TimeSeriesDataset(X_eval, y_eval)

    _log.debug(f"Created datasets with {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples")
    _log.debug(f"Each sample shape: X={X_train.shape[1:]}, y={y_train.shape[1:]}")

    return train_dataset, eval_dataset