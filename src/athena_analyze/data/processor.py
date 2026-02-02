import pandas as pd
from typing import List, Tuple
from pathlib import Path

from statsmodels.tsa.seasonal import STL

from utils.logging import setup_logging

_log = setup_logging()

class DataProcessor:
    """
    DataProcessor の Docstring
    """

    def __init__(self, data_fol:str):
        self.data_folder = Path(data_fol)
        _log.debug(f"DataProcessor initialized with data folder: {self.data_folder}")
    
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
    
    def preprocess_data(self, df: pd.DataFrame, add_features: List[str]) -> pd.DataFrame:
        """
        データの前処理を行います。ここでは、欠損値の除去と重複行の削除を行います。

        Args:
            df (pd.DataFrame): 前処理を行う DataFrame

        Returns:
            pd.DataFrame: 前処理後の DataFrame
        """
        _log.debug("Starting data preprocessing")
        _df = df.dropna()
        _log.debug(f"Dropped NA values, remaining rows: {_df.shape[0]}")
        _df = _df.drop_duplicates()
        _log.debug(f"Dropped duplicate rows, remaining rows: {_df.shape[0]}")
        _df = self.add_features(_df, add_features)
        _log.info(f"Dropped {len(df) - len(_df)} rows, Added {len(add_features)} features")
        _log.info("Data preprocessing completed")
        return _df

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
            df['year'] = pd.to_datetime(df['date']).dt.year
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['day'] = pd.to_datetime(df['date']).dt.day
            df['weekday'] = pd.to_datetime(df['date']).dt.weekday
            df['yearmonth'] = pd.to_datetime(df['date']).dt.to_period('M')
            _log.debug("Added 'year', 'month', 'day', 'weekday', 'yearmonth' features")
        if 'season' in features and 'month' in df.columns:
            df['season'] = df['month'] % 12 // 3 + 1
            _log.debug("Added 'season' feature")
        _log.debug("Feature addition completed")
        return df
    
    def run_stl_decomposition(self, df: pd.DataFrame, date_col: str, target_col: str="OT") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        STL分解を実行し、分解された成分を DataFrame に追加します。

        :param df: ターゲットとなる DataFrame(Time Series を含む)
        :type df: pd.DataFrame
        :param date_col: 日付を示す列名
        :type date_col: str
        :param target_col: 分解対象の列名
        :type target_col: str
        :return: 分解結果を含む DataFrame のタプル
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """

        stl = STL(df[target_col], seasonal=13, trend=25, period=24, robust=True)
        result = stl.fit()
        df['trend'] = result.trend
        df['seasonal'] = result.seasonal
        df['resid'] = result.resid
        result_df = pd.DataFrame()
        result_df['trend'] = result.trend
        result_df['seasonal'] = result.seasonal
        result_df['resid'] = result.resid
        _log.debug("STL decomposition completed and components added to DataFrame")
        return (df, result_df)

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