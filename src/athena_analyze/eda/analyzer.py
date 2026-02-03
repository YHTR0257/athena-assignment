from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram
from statsmodels.tsa.stattools import acf, pacf

from utils.logging import setup_logging

_log = setup_logging()


def analyze_periodicity(df: pd.DataFrame, fs=1.0, max_period: Optional[int]=None,
                        target_col: str="OT", **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    データフレーム内の指定された列に対して周期性分析を実行します。
    与えられたデータフレームのdate列を時刻単位として用いるため、出力される周期も与えたデータの時間単位（ここでは時間）となります。

    :param df: 分析対象のデータフレーム（DatetimeIndexまたは'date'列を持つ）
    :type df: pd.DataFrame
    :param fs: サンプリング周波数（デフォルトは1.0）
    :type fs: float
    :param max_period: 最大周期。Noneの場合はデータ長の半分に自動設定
    :type max_period: Optional[int]
    :param target_col: 周期性分析を行う列名（デフォルトは"OT"）
    :type target_col: str
    :param kwargs: その他のオプション引数
        - top_n (int): 上位何個の周期を返すか（デフォルトは5）
    :type kwargs: dict
    :return: 検出された周期のデータフレーム、全パワースペクトルと全周期のデータフレーム
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]

    Examples:
    ---------
    >>> df = pd.read_csv('ETTh1.csv', parse_dates=['date'], index_col='date')
    >>> detected_df, period_power_df = analyze_periodicity(df, target_col='OT')
    >>> print(detected_df)
    """

    series = df[target_col].dropna()
    n = len(series)

    # max_period が未指定の場合、データ長の半分に設定
    if max_period is None:
        max_period = n // 2
    else:
        # データ長の半分を超える周期は検出が不安定なため制限
        max_period = min(max_period, n // 2)

    _log.debug(f"Analyzing periodicity for column: {target_col} with fs={fs} and max_period={max_period} (data length: {n})")

    if n < 10:
        _log.warning(f"Insufficient data points for periodicity analysis: {n}")
        return pd.DataFrame(), pd.DataFrame()
    
    # パワースペクトル計算
    frequencies, power = periodogram(series, fs=fs)
    
    # 周波数を周期に変換（0除算を避けるため最初の要素をスキップ）
    periods = 1 / frequencies[1:]
    power = power[1:]
    
    # 最大周期でフィルタリング
    _mask = periods <= max_period
    periods_filtered = periods[_mask]
    power_filtered = power[_mask]
    
    # パワースペクトルの周期に対するピークを検出
    peaks, _ = find_peaks(power_filtered, height=np.percentile(power_filtered, 90))
    
    # 検出された周期をリストに格納
    detected_periods = []
    for peak in peaks:
        if periods_filtered[peak] > 2:  # 2時間以上の周期のみ
            detected_periods.append({
                "period": periods_filtered[peak],
                "power": power_filtered[peak],
                "relative_power": power_filtered[peak] / np.max(power_filtered)
            })
    
    # パワーの大きい順にソート
    detected_periods = sorted(detected_periods, key=lambda x: x["power"], reverse=True)
    
    # 上位N個に制限
    top_n = kwargs.get("top_n", 5)
    detected_periods = detected_periods[:top_n]
    
    # DataFrameに変換
    detected_periods_df = pd.DataFrame(detected_periods)
    period_power_df = pd.DataFrame({
        "period": periods_filtered,
        "power": power_filtered
    })
    
    _log.info(f"Detected {len(detected_periods_df)} significant periods for {target_col}")
    if len(detected_periods_df) > 0:
        _log.debug(f"Top period: {detected_periods_df.iloc[0]['period']:.1f} "
                   f"(relative power: {detected_periods_df.iloc[0]['relative_power']:.3f})")
    
    return detected_periods_df, period_power_df


def compute_acf(df: pd.DataFrame, max_lag: Optional[int]=None,
                target_col: str="OT") -> pd.DataFrame:
    """
    自己相関関数（ACF）を計算します。

    :param df: 分析対象のデータフレーム
    :type df: pd.DataFrame
    :param max_lag: 最大ラグ。Noneの場合はデータ長の半分に自動設定
    :type max_lag: Optional[int]
    :param target_col: ACFを計算する列名（デフォルトは"OT"）
    :type target_col: str
    :return: ラグとACF値を含むデータフレーム
    :rtype: pd.DataFrame
    """

    series = df[target_col].dropna()
    n = len(series)

    if max_lag is None:
        max_lag = n // 2
    else:
        max_lag = min(max_lag, n // 2)

    _log.debug(f"Computing ACF for column: {target_col} with max_lag={max_lag} (data length: {n})")

    if n < 10:
        _log.warning(f"Insufficient data points for ACF: {n}")
        return pd.DataFrame()

    acf_values = acf(series, nlags=max_lag, fft=True)
    lags = np.arange(len(acf_values))

    return pd.DataFrame({"lag": lags, "acf": acf_values})


def compute_pacf(df: pd.DataFrame, max_lag: Optional[int]=None,
                 target_col: str="OT") -> pd.DataFrame:
    """
    偏自己相関関数（PACF）を計算します。

    :param df: 分析対象のデータフレーム
    :type df: pd.DataFrame
    :param max_lag: 最大ラグ。Noneの場合はデータ長の半分に自動設定（最大1000）
    :type max_lag: Optional[int]
    :param target_col: PACFを計算する列名（デフォルトは"OT"）
    :type target_col: str
    :return: ラグとPACF値を含むデータフレーム
    :rtype: pd.DataFrame
    """

    series = df[target_col].dropna()
    n = len(series)

    if max_lag is None:
        max_lag = min(n // 2, 1000)
    else:
        max_lag = min(max_lag, n // 2, 1000)

    _log.debug(f"Computing PACF for column: {target_col} with max_lag={max_lag} (data length: {n})")

    if n < 10:
        _log.warning(f"Insufficient data points for PACF: {n}")
        return pd.DataFrame()

    pacf_values = pacf(series, nlags=max_lag)
    lags = np.arange(len(pacf_values))

    return pd.DataFrame({"lag": lags, "pacf": pacf_values})
