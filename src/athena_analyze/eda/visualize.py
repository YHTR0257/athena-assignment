import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.signal import find_peaks
from typing import List, Tuple

from utils.logging import setup_logging

_log = setup_logging()

def plot_time_series(df:pd.DataFrame, date_col:str, value_cols:List[str], **kwargs) -> Figure:
    """
    plot_time_series の Docstring
    時系列データのプロットを作成
    
    :param df: 時系列データを含む DataFrame
    :type df: pd.DataFrame
    :param date_col: 日付を表す列名
    :type date_col: str
    :param value_cols: プロットする値の列名リスト
    :type value_cols: List[str]
    :return: プロットされた Figure オブジェクト
    :rtype: Figure
    """

    _log.debug(f"Plotting time series for columns: {value_cols} with date column: {date_col}")

    plot_df = df.copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    plot_df = plot_df.set_index(date_col)

    figsize = kwargs.get("figsize", (10, 5))
    fig, ax = plt.subplots(figsize=figsize)

    for col in value_cols:
        ax.plot(plot_df.index, plot_df[col], label=col)
    ax.set_xlabel(kwargs.get("xlabel", "Date"))

    fig.tight_layout()
    ax.legend()
    return fig

def plot_power_spectrum(df: pd.DataFrame, **kwargs) -> Tuple[Figure, Axes]:
    """
    パワースペクトルのプロットを作成。絶対値と相対値の2つのプロットを表示。
    横軸に周期を、縦軸にパワースペクトルをプロット。どの周期成分が強いのかを視覚化。
    検出されたピークを自動的にマークし、主要な周期を視覚的に強調表示。
    
    :param df: パワースペクトルのデータフレーム（周期とパワーを含む）
    :type df: pd.DataFrame
    :param detected_peaks: analyze_periodicity()で検出されたピークのデータフレーム（オプション）
    :type detected_peaks: pd.DataFrame
    :param kwargs: 追加のプロットオプション
        - figsize: 図のサイズ (default: (15, 10))
        - periods_col: 周期の列名 (default: "period")
        - power_col: パワーの列名 (default: "power")
        - max_period: 表示する最大周期 (default: 200)
        - peak_threshold: ピーク検出の閾値（パーセンタイル） (default: 90)
        - show_peak_labels: ピークラベルを表示するか (default: True)
        - show_known_periods: 既知の周期（12h, 24h, 168h）をマークするか (default: True)
        - top_n_peaks: 表示する上位N個のピーク (default: 5)
    :return: 図とAxesオブジェクトのタプル
    :rtype: Tuple[Figure, Axes]
    
    Examples:
    ---------
    >>> detected_df, all_periods, all_power = analyze_periodicity(df, target_col='OT')
    >>> power_df = pd.DataFrame({'period': all_periods, 'power': all_power})
    >>> fig, axes = plot_power_spectrum(power_df, detected_peaks=detected_df)
    """
    
    # パラメータ取得
    figsize = kwargs.get("figsize", (8, 6))
    periods_col = kwargs.get("periods_col", "period")
    power_col = kwargs.get("power_col", "power")
    max_period = kwargs.get("max_period", 1000)
    peak_threshold = kwargs.get("peak_threshold", 90)
    show_peak_labels = kwargs.get("show_peak_labels", True)
    show_known_periods = kwargs.get("show_known_periods", True)
    top_n_peaks = kwargs.get("top_n_peaks", 5)
    
    # 相対パワーの計算
    df = df.copy()
    df["relative_power"] = df[power_col] / df[power_col].max()
    
    # 表示範囲でフィルタリング
    df_plot = df[df[periods_col] <= max_period].copy()
    
    # ピーク検出（detected_peaksが提供されていない場合）
    peaks, properties = find_peaks(
        df_plot[power_col].values, 
        height=np.percentile(df_plot[power_col].values, peak_threshold)
    )
    
    # 検出されたピークをDataFrameに変換
    detected_peaks = pd.DataFrame({
        'period_hours': df_plot.iloc[peaks][periods_col].values,
        'power': df_plot.iloc[peaks][power_col].values,
        'relative_power': df_plot.iloc[peaks]["relative_power"].values
    })
    detected_peaks = detected_peaks.sort_values('power', ascending=False).head(top_n_peaks)
    
    # プロット作成
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    
    # カラーマップの準備
    if len(detected_peaks) > 0:
        colors = plt.cm.Set1(np.linspace(0, 1, len(detected_peaks)))
    
    # 既知の周期
    known_periods = {12: '12h\n(semi-daily)', 24: '24h\n(daily)', 168: '168h\n(weekly)'}
    
    # ===== 1. 絶対パワースペクトル =====
    axes[0].semilogy(df_plot[periods_col], df_plot[power_col], 
                     linewidth=1.5, color='navy', alpha=0.7, label='Power Spectrum')
    axes[0].set_title(kwargs.get("title_abs", "Power Spectrum (Absolute)"), 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel(kwargs.get("xlabel_abs", "Period (hours)"), fontsize=12)
    axes[0].set_ylabel(kwargs.get("ylabel_abs", "Power"), fontsize=12)
    axes[0].grid(True, alpha=0.3, which='both')

    # 検出されたピークをマーク
    if len(detected_peaks) > 0:
        for idx, (_, peak) in enumerate(detected_peaks.iterrows()):
            period = peak['period_hours']
            power = peak['power']
            
            # ピーク位置にマーカー
            axes[0].plot(period, power, 'o', color=colors[idx], 
                        markersize=12, markeredgecolor='black', 
                        markeredgewidth=1.5, zorder=5,
                        label=f'Peak: {period:.0f}h')
            
            # 垂直線
            axes[0].axvline(x=period, color=colors[idx], 
                          linestyle='--', alpha=0.6, linewidth=2)
            
            # ラベル表示
            if show_peak_labels:
                y_pos = power * 1.5
                axes[0].text(period, y_pos, 
                           f'{period:.0f}h\n({peak["relative_power"]:.2f})',
                           ha='center', va='bottom', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=colors[idx], alpha=0.3,
                                   edgecolor='black', linewidth=1))
    
    # 既知の周期をマーク
    if show_known_periods:
        for period, label in known_periods.items():
            if period <= max_period:
                axes[0].axvline(x=period, color='red', linestyle=':', 
                              alpha=0.4, linewidth=2)
                y_pos = axes[0].get_ylim()[1] * 0.3
                axes[0].text(period, y_pos, label, 
                           ha='center', va='center', fontsize=8,
                           color='red', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', alpha=0.7,
                                   edgecolor='red', linewidth=1))
    
    axes[0].legend(loc='upper right', fontsize=9, ncol=2)
    axes[0].set_xlim([df_plot[periods_col].min(), max_period])
    
    # ===== 2. 相対パワースペクトル =====
    axes[1].plot(df_plot[periods_col], df_plot["relative_power"], 
                     linewidth=1.5, color='navy', alpha=0.7, label='Relative Power')
    axes[1].set_title(kwargs.get("title_rel", "Power Spectrum (Relative)"), 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel(kwargs.get("xlabel_rel", "Period (hours)"), fontsize=12)
    axes[1].set_ylabel(kwargs.get("ylabel_rel", "Relative Power"), fontsize=12)
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3, which='both')
    
    # 検出されたピークをマーク
    if len(detected_peaks) > 0:
        for idx, (_, peak) in enumerate(detected_peaks.iterrows()):
            period = peak['period_hours']
            rel_power = peak['relative_power']
            
            # ピーク位置にマーカー
            axes[1].plot(period, rel_power, 'o', color=colors[idx], 
                        markersize=12, markeredgecolor='black', 
                        markeredgewidth=1.5, zorder=5,
                        label=f'Peak: {period:.0f}h')
            
            # 垂直線
            axes[1].axvline(x=period, color=colors[idx], 
                          linestyle='--', alpha=0.6, linewidth=2)
            
            # ラベル表示
            if show_peak_labels:
                y_pos = rel_power * 1.5
                axes[1].text(period, y_pos, 
                           f'{period:.0f}h\n({rel_power:.2f})',
                           ha='center', va='bottom', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=colors[idx], alpha=0.3,
                                   edgecolor='black', linewidth=1))
    
    # 既知の周期をマーク
    if show_known_periods:
        for period, label in known_periods.items():
            if period <= max_period:
                axes[1].axvline(x=period, color='red', linestyle=':', 
                              alpha=0.4, linewidth=2)
    
    # 相対パワーの閾値ライン（強度判定用）
    axes[1].axhline(y=0.6, color='green', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label='Strong seasonality (0.6)')
    axes[1].axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label='Moderate seasonality (0.3)')
    
    axes[1].legend(loc='upper right', fontsize=9, ncol=2)
    axes[1].set_xlim([df_plot[periods_col].min(), max_period])
    axes[1].set_ylim([1e-3, 2])  # 相対パワーの範囲
    
    fig.tight_layout()
    return fig, axes


def plot_acf(df: pd.DataFrame, **kwargs) -> Tuple[Figure, Axes]:
    """
    自己相関関数（ACF）のプロットを作成。
    横軸にラグ（周期）、縦軸にACF値をプロット。
    ピークを自動検出してマークし、既知の周期に参照線を表示。

    :param df: ACFのデータフレーム（lag列とacf列を含む）
    :type df: pd.DataFrame
    :param kwargs: 追加のプロットオプション
        - figsize: 図のサイズ (default: (15, 6))
        - lag_col: ラグの列名 (default: "lag")
        - acf_col: ACFの列名 (default: "acf")
        - max_lag: 表示する最大ラグ (default: None, 全て表示)
        - peak_threshold: ピーク検出の閾値 (default: 0.1)
        - top_n_peaks: 表示する上位N個のピーク (default: 5)
        - show_known_periods: 既知の周期をマークするか (default: True)
        - known_periods: 既知の周期のdict {lag: label} (default: {24: '24h', 168: '168h', 720: '30d', 2160: '90d'})
    :return: 図とAxesオブジェクトのタプル
    :rtype: Tuple[Figure, Axes]
    """

    figsize = kwargs.get("figsize", (10, 5))
    lag_col = kwargs.get("lag_col", "lag")
    acf_col = kwargs.get("acf_col", "acf")
    max_lag = kwargs.get("max_lag", None)
    peak_threshold = kwargs.get("peak_threshold", 0.1)
    top_n_peaks = kwargs.get("top_n_peaks", 5)
    show_known_periods = kwargs.get("show_known_periods", True)
    known_periods = kwargs.get("known_periods", {24: '24h', 168: '168h (1w)', 720: '720h (30d)', 2160: '2160h (90d)'})

    df_plot = df.copy()
    if max_lag is not None:
        df_plot = df_plot[df_plot[lag_col] <= max_lag]

    fig, ax = plt.subplots(figsize=figsize)

    # ACFプロット
    ax.plot(df_plot[lag_col], df_plot[acf_col], linewidth=1.5, color='navy', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # ピーク検出（lag > 0 の部分のみ）
    acf_values = df_plot[acf_col].values[1:]  # lag=0を除く
    lags = df_plot[lag_col].values[1:]
    peaks, _ = find_peaks(acf_values, height=peak_threshold, distance=12)

    if len(peaks) > 0:
        peak_data = [(lags[p], acf_values[p]) for p in peaks]
        peak_data = sorted(peak_data, key=lambda x: x[1], reverse=True)[:top_n_peaks]

        colors = plt.cm.Set1(np.linspace(0, 1, len(peak_data)))
        for idx, (lag, acf_val) in enumerate(peak_data):
            ax.plot(lag, acf_val, 'o', color=colors[idx],
                    markersize=10, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=5)
            ax.axvline(x=lag, color=colors[idx], linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(lag, acf_val + 0.05, f'{lag:.0f}',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[idx], alpha=0.3))

    # 既知の周期をマーク
    if show_known_periods:
        max_display_lag = df_plot[lag_col].max()
        for period, label in known_periods.items():
            if period <= max_display_lag:
                ax.axvline(x=period, color='red', linestyle=':', alpha=0.5, linewidth=2)
                ax.text(period, ax.get_ylim()[1] * 0.9, label,
                        ha='center', va='top', fontsize=8, color='red',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_title(kwargs.get("title", "Autocorrelation Function (ACF)"), fontsize=14, fontweight='bold')
    ax.set_xlabel(kwargs.get("xlabel", "Lag"), fontsize=12)
    ax.set_ylabel(kwargs.get("ylabel", "ACF"), fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_pacf(df: pd.DataFrame, **kwargs) -> Tuple[Figure, Axes]:
    """
    偏自己相関関数（PACF）のプロットを作成。
    横軸にラグ、縦軸にPACF値をプロット。
    ピークを自動検出してマークし、既知の周期に参照線を表示。

    :param df: PACFのデータフレーム（lag列とpacf列を含む）
    :type df: pd.DataFrame
    :param kwargs: 追加のプロットオプション
        - figsize: 図のサイズ (default: (15, 6))
        - lag_col: ラグの列名 (default: "lag")
        - pacf_col: PACFの列名 (default: "pacf")
        - max_lag: 表示する最大ラグ (default: None, 全て表示)
        - peak_threshold: ピーク検出の閾値 (default: 0.1)
        - top_n_peaks: 表示する上位N個のピーク (default: 5)
        - show_known_periods: 既知の周期をマークするか (default: True)
        - known_periods: 既知の周期のdict {lag: label} (default: {24: '24h', 168: '168h', 720: '30d', 2160: '90d'})
    :return: 図とAxesオブジェクトのタプル
    :rtype: Tuple[Figure, Axes]
    """

    figsize = kwargs.get("figsize", (10, 5))
    lag_col = kwargs.get("lag_col", "lag")
    pacf_col = kwargs.get("pacf_col", "pacf")
    max_lag = kwargs.get("max_lag", None)
    peak_threshold = kwargs.get("peak_threshold", 0.1)
    top_n_peaks = kwargs.get("top_n_peaks", 5)
    show_known_periods = kwargs.get("show_known_periods", True)
    known_periods = kwargs.get("known_periods", {24: '24h', 168: '168h (1w)', 720: '720h (30d)', 2160: '2160h (90d)'})

    df_plot = df.copy()
    if max_lag is not None:
        df_plot = df_plot[df_plot[lag_col] <= max_lag]

    fig, ax = plt.subplots(figsize=figsize)

    # PACFプロット
    ax.plot(df_plot[lag_col], df_plot[pacf_col], linewidth=1.5, color='darkgreen', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # ピーク検出（lag > 0 の部分のみ、正負両方）
    pacf_values = df_plot[pacf_col].values[1:]  # lag=0を除く
    lags = df_plot[lag_col].values[1:]

    # 正のピーク
    peaks_pos, _ = find_peaks(pacf_values, height=peak_threshold, distance=12)
    # 負のピーク
    peaks_neg, _ = find_peaks(-pacf_values, height=peak_threshold, distance=12)

    all_peaks = []
    for p in peaks_pos:
        all_peaks.append((lags[p], pacf_values[p], abs(pacf_values[p])))
    for p in peaks_neg:
        all_peaks.append((lags[p], pacf_values[p], abs(pacf_values[p])))

    if len(all_peaks) > 0:
        all_peaks = sorted(all_peaks, key=lambda x: x[2], reverse=True)[:top_n_peaks]

        colors = plt.cm.Set1(np.linspace(0, 1, len(all_peaks)))
        for idx, (lag, pacf_val, _) in enumerate(all_peaks):
            ax.plot(lag, pacf_val, 'o', color=colors[idx],
                    markersize=10, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=5)
            ax.axvline(x=lag, color=colors[idx], linestyle='--', alpha=0.5, linewidth=1.5)
            offset = 0.05 if pacf_val >= 0 else -0.05
            va = 'bottom' if pacf_val >= 0 else 'top'
            ax.text(lag, pacf_val + offset, f'{lag:.0f}',
                    ha='center', va=va, fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[idx], alpha=0.3))

    # 既知の周期をマーク
    if show_known_periods:
        max_display_lag = df_plot[lag_col].max()
        for period, label in known_periods.items():
            if period <= max_display_lag:
                ax.axvline(x=period, color='red', linestyle=':', alpha=0.5, linewidth=2)
                ax.text(period, ax.get_ylim()[1] * 0.9, label,
                        ha='center', va='top', fontsize=8, color='red',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_title(kwargs.get("title", "Partial Autocorrelation Function (PACF)"), fontsize=14, fontweight='bold')
    ax.set_xlabel(kwargs.get("xlabel", "Lag"), fontsize=12)
    ax.set_ylabel(kwargs.get("ylabel", "PACF"), fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax
