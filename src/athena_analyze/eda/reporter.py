import pandas as pd
import io
from typing import Optional

from utils.logging import setup_logging

_log = setup_logging()


def write_timeseries_report(df: pd.DataFrame, output_path: str, date_col: str = 'date') -> None:
    """
    時系列データに関する情報をMarkdown形式でレポートとして保存します。

    Args:
        df (pd.DataFrame): レポート対象の DataFrame
        output_path (str): 出力先のパス
        date_col (str): 日付カラム名（デフォルト: 'date'）
    """
    _log.debug(f"Writing report to: {output_path}")

    with open(output_path, 'w') as f:
        f.write("# Time Series Data Report\n\n")

        # 基本情報
        f.write("## 1. 基本情報\n\n")
        f.write(f"- **レコード数**: {len(df):,}\n")
        f.write(f"- **カラム数**: {len(df.columns)}\n")
        f.write(f"- **カラム一覧**: {', '.join(df.columns.tolist())}\n\n")

        # 時系列情報
        if date_col in df.columns:
            f.write("## 2. 時系列情報\n\n")
            dates = pd.to_datetime(df[date_col])
            f.write(f"- **期間**: {dates.min()} ～ {dates.max()}\n")
            f.write(f"- **データ日数**: {dates.nunique():,} 日\n")

            # 頻度推定
            date_diff = dates.sort_values().diff().dropna()
            if len(date_diff) > 0:
                median_diff = date_diff.median()
                if median_diff <= pd.Timedelta(days=1):
                    freq_str = "日次"
                elif median_diff <= pd.Timedelta(days=7):
                    freq_str = "週次"
                elif median_diff <= pd.Timedelta(days=31):
                    freq_str = "月次"
                else:
                    freq_str = f"{median_diff.days}日間隔"
                f.write(f"- **推定頻度**: {freq_str}\n")
                f.write(f"- **中央値間隔**: {median_diff}\n")

            # 欠損期間の検出
            if len(date_diff) > 0:
                expected_diff = date_diff.mode().iloc[0] if len(date_diff.mode()) > 0 else median_diff
                gaps = date_diff[date_diff > expected_diff * 1.5]
                if len(gaps) > 0:
                    f.write(f"- **欠損期間数**: {len(gaps)}\n")
            f.write("\n")

            # 年月別データ件数
            f.write("### 年月別データ件数\n\n")
            monthly_counts = df.groupby(dates.dt.to_period('M')).size()
            f.write("| 年月 | 件数 |\n")
            f.write("|------|------|\n")
            for period, count in monthly_counts.items():
                f.write(f"| {period} | {count:,} |\n")
            f.write("\n")

        # 欠損値情報
        f.write("## 3. 欠損値情報\n\n")
        missing = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        f.write("| カラム | 欠損数 | 欠損率(%) |\n")
        f.write("|--------|--------|----------|\n")
        for col in df.columns:
            f.write(f"| {col} | {missing[col]:,} | {missing_pct[col]:.2f} |\n")
        f.write("\n")

        # 数値カラムの統計量
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            f.write("## 4. 数値カラムの統計量\n\n")
            f.write("```\n")
            f.write(str(df[numeric_cols].describe()))
            f.write("\n```\n\n")

            # 時系列トレンド情報
            if date_col in df.columns:
                f.write("### 数値カラムの時系列トレンド\n\n")
                f.write("| カラム | 最初の値 | 最後の値 | 変化率(%) |\n")
                f.write("|--------|----------|----------|----------|\n")
                sorted_df = df.sort_values(date_col)
                for col in numeric_cols:
                    first_val = sorted_df[col].iloc[0]
                    last_val = sorted_df[col].iloc[-1]
                    if first_val != 0:
                        change_pct = ((last_val - first_val) / first_val * 100)
                        f.write(f"| {col} | {first_val:.2f} | {last_val:.2f} | {change_pct:+.2f} |\n")
                    else:
                        f.write(f"| {col} | {first_val:.2f} | {last_val:.2f} | N/A |\n")
                f.write("\n")

        # カテゴリカラムの情報
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if date_col in cat_cols:
            cat_cols.remove(date_col)
        if cat_cols:
            f.write("## 5. カテゴリカラムの情報\n\n")
            for col in cat_cols:
                f.write(f"### {col}\n\n")
                f.write(f"- **ユニーク数**: {df[col].nunique()}\n")
                value_counts = df[col].value_counts().head(10)
                f.write("- **上位10件**:\n\n")
                f.write("| 値 | 件数 | 割合(%) |\n")
                f.write("|-----|------|--------|\n")
                for val, count in value_counts.items():
                    pct = count / len(df) * 100
                    f.write(f"| {val} | {count:,} | {pct:.2f} |\n")
                f.write("\n")

        # DataFrame Info
        f.write("## 6. DataFrame Info\n\n")
        f.write("```\n")
        buffer = io.StringIO()
        df.info(buf=buffer)
        f.write(buffer.getvalue())
        f.write("```\n")

    _log.info(f"Report written to: {output_path}")
