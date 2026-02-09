import ctypes
import os
import site

import joblib

# nvidia pip packages内のlibcusparseLtをプリロード（PyTorch GPU版用）
_site_packages = site.getsitepackages()[0]
_cusparselt_so = os.path.join(_site_packages, "nvidia", "cusparselt", "lib", "libcusparseLt.so.0")
if os.path.isfile(_cusparselt_so):
    ctypes.cdll.LoadLibrary(_cusparselt_so)

import gc
from datetime import datetime
import pandas as pd
from athena_analyze.data.processor import DataProcessor
from athena_analyze.eda.visualize import plot_time_series
from utils.logging import setup_logging
from utils.config import load_config_section
from pathlib import Path
import click

_LOG_FILE = str(Path(__file__).parent.parent / "logs" / "experiment.log")
_log = setup_logging(log_file=_LOG_FILE)

class ExperimentRunner:
    def __init__(self, exp_name: str, root_dir: Path):
        self.exp_name = exp_name
        self.root_dir = root_dir
        self.general_cfg_path = self.root_dir / "config/config.yml"
        self.exp_cfg_path = self.root_dir / f"config/{exp_name}.yml"
        self.data_cfg = load_config_section(self.general_cfg_path, "data")
        raw_dir = self.data_cfg["raw"].replace("../", str(self.root_dir) + "/")
        self.processor = DataProcessor(data_folder=raw_dir)
        self.trained_models = []
        self.evaluation_results = {}
        self.model_infos = {}

    def load_and_preprocess(self, ds_label):
        pp_general_cfg = load_config_section(self.general_cfg_path, "preprocess")
        pp_exp_cfg = load_config_section(self.exp_cfg_path, "preprocess")

        train_name = f"train_{ds_label}"
        test_name = f"test_{ds_label}"
        cache_path = self.root_dir / f"data/experiment/{self.exp_name}/{train_name}.parquet"

        if cache_path.exists():
            _log.warning(f"Experiment data {train_name} already exists, loading from cache.")
            train_df = self.processor.load_data(f"{self.exp_name}/{train_name}.parquet",
                                                data_folder=self.root_dir / "data/experiment")
            test_df = self.processor.load_data(f"{self.exp_name}/{test_name}.parquet",
                                               data_folder=self.root_dir / "data/experiment")
        else:
            raw_df = self.processor.load_data(f"ETT{ds_label}.csv")
            _log.info(f"Processing dataset {ds_label}")
            pp_cfg = dict(**pp_general_cfg, **pp_exp_cfg[ds_label])
            train_df, test_df = self.processor.preprocess_data(raw_df, **pp_cfg)

            from utils.data_io import save_dataframe_to_parquet
            data_cfg = load_config_section(self.general_cfg_path, "data")
            exp_dir = data_cfg["experiment"].replace("../", str(self.root_dir) + "/")
            data_cfg_abs = {**data_cfg, "experiment": exp_dir}
            train_df = train_df[1000:]  # キャッシュ保存前に先頭1000行を削除
            save_dataframe_to_parquet(train_df, f"{self.exp_name}/{train_name}.parquet", config=data_cfg_abs)
            save_dataframe_to_parquet(test_df, f"{self.exp_name}/{test_name}.parquet", config=data_cfg_abs)

        return train_df, test_df

    def setup_models(self):
        from athena_analyze.models.base import ModelRegistry

        self.model_cfg = load_config_section(self.exp_cfg_path, "models")

        self.registry = ModelRegistry()
        self.registry.auto_discover()
        self.registry.list_models()
    
    def get_model_info(self):
        models_dir = self.root_dir / "models" / self.exp_name
        models_dir.mkdir(parents=True, exist_ok=True)

        for name, ds_label, model in self.trained_models:
            if name == "tft":
                # TFTは2モデル構成（resid + trend）
                combined_info = {"model_type": "TFT (resid + trend)"}
                for sub_name, sub_model in model.items():
                    save_path = str(models_dir / f"{name}_{sub_name}_{ds_label}.pt")
                    sub_model.save_model(save_path)
                    info = sub_model.get_info()
                    info['save_path'] = save_path
                    combined_info[sub_name] = info
                    _log.info(f"--- {name}_{sub_name} ({ds_label}) ---")
                    _log.info(f"Model Type: {info['model_type']}")
                    if info.get('best_params'):
                        _log.info(f"Best Params: {info['best_params']}")
                    if info.get('optuna_best_value'):
                        _log.info(f"Optuna Best RMSE: {info['optuna_best_value']:.4f}")
                    _log.info(f"Saved to: {save_path}")
                self.model_infos[(name, ds_label)] = combined_info
            else:
                if name == "sarima":
                    save_path = str(models_dir / f"{name}_{ds_label}.pkl")
                else:
                    save_path = str(models_dir / f"{name}_{ds_label}.txt")

                model.save_model(save_path)
                info = model.get_info()
                info['save_path'] = save_path

                self.model_infos[(name, ds_label)] = info

                _log.info(f"--- {name} ({ds_label}) ---")
                _log.info(f"Model Type: {info['model_type']}")
                if info.get('num_trees'):
                    _log.info(f"Num Trees: {info['num_trees']}")
                if info.get('order'):
                    _log.info(f"Order: {info['order']}")
                    _log.info(f"Seasonal Order: {info['seasonal_order']}")
                    _log.info(f"AIC: {info['aic']:.4f}")
                if info.get('best_params'):
                    _log.info(f"Best Params: {info['best_params']}")
                if info.get('optuna_best_value'):
                    _log.info(f"Optuna Best RMSE: {info['optuna_best_value']:.4f}")
                _log.info(f"Saved to: {save_path}")

    def run_evaluation(self, test_df):
        for name, ds_label, model in self.trained_models:
            if name == "sarima":
                eval_result = model.evaluate(test_df)
            elif name == "light_gbm":
                target_col = self.model_cfg[name].get("target_col", "OT")
                target_resid_col = f"stl_{target_col}_resid"
                stl_target_col = [c for c in test_df.columns if f"stl_{target_col}_" in c]
                drop_cols = [target_col, "date"] + stl_target_col
                feature_cols = [c for c in test_df.columns if c not in drop_cols]
                X_test = test_df[feature_cols].copy()
                X_test[target_resid_col] = model.predict(X_test)
                stl_cols = [c for c in test_df.columns if f"stl_{target_col}_" in c]
                # inverse
                for col in stl_cols:
                    if col == target_resid_col:
                        scaler = joblib.load(self.processor.data_folder / f'scaler_{col}.pkl')
                        X_test[col] = scaler.inverse_transform(X_test[[col]])
                    else:
                        scaler = joblib.load(self.processor.data_folder / f'scaler_{col}.pkl')
                        X_test[col] = scaler.inverse_transform(test_df[[col]])
                pred_df = X_test[stl_cols].copy()
                pred_df[target_col] = pred_df[stl_cols].sum(axis=1)
                self.pred_df = pred_df
                # 実測値もinverse transformしてから比較
                ot_scaler = joblib.load(self.processor.data_folder / f'scaler_{target_col}.pkl')
                actual_inv = ot_scaler.inverse_transform(test_df[[target_col]]).flatten()
                eval_result = model.evaluate(actual_inv, pred_df[target_col])
            elif name == "tft":
                target_col = self.model_cfg[name].get("target_col", "OT")
                target_resid_col = f"stl_{target_col}_resid"
                target_trend_col = f"stl_{target_col}_trend"
                stl_target_col = [c for c in test_df.columns if f"stl_{target_col}_" in c]
                drop_cols = [target_col, "date"] + stl_target_col
                exclude_patterns = ("_lag_", "_ma_")
                feature_cols = [
                    c for c in test_df.columns
                    if c not in drop_cols and not any(p in c for p in exclude_patterns)
                ]
                X_test = test_df[feature_cols].copy()

                # resid と trend を各モデルで予測
                X_test[target_resid_col] = model["resid"].predict(X_test[feature_cols])
                X_test[target_trend_col] = model["trend"].predict(X_test[feature_cols])

                stl_cols = [c for c in test_df.columns if f"stl_{target_col}_" in c]
                # inverse transform（seasonal は test_df の実測値を使用）
                for col in stl_cols:
                    scaler = joblib.load(self.processor.data_folder / f'scaler_{col}.pkl')
                    if col in (target_resid_col, target_trend_col):
                        X_test[col] = scaler.inverse_transform(X_test[[col]])
                    else:
                        X_test[col] = scaler.inverse_transform(test_df[[col]])
                pred_df = X_test[stl_cols].copy()
                pred_df[target_col] = pred_df[stl_cols].sum(axis=1)
                self.pred_df = pred_df
                # 実測値もinverse transformしてから比較
                ot_scaler = joblib.load(self.processor.data_folder / f'scaler_{target_col}.pkl')
                actual_inv = ot_scaler.inverse_transform(test_df[[target_col]]).flatten()
                eval_result = model["resid"].evaluate(actual_inv, pred_df[target_col])
            else:
                raise ValueError(f"Unsupported model type for evaluation: {name}")
            
            # 評価結果を保存
            self.evaluation_results[(name, ds_label)] = eval_result

            _log.info(f"--- {name} ({ds_label}) ---")
            for metric, value in eval_result.items():
                _log.info(f"  {metric}: {value:.4f}")

    def plot_results(self, test_df):
        reports_dir = self.root_dir / "reports" / self.exp_name
        reports_dir.mkdir(parents=True, exist_ok=True)

        for name, ds_label, model in self.trained_models:
            if name == "sarima":
                y_col = self.model_cfg[name].get("y_col", "OT")
                preds = model.predict(test_df)
                actual = test_df[y_col].values
            elif name in ("light_gbm", "tft"):
                target_col = self.model_cfg[name].get("target_col", "OT")
                preds = self.pred_df[target_col].values
                scaler = joblib.load(self.processor.data_folder / f'scaler_{target_col}.pkl')
                actual = scaler.inverse_transform(test_df[[target_col]]).flatten()
                # tftのevaluateでは使用しないため、pred_dfのOTのinverse対象はscalerからは不要
            else:
                raise ValueError(f"Unsupported model type for plotting: {name}")

            plot_df = pd.DataFrame({
                "date": pd.to_datetime(test_df["date"]).values,
                "actual": actual,
                "predicted": preds,
            })

            fig = plot_time_series(
                df=plot_df,
                date_col="date",
                value_cols=["actual", "predicted"],
                figsize=(14, 5),
            )
            fig.suptitle(f"{name} ({ds_label})", fontsize=14, fontweight="bold", y=1.02)

            save_path = reports_dir / f"{name}_{ds_label}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            _log.info(f"Plot saved to {save_path}")

            # 予測データを保存
            from utils.data_io import save_dataframe_to_parquet
            data_cfg = load_config_section(self.general_cfg_path, "data")
            exp_dir = data_cfg["experiment"].replace("../", str(self.root_dir) + "/")
            data_cfg_abs = {**data_cfg, "experiment": exp_dir}
            pred_filename = f"{self.exp_name}/predictions_{name}_{ds_label}.parquet"
            save_dataframe_to_parquet(plot_df, pred_filename, config=data_cfg_abs)

    def generate_report(self, ds_label: str):
        """Markdownレポートを生成する"""
        reports_dir = self.root_dir / "reports" / self.exp_name
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / f"report_{ds_label}.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# Experiment Report: {self.exp_name}",
            "",
            f"**Dataset**: {ds_label}",
            f"**Generated**: {timestamp}",
            "",
            "---",
            "",
            "## Model Results",
            "",
        ]

        # 各モデルの結果を追加
        for name, label, _ in self.trained_models:
            if label != ds_label:
                continue

            info = self.model_infos.get((name, label), {})
            eval_result = self.evaluation_results.get((name, label), {})

            lines.append(f"### {info.get('model_type', name)}")
            lines.append("")

            # モデル情報（収束結果）
            lines.append("#### Model Configuration")
            lines.append("")
            if info.get('order'):
                lines.append(f"- **Order (p, d, q)**: {info['order']}")
                lines.append(f"- **Seasonal Order (P, D, Q, m)**: {info['seasonal_order']}")
                lines.append(f"- **AIC**: {info['aic']:.4f}")
                if info.get('bic'):
                    lines.append(f"- **BIC**: {info['bic']:.4f}")
            if info.get('num_trees'):
                lines.append(f"- **Number of Trees**: {info['num_trees']}")
            if info.get('best_params'):
                lines.append("- **Best Parameters (Optuna)**:")
                for param, value in info['best_params'].items():
                    if isinstance(value, float):
                        lines.append(f"  - {param}: {value:.6f}")
                    else:
                        lines.append(f"  - {param}: {value}")
            if info.get('optuna_best_value'):
                lines.append(f"- **Optuna Best RMSE**: {info['optuna_best_value']:.4f}")
            if info.get('save_path'):
                lines.append(f"- **Model Path**: `{info['save_path']}`")
            lines.append("")

            # 評価指標
            lines.append("#### Evaluation Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for metric, value in eval_result.items():
                lines.append(f"| {metric} | {value:.4f} |")
            lines.append("")

            # プロット画像への参照
            plot_path = f"{name}_{ds_label}.png"
            lines.append("#### Prediction Plot")
            lines.append("")
            lines.append(f"![{name} predictions]({plot_path})")
            lines.append("")
            lines.append("---")
            lines.append("")

        # ファイルに書き込み
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        _log.info(f"Report saved to {report_path}")

    def load_trained_model(self, model_name: str, ds_label: str):
        """学習済みモデルを読み込む"""
        models_dir = self.root_dir / "models" / self.exp_name

        if model_name == "tft":
            # TFTは2モデル構成（resid + trend）
            resid_path = models_dir / f"{model_name}_resid_{ds_label}.pt"
            trend_path = models_dir / f"{model_name}_trend_{ds_label}.pt"
            if not resid_path.exists() or not trend_path.exists():
                return None
            model_cls = self.registry.get_model(model_name)
            model_resid = model_cls(config=self.model_cfg[model_name])
            model_resid.load_model(str(resid_path))
            model_trend = model_cls(config=self.model_cfg[model_name])
            model_trend.load_model(str(trend_path))
            _log.info(f"Loaded TFT models from {resid_path} and {trend_path}")
            return {"resid": model_resid, "trend": model_trend}

        if model_name == "sarima":
            model_path = models_dir / f"{model_name}_{ds_label}.pkl"
        else:
            model_path = models_dir / f"{model_name}_{ds_label}.txt"

        if not model_path.exists():
            return None

        model_cls = self.registry.get_model(model_name)
        model = model_cls(config=self.model_cfg[model_name])
        model.load_model(str(model_path))
        _log.info(f"Loaded trained model from {model_path}")
        return model

    def execute_training_or_load(self, train_df, ds_label):
        """学習済みモデルがあれば読み込み、なければ学習する"""
        each_model_cfg = [k for k in self.model_cfg.keys() if k in self.model_cfg["general"]["models"]]
        self.trained_models = []

        for model_name, each_cfg in zip(self.model_cfg["general"]["models"], each_model_cfg):
            # 学習済みモデルの読み込みを試みる
            model = self.load_trained_model(each_cfg, ds_label)

            if model is not None:
                _log.info(f"Using pre-trained model: {each_cfg} on {ds_label}")
                if each_cfg == "tft":
                    target_col = self.model_cfg[each_cfg].get("target_col", "OT")
                    stl_target_col = [c for c in train_df.columns if f"stl_{target_col}_" in c]
                    target_resid_col = f"stl_{target_col}_resid"
                    target_trend_col = f"stl_{target_col}_trend"
                    drop_cols = [target_col, "date"] + stl_target_col
                    exclude_patterns = ("_lag_", "_ma_")
                    feature_cols = [
                        c for c in train_df.columns
                        if c not in drop_cols and not any(p in c for p in exclude_patterns)
                    ]
                    model["resid"].setup_prediction_context(train_df, feature_cols, target_resid_col)
                    model["trend"].setup_prediction_context(train_df, feature_cols, target_trend_col)
            else:
                _log.info(f"Training model: {each_cfg} on {ds_label}")
                model_cls = self.registry.get_model(model_name)
                model = model_cls(config=self.model_cfg[each_cfg])

                if each_cfg == "sarima":
                    model.train(train_df)
                elif each_cfg == "light_gbm":
                    target_col = self.model_cfg[each_cfg].get("target_col", "OT")
                    stl_target_col = [c for c in train_df.columns if f"stl_{target_col}_" in c]
                    target_resid_col = f"stl_{target_col}_resid"
                    drop_cols = [target_col, "date"] + stl_target_col
                    feature_cols = [c for c in train_df.columns if c not in drop_cols]
                    _log.info(f"{each_cfg} training with features: {feature_cols} and target: {target_resid_col}")

                    X_train = train_df[feature_cols]
                    y_train = train_df[target_resid_col]

                    model.train(X_train, y_train)
                elif each_cfg == "tft":
                    target_col = self.model_cfg[each_cfg].get("target_col", "OT")
                    stl_target_col = [c for c in train_df.columns if f"stl_{target_col}_" in c]
                    target_resid_col = f"stl_{target_col}_resid"
                    target_trend_col = f"stl_{target_col}_trend"
                    drop_cols = [target_col, "date"] + stl_target_col
                    # TFTはLSTM+Attentionで時系列パターンを内部学習するため、
                    # ラグ・移動平均・他変数STLを除外し生データ+日付特徴に絞る
                    exclude_patterns = ("_lag_", "_ma_")
                    feature_cols = [
                        c for c in train_df.columns
                        if c not in drop_cols and not any(p in c for p in exclude_patterns)
                    ]
                    _log.info(f"TFT feature count: {len(feature_cols)} (filtered from {len(train_df.columns)} total)")
                    X_train = train_df[feature_cols]

                    # resid用モデル
                    _log.info(f"TFT training (resid) with features: {feature_cols} and target: {target_resid_col}")
                    model_resid = model_cls(config=self.model_cfg[each_cfg])
                    model_resid.train(X_train, train_df[target_resid_col])

                    # trend用モデル
                    _log.info(f"TFT training (trend) with features: {feature_cols} and target: {target_trend_col}")
                    model_trend = model_cls(config=self.model_cfg[each_cfg])
                    model_trend.train(X_train, train_df[target_trend_col])

                    model = {"resid": model_resid, "trend": model_trend}
                else:
                    raise ValueError(f"Unsupported model type for training: {each_cfg}")

            self.trained_models.append((each_cfg, ds_label, model))

        _log.info(f"Model preparation complete for {ds_label}.")

@click.command()
@click.option("--exp-name", default="exp_test", help="Experiment name")
def main(exp_name: str):
    cwd = Path(__file__).parent.parent
    runner = ExperimentRunner(exp_name, root_dir=cwd)
    runner.setup_models()

    dataset_labels = ["h1", "h2"]
    for ds_label in dataset_labels:
        _log.info(f"=== Processing dataset {ds_label} ===")
        train_df, test_df = runner.load_and_preprocess(ds_label)
        runner.execute_training_or_load(train_df, ds_label)
        runner.get_model_info()
        runner.run_evaluation(test_df)
        runner.plot_results(test_df)
        runner.generate_report(ds_label)
        runner.trained_models.clear()
        del train_df, test_df
        gc.collect()

if __name__ == "__main__":
    main()
