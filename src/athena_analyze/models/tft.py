from typing import Dict, Any, Optional, List
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import torch
from pytorch_lightning.callbacks import EarlyStopping
from darts import TimeSeries
from darts.models import TFTModel as DartsTFTModel
from darts.dataprocessing.transformers import Scaler
from .base import Model, ModelRegistry

from utils.logging import setup_logging
_log = setup_logging()


@ModelRegistry.register("tft")
class TFTModel(Model):
    """
    dartsのTemporal Fusion Transformerをバックエンドとするモデル。
    Optunaによるハイパーパラメータチューニング機能を備える。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.target_col: str = config.get("target_col", "OT")
        self.input_chunk_length: int = config.get("input_chunk_length", 168)
        self.output_chunk_length: int = config.get("output_chunk_length", 24)
        self.max_epochs: int = config.get("max_epochs", 50)
        self.batch_size: int = config.get("batch_size", 64)
        self.n_trials: int = config.get("n_trials", 20)
        self.random_state: int = config.get("random_state", 42)

        self.params: Dict[str, Any] = config.get("params", {})
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[optuna.study.Study] = None
        self.model: Optional[DartsTFTModel] = None

        self._validate_config()

    def _validate_config(self) -> None:
        key_types = {
            "required": {
                "target_col": str,
                "input_chunk_length": int,
                "output_chunk_length": int,
            },
            "option": {
                "max_epochs": int,
                "batch_size": int,
                "n_trials": int,
                "random_state": int,
            }
        }

        for k, expected_type in key_types["required"].items():
            if k not in self.config:
                raise ValueError(f"Missing required config key: {k}")
            if not isinstance(self.config[k], expected_type):
                raise TypeError(f"Config key '{k}' must be of type {expected_type}")

        for k, expected_type in key_types["option"].items():
            if k in self.config and not isinstance(self.config[k], expected_type):
                raise TypeError(f"Config key '{k}' must be of type {expected_type}")

        _log.info("Config validation for TFT model passed.")

    def _df_to_timeseries(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
    ) -> tuple:
        """DataFrameをdarts TimeSeries (target, covariates) に変換する。"""
        df_f32 = df.reset_index(drop=True).astype(np.float32)

        target_ts = TimeSeries.from_dataframe(
            df_f32,
            value_cols=[target_col],
            fill_missing_dates=False,
        )

        if feature_cols:
            cov_ts = TimeSeries.from_dataframe(
                df_f32,
                value_cols=feature_cols,
                fill_missing_dates=False,
            )
        else:
            cov_ts = None

        return target_ts, cov_ts

    def _build_model(
        self,
        params: Dict[str, Any],
        trial: Optional[optuna.trial.Trial] = None,
    ) -> DartsTFTModel:
        """パラメータからdarts TFTModelインスタンスを構築する。"""
        callbacks = []
        if trial is not None:
            # EarlyStopping: val_lossが改善しなければ学習を早期停止
            callbacks.append(
                EarlyStopping(monitor="val_loss", patience=3, mode="min")
            )

        pl_trainer_kwargs = {
            "enable_progress_bar": True,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "callbacks": callbacks,
        }

        return DartsTFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            hidden_size=params.get("hidden_size", 64),
            lstm_layers=params.get("lstm_layers", 1),
            num_attention_heads=params.get("num_attention_heads", 4),
            dropout=params.get("dropout", 0.1),
            batch_size=self.batch_size,
            n_epochs=self.max_epochs,
            random_state=self.random_state,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
            log_tensorboard=False,
            optimizer_kwargs={"lr": params.get("learning_rate", 1e-3)},
            add_relative_index=True,
        )

    def _objective(
        self,
        trial: optuna.trial.Trial,
        train_target: TimeSeries,
        val_target: TimeSeries,
        full_cov: Optional[TimeSeries],
    ) -> float:
        """Optunaの目的関数。検証データに対するRMSEを最小化する。"""
        hidden_size_range = self.params.get("hidden_size", {}).get("range", [32, 128])
        lstm_layers_range = self.params.get("lstm_layers", {}).get("range", [1, 3])
        num_heads_range = self.params.get("num_attention_heads", {}).get("range", [1, 8])
        dropout_range = self.params.get("dropout", {}).get("range", [0.05, 0.3])
        lr_range = self.params.get("learning_rate", {}).get("range", [1e-4, 1e-2])

        params = {
            "hidden_size": trial.suggest_int("hidden_size", hidden_size_range[0], hidden_size_range[1]),
            "lstm_layers": trial.suggest_int("lstm_layers", lstm_layers_range[0], lstm_layers_range[1]),
            "num_attention_heads": trial.suggest_int("num_attention_heads", num_heads_range[0], num_heads_range[1]),
            "dropout": trial.suggest_float("dropout", dropout_range[0], dropout_range[1]),
            "learning_rate": trial.suggest_float(
                "learning_rate", lr_range[0], lr_range[1],
                log=self.params.get("learning_rate", {}).get("log", True),
            ),
        }

        model = self._build_model(params, trial=trial)

        model.fit(
            series=train_target,
            past_covariates=full_cov,
            val_series=val_target,
            val_past_covariates=full_cov,
        )

        pred = model.predict(
            n=len(val_target),
            series=train_target,
            past_covariates=full_cov,
        )

        pred_vals = pred.values(copy=False).flatten()
        actual_vals = val_target.values(copy=False).flatten()

        min_len = min(len(pred_vals), len(actual_vals))
        rmse = float(self._calculate_rmse(actual_vals[:min_len], pred_vals[:min_len]))
        return rmse

    def train(
        self,
        train_data: pd.DataFrame,
        label: pd.Series,
    ) -> None:
        """
        モデルの学習を行う。train_dataを8:2に分割し、Optunaでハイパーパラメータチューニングを実施。

        :param train_data: 学習用特徴量データ
        :param label: 学習用ターゲットデータ
        """
        df = train_data.copy()
        target_resid_col = label.name or self.target_col
        df[target_resid_col] = label.values
        feature_cols = [c for c in train_data.columns]

        # 全データから連続したTimeSeriesを作成し、インデックスでスライス
        full_target, full_cov = self._df_to_timeseries(df, target_resid_col, feature_cols)
        split_idx = int(len(df) * 0.8)
        train_target = full_target[:split_idx]
        val_target = full_target[split_idx:]

        # Optunaでハイパーパラメータチューニング（EarlyStoppingで各trialを早期打ち切り）
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(
            lambda trial: self._objective(trial, train_target, val_target, full_cov),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params
        _log.info(f"Optuna best params: {self.best_params}")
        _log.info(f"Optuna best RMSE: {self.study.best_value}")

        # 最適パラメータで全データを使い最終学習
        self.model = self._build_model(self.best_params)
        self.model.fit(
            series=full_target,
            past_covariates=full_cov,
        )

        self._train_target = full_target
        self._train_cov = full_cov
        self._feature_cols = feature_cols
        self._target_resid_col = target_resid_col

        _log.info("TFT model trained successfully.")

    def setup_prediction_context(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> None:
        """ロード済みモデルに対して予測に必要なコンテキストを設定する。"""
        self._feature_cols = feature_cols
        self._target_resid_col = target_col
        df = train_df[feature_cols + [target_col]].copy()
        self._train_target, self._train_cov = self._df_to_timeseries(
            df, target_col, feature_cols,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みモデルで予測を行う。

        :param X: 予測対象の特徴量データ
        :return: 予測値の配列
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        n = len(X)
        cov_df = X[self._feature_cols].reset_index(drop=True).astype(np.float32)

        # past_covariatesは学習時+予測時分を結合する必要がある
        train_cov_df = self._train_cov.to_dataframe()
        full_cov_df = pd.concat([train_cov_df, cov_df], ignore_index=True)
        full_cov_ts = TimeSeries.from_dataframe(
            full_cov_df,
            value_cols=self._feature_cols,
            fill_missing_dates=False,
        )

        pred = self.model.predict(
            n=n,
            series=self._train_target,
            past_covariates=full_cov_ts,
        )

        return pred.values(copy=False).flatten()[:n]

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        モデルの性能評価を行う。

        :param y_true: 真値
        :param y_pred: 予測値
        :return: 評価指標の辞書
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        return {
            "mape": self._calculate_mape(y_true, y_pred),
            "rmse": self._calculate_rmse(y_true, y_pred),
            "mae": self._calculate_mae(y_true, y_pred),
            "r2": self._calculate_r2(y_true, y_pred),
        }

    def get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "model_type": "TFT",
            "config": self.config,
            "is_trained": self.model is not None,
        }

        if self.model is not None:
            info["input_chunk_length"] = self.input_chunk_length
            info["output_chunk_length"] = self.output_chunk_length

        if self.best_params is not None:
            info["best_params"] = self.best_params

        if self.study is not None:
            info["optuna_best_value"] = self.study.best_value
            info["optuna_n_trials"] = len(self.study.trials)

        return info

    def save_model(self, file_path: str) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        self.model.save(file_path)
        meta_path = Path(file_path).with_suffix(".meta.pkl")
        joblib.dump({
            "feature_cols": self._feature_cols,
            "train_target": self._train_target,
            "train_cov": self._train_cov,
            "target_resid_col": self._target_resid_col,
        }, meta_path)
        _log.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        self.model = DartsTFTModel.load(file_path)
        meta_path = Path(file_path).with_suffix(".meta.pkl")
        if meta_path.exists():
            meta = joblib.load(meta_path)
            self._feature_cols = meta["feature_cols"]
            self._train_target = meta["train_target"]
            self._train_cov = meta["train_cov"]
            self._target_resid_col = meta["target_resid_col"]
        _log.info(f"Model loaded from {file_path}")
