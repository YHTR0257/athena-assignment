from typing import Dict, Any
import copy
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
import pmdarima as pm
from utils.array_backend import xp
from .base import Model, ModelRegistry

from utils.logging import setup_logging
_log = setup_logging()

@ModelRegistry.register("sarima")
class SarimaModel(Model):
    """
    Sarimaモデルを実装したクラス。pmdarimaを用いることでSARIMAモデルのパラメータである
    (p, d, q)(P, D, Q, S) の自動設定と学習、予測を行う。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        __init__ の Docstring
        
        :param self: 説明
        :param config: 説明
        :type config: Dict[str, Any]
        """
        super().__init__(config)  # 親クラスの初期化を呼び出す

        self._validate_config()

        self.seasonal=config.get("seasonal", True)
        self.y_col = config.get("y_col", None)
        self.x_col = config.get("x_col", None)
        self.m = config.get("m", 1)  # 季節性の周期
        self.trace = config.get("trace", True)
        self.error_action = config.get("error_action", "ignore")
        self.suppress_warnings = config.get("suppress_warnings", True)

        self.max_p = config.get("max_p", 3)
        self.max_q = config.get("max_q", 3)
        self.max_P = config.get("max_P", 2)
        self.max_Q = config.get("max_Q", 2)
        
        # 学習済みモデル
        self.model = None
        self.fitted_model = None
    
    def _validate_config(self) -> None:
        """コンフィグのバリデーションを行う"""
        required_keys = ["seasonal", "y_col", "x_col", "m", "trace",
                         "error_action", "suppress_warnings", "max_p",
                         "max_q", "max_P", "max_Q"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # type checks
        if not isinstance(self.config["seasonal"], bool):
            raise TypeError("Config 'seasonal' must be a boolean")
        if not isinstance(self.config["y_col"], str):
            raise TypeError("Config 'y_col' must be a string")
        if not isinstance(self.config["x_col"], str):
            raise TypeError("Config 'x_col' must be a string")
        if not isinstance(self.config["m"], int) or self.config["m"] < 1:
            raise ValueError("Config 'm' must be a positive integer")
        if not isinstance(self.config["max_p"], int) or self.config["max_p"] < 0:
            raise ValueError("Config 'max_p' must be a non-negative integer")
        if not isinstance(self.config["max_q"], int) or self.config["max_q"] < 0:
            raise ValueError("Config 'max_q' must be a non-negative integer")
        if not isinstance(self.config["max_P"], int) or self.config["max_P"] < 0:
            raise ValueError("Config 'max_P' must be a non-negative integer")
        if not isinstance(self.config["max_Q"], int) or self.config["max_Q"] < 0:
            raise ValueError("Config 'max_Q' must be a non-negative integer")
    
    def train(self, train_data: pd.DataFrame) -> None:
        """
        モデルの学習を行う。
        
        :param self: 説明
        :param train_data: 時系列データで、インデックスがDatetimeであることを想定
        :type train_data: pd.DataFrame
        """
        _log.info(f"Training SARIMA model with pmd_arima...")

        _log.info(f"Train and Eval splited")
        y = train_data[self.y_col]
        x = train_data[self.x_col]

        self.model = pm.auto_arima(
            y=y,
            exogenous=x,
            seasonal=self.seasonal,
            m=self.m,
            trace=self.trace,
            error_action=self.error_action,
            suppress_warnings=self.suppress_warnings,
            stepwise=True,
            max_p=self.max_p,
            max_q=self.max_q,
            max_P=self.max_P,
            max_Q=self.max_Q
        )

        _log.info(f"Done defined params of SARIMA model")

        try:
            self.model.fit(y, exogenous=x)
            self.fitted_model = self.model
        except Exception as e:
            _log.error(f"Error during SARIMA model fitting: {e}")
            raise e
        _log.info(f"SARIMA model training completed.")
        

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        ローリング予測（walk-forward）を行う。
        1ステップずつ予測し、実績値でモデルを更新してから次を予測する。

        注意: ARIMAのローリング予測は順次処理のため、GPU加速できません。
        各予測が前のステップに依存し、バッチ化が不可能なためです。

        :param test_data: テストデータ（y_colとx_colを含むDataFrame）
        :type test_data: pd.DataFrame
        :return: 予測結果の配列
        :rtype: np.ndarray
        """
        if self.fitted_model is None:
            _log.error("Model is not trained yet. Call train() before predict().")
            raise ValueError("Model must be trained before prediction.")

        y_actual = test_data[self.y_col].values
        x_exog = test_data[[self.x_col]].values
        n_samples = len(test_data)

        # 最適化: 事前にarray確保（リストappendより高速）
        forecasts = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            # pmdarimaのpredict()はnumpy配列を返す
            # 最適化: 不要な型変換を削除し、直接アクセス
            pred = self.fitted_model.predict(n_periods=1, exogenous=x_exog[i:i+1])
            forecasts[i] = pred[0]
            self.fitted_model.update(y_actual[i], exogenous=x_exog[i:i+1])

        return forecasts

    def get_info(self) -> dict:
        """
        モデルの情報を取得する。

        :return: モデル情報の辞書
        :rtype: dict
        """
        if self.fitted_model is None:
            return {"model_type": "SARIMA", "status": "untrained"}

        return {
            "model_type": "SARIMA",
            "status": "trained",
            "order": self.fitted_model.order,
            "seasonal_order": self.fitted_model.seasonal_order,
            "aic": self.fitted_model.aic(),
            "bic": self.fitted_model.bic(),
        }

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        ローリング予測を用いてモデルの性能評価を行う。
        モデルのコピーを使用し、元のモデル状態は保持する。

        :param test_data: テストデータ（y_colとx_colを含むDataFrame）
        :type test_data: pd.DataFrame
        :return: 評価指標の辞書
        :rtype: Dict[str, float]
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained before evaluation.")

        y = test_data[self.y_col].values
        if len(y) == 0:
            raise ValueError("Evaluation data is empty.")

        # モデルのコピーを使用して元の状態を保持
        original_model = self.fitted_model
        self.fitted_model = copy.deepcopy(original_model)

        predictions = self.predict(test_data)

        # 元のモデルを復元
        self.fitted_model = original_model

        # 評価指標の計算（base.pyのメソッドがnumpy/cupy配列を自動変換）
        # ゼロ除算を避けるため、yがゼロの場合は除外
        mask = np.asarray(y != 0)
        if not np.any(mask):
            raise ValueError("All target values are zero, cannot compute MAPE.")
        mape = self._calculate_mape(y[mask], predictions[mask])
        rmse = self._calculate_rmse(y, predictions)
        r2 = self._calculate_r2(y, predictions)
        return {"MAPE": mape, "RMSE": rmse, "R2": r2}

    def save_model(self, file_path: str) -> None:
        """
        モデルをファイルに保存する。

        :param file_path: 保存先のファイルパス
        :type file_path: str
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained before saving.")

        path = Path(file_path)

        if path.suffix.lower() != ".pkl":
            raise ValueError("Model file must have a .pkl extension.")

        with open(path, 'wb') as f:
            pickle.dump(self.fitted_model, f)
        _log.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """
        保存されたモデルを読み込む。

        :param file_path: モデルファイルのパス
        :type file_path: str
        """
        with open(file_path, 'rb') as f:
            self.fitted_model = pickle.load(f)
            self.model = self.fitted_model
        _log.info(f"Model loaded from {file_path}")
