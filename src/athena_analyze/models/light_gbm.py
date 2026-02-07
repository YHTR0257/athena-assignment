from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from .base import Model, ModelRegistry

from utils.logging import setup_logging
_log = setup_logging()

@ModelRegistry.register("light_gbm")
class LightGBM(Model):
    """
    Optunaを用いたハイパーパラメータチューニング機能を備えたLightGBMモデルの実装クラス。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        LightGBMモデルの初期化。

        :param config: モデル設定辞書
        :type config: Dict[str, Any]
        """
        super().__init__(config)

        self.params: Dict[str, Any] = config.get("params", {})
        self.num_boost_round: int = config.get("num_boost_round", 100)
        self.target_col: str = config.get("target_col", "target")
        self.model: Optional[lgb.Booster] = None
        self.early_stopping_rounds: int = config.get("early_stopping_rounds", 10)
        self.log_evaluation: int = config.get("log_evaluation", 10)

        # 並列処理設定
        self.n_jobs: int = config.get("n_jobs", -1)
        self.num_threads: int = config.get("num_threads", -1)
        self.n_trials: int = config.get("n_trials", 100)

        self._validate_config()

        self.objective = self.params.get("objective", "regression")
        self.bagging_freq = self.params.get("bagging_freq", {})
        self.bagging_fraction = self.params.get("bagging_fraction", {})
        self.min_child_samples = self.params.get("min_child_samples", {})
        self.boosting_type = self.params.get("boosting", "gbdt")
        self.learning_rate = self.params.get("learning_rate", {})
        self.feature_fraction = self.params.get("feature_fraction", {})
        self.n_estimators = self.params.get("n_estimators", 100)
        self.num_leaves = self.params.get("num_leaves", {})

        # Optunaチューニング結果
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[optuna.study.Study] = None
    
    def _validate_config(self) -> None:
        """コンフィグのバリデーションを行う"""
        # 一般のコンフィグの必須キーと形を定義
        key_types = {
            "required": {
                "target_col": str,
            },
            "option": {
                "num_boost_round": int,
                "early_stopping_rounds": int,
                "log_evaluation": int,
            }
        }

        # paramsの必須キーと型を定義
        key_types = {
            "required": {
                "objective": str, "num_leaves": dict, "n_estimators": int, "metric": str
            },
            "option": {
                "min_data_in_leaf": int, "boosting": str, "learning_rate": dict,
                "max_depth": int, "feature_fraction": dict, "bagging_fraction": dict,
                "bagging_freq": dict, "lambda_l1": float, "lambda_l2": float,
                "num_boost_round": int, "random_state": int, "min_child_samples": dict,
                "verbosity": int
            }
        }

        for k, value in key_types["required"].items():
            if k not in self.params:
                raise ValueError(f"Missing required config key: {k}")
            if not isinstance(self.params[k], value):
                raise TypeError(f"Config key '{k}' must be of type {value}")
        
        for k, value in key_types["option"].items():
            if k in self.params and not isinstance(self.params[k], value):
                raise TypeError(f"Config key '{k}' must be of type {value}")
        
        _log.info("Config validation for LightGBM model passed.")
    
    def _objective(
        self,
        trial: optuna.trial.Trial,
        train_set: lgb.Dataset,
        valid_set: lgb.Dataset
    ) -> float:
        """
        Optunaの目的関数。RMSEを最小化する。

        :param trial: Optunaのトライアルオブジェクト
        :param train_set: 学習用LightGBM Dataset
        :param valid_set: 検証用LightGBM Dataset
        :return: 検証データに対するRMSE
        """
        params: Dict[str, Any] = {
            'objective': self.objective,
            'metric': self.params.get('metric', 'rmse'),
            'verbosity': self.params.get('verbosity', -1),
            'boosting_type': self.boosting_type,
            'num_threads': self.num_threads,
            'feature_pre_filter': False,
            'learning_rate': trial.suggest_float(
                'learning_rate',
                self.learning_rate['range'][0],
                self.learning_rate['range'][1],
                log=self.learning_rate.get('log', True)
            ),
            'num_leaves': trial.suggest_int(
                'num_leaves',
                self.num_leaves['range'][0],
                self.num_leaves['range'][1]
            ),
            'bagging_fraction': trial.suggest_float(
                'bagging_fraction',
                self.bagging_fraction['range'][0],
                self.bagging_fraction['range'][1]
            ),
            'min_child_samples': trial.suggest_int(
                'min_child_samples',
                self.min_child_samples['range'][0],
                self.min_child_samples['range'][1]
            ),
            'bagging_freq': trial.suggest_int(
                'bagging_freq',
                self.bagging_freq['range'][0],
                self.bagging_freq['range'][1]
            ),
            'feature_fraction': trial.suggest_float(
                'feature_fraction',
                self.feature_fraction['range'][0],
                self.feature_fraction['range'][1]
            ),
        }

        model = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        # LightGBMの返り値はnumpy配列なので、評価メトリクスに直接渡す
        # base.pyのメソッドが自動的にxp.asarray()で変換する
        preds = model.predict(valid_set.get_data())
        y_true = valid_set.get_label()
        rmse = self._calculate_rmse(y_true, preds)
        return float(rmse)

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
        split_idx = int(len(train_data) * 0.8)
        t_data, v_data = train_data.iloc[:split_idx], train_data.iloc[split_idx:]
        t_label, v_label = label.iloc[:split_idx], label.iloc[split_idx:]

        train_set = lgb.Dataset(t_data, label=t_label, free_raw_data=False)
        valid_set = lgb.Dataset(v_data, label=v_label, reference=train_set, free_raw_data=False)

        # Optunaでハイパーパラメータチューニング
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(
            lambda trial: self._objective(trial, train_set, valid_set),
            n_trials=self.n_trials,
            n_jobs=1,
            show_progress_bar=True
        )

        self.best_params = self.study.best_params
        _log.info(f"Optuna best params: {self.best_params}")
        _log.info(f"Optuna best RMSE: {self.study.best_value}")

        # 最適パラメータで最終学習
        final_params: Dict[str, Any] = {
            'objective': self.objective,
            'metric': self.params.get('metric', 'rmse'),
            'verbosity': self.params.get('verbosity', -1),
            'boosting_type': self.boosting_type,
            'num_threads': self.num_threads,
            **self.best_params
        }

        self.model = lgb.train(
            final_params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(self.log_evaluation)
            ]
        )

        _log.info("LightGBM model trained successfully.")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みモデルで予測を行う。

        :param X: 予測対象の特徴量データ
        :return: 予測値の配列
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        # LightGBMの予測結果はnumpy配列として返す
        return np.asarray(self.model.predict(X))
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        モデルの性能評価を行う。MAPE, RMSE, MAE, R2を計算。

        :param X: 評価用特徴量データ
        :param y: 評価用ターゲットデータ
        :return: 評価指標の辞書
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        y_pred = self.predict(X)
        y_true = np.asarray(y)

        # 評価メトリクスの呼び出し（base.pyのメソッドが自動的にxp.asarray()で変換）
        return {
            'mape': self._calculate_mape(y_true, y_pred),
            'rmse': self._calculate_rmse(y_true, y_pred),
            'mae': self._calculate_mae(y_true, y_pred),
            'r2': self._calculate_r2(y_true, y_pred),
        }

    def get_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得する。

        :return: モデル情報の辞書
        """
        info: Dict[str, Any] = {
            'model_type': 'LightGBM',
            'config': self.config,
            'is_trained': self.model is not None,
        }

        if self.model is not None:
            info['feature_importance'] = dict(
                zip(
                    self.model.feature_name(),
                    self.model.feature_importance(importance_type='gain').tolist()
                )
            )
            info['num_trees'] = self.model.num_trees()

        if self.best_params is not None:
            info['best_params'] = self.best_params

        if self.study is not None:
            info['optuna_best_value'] = self.study.best_value
            info['optuna_n_trials'] = len(self.study.trials)

        return info

    def save_model(self, file_path: str) -> None:
        """
        モデルをファイルに保存する。

        :param file_path: 保存先のファイルパス
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        self.model.save_model(file_path)
        _log.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """
        保存されたモデルをファイルから読み込む。

        :param file_path: 読み込み元のファイルパス
        """
        self.model = lgb.Booster(model_file=file_path)
        _log.info(f"Model loaded from {file_path}")
