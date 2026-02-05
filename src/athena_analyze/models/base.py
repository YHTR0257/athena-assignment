"""
油温予測モデルのためのモデル選択モジュール。
ここでは、複数の回帰モデルを定義し、選択できるようにします。パラメータの保存にも対応させます。
"""

from functools import wraps
from argon2 import Type
import numpy as np
from pydantic import BaseModel, Field
from typing import Callable, Union, Literal, Any, Dict, Type, List
from sklearn.metrics import r2_score, f1_score, mean_absolute_error, mean_squared_error
from enum import Enum
from abc import ABC, abstractmethod
import importlib
import pkgutil
from pathlib import Path

from utils.logging import setup_logging
_log = setup_logging()

class Model(ABC):
    """
    様々な機械学習モデルの基底クラス
    すべてのモデルはこのクラスを継承し、以下のメソッドを実装する必要があります。
    1. train: モデルの学習を行う
    2. predict: 学習済みモデルを用いて予測を行う
    3. evaluate: モデルの性能評価を行う
    4. get_info: モデルの情報を取得する
    5. save_model: モデルのパラメータを保存する
    6. load_model: 保存されたモデルのパラメータを読み込む
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: モデル固有の設定辞書
        """
        self.config = config
        self.model = None  # 学習済みモデルを保存するための属性

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass

    @staticmethod                                                                                                          
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:                                                  
        """MAPE (Mean Absolute Percentage Error) を計算"""                                                                 
        mask = y_true != 0                                                                                                 
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100                                         
                                                                                                                            
    @staticmethod                                                                                                          
    def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:                                                  
        """RMSE (Root Mean Squared Error) を計算"""                                                                        
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def _calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """MAE (Mean Absolute Error) を計算"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def _calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """R2 スコアを計算"""
        r2 = r2_score(y_true, y_pred)
        return np.float64(r2)
    
    @abstractmethod
    def _validate_config(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> Dict[str, np.float64]:
        pass

    @abstractmethod
    def get_info(self) -> dict:
        pass

    @abstractmethod
    def save_model(self, file_path: str) -> None:
        pass

    @abstractmethod
    def load_model(self, file_path: str) -> None:
        pass

class ModelRegistry:
    """
    モデルの登録と管理を行うクラス
    モデルクラスを登録し、名前で取得できるようにします。
    さらに、models/ ディレクトリを自動スキャンしてモデルを登録する機能も提供します。
    """
    _registry: Dict[str, Type[Model]] = {}

    @classmethod
    def register(cls, name: str, allow_overwrite: bool = False) -> Callable:
        def decorator(model_cls: Type[Model]) -> Type[Model]:
            if not issubclass(model_cls, Model):
                raise TypeError(f"{model_cls.__name__} is not a subclass of Model")
            if name in cls._registry and not allow_overwrite:
                raise ValueError(f"Model '{name}' is already registered.")
            _log.info(f"Registering model '{name}' with class {model_cls.__name__}")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(cls._registry.keys())
    
    @classmethod
    def get_model(cls, name: str) -> Type[Model]:
        if name not in cls._registry:
            available = ", ".join(cls.list_models())
            raise ValueError(f"Model '{name}' is not registered. Available models: {available}")
        return cls._registry[name]
    
    @classmethod
    def auto_discover(cls) -> List[str]:
        """base.py と同じディレクトリ内の models/ フォルダをスキャンし、モデルを自動登録する。"""
        models_dir = Path(__file__).parent
        _log.debug(f"Auto-discovering models in directory: {models_dir}")
        
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        for (_, module_name, _) in pkgutil.iter_modules([str(models_dir)]):
            try:
                importlib.import_module(f"athena_analyze.models.{module_name}")
            except Exception as e:
                print(f"⚠️  Failed to import {module_name}: {e}")
        
        return cls.list_models()
