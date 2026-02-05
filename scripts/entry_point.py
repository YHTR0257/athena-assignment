import pandas as pd
from athena_analyze.data.processor import DataProcessor
from athena_analyze.eda.visualize import plot_time_series
from utils.logging import setup_logging
from utils.config import load_config_section
from pathlib import Path

_log = setup_logging()

class ExperimentRunner:
    def __init__(self, exp_name: str, root_dir: Path):
        self.exp_name = exp_name
        self.root_dir = root_dir
        self.general_cfg_path = self.root_dir / "config/config.yml"
        self.exp_cfg_path = self.root_dir / f"config/{exp_name}.yml"
        self.data_cfg = load_config_section(self.general_cfg_path, "data")
        raw_dir = self.data_cfg["raw"].replace("../", str(self.root_dir) + "/")
        self.processor = DataProcessor(data_fol=raw_dir)
        self.trained_models = []
        self.train_dfs = []
        self.test_dfs = []

    def setup(self):
        dfs = []
        dfs.append(self.processor.load_data("ETTh1.csv"))
        dfs.append(self.processor.load_data("ETTh2.csv"))
        return dfs
    
    def load_and_preprocess(self, dfs):
        pp_general_cfg = load_config_section(self.general_cfg_path, "preprocess")
        pp_exp_cfg = load_config_section(self.exp_cfg_path, "preprocess")
        self.train_dfs = []
        self.test_dfs = []
        if (self.root_dir / f"data/experiment/{self.exp_name}/train_h1.parquet").exists() == True:
            _log.warning(f"Experiment folder /data/experiment/{self.exp_name} already exists.")
            for name in ["train_h1", "train_h2", "test_h1", "test_h2"]:
                _log.info(f"Loading dataset {name}")
                df = self.processor.load_data(f"{self.exp_name}/{name}.parquet",
                                        data_folder=self.root_dir / "data/experiment")
                if "train" in name:
                    self.train_dfs.append(df)
                else:
                    self.test_dfs.append(df)
        else:
            for i, df in enumerate(dfs):
                _log.info(f"Processing dataset {i}")
                pp_cfg = dict(**pp_general_cfg, **pp_exp_cfg[f"h{i+1}"])
                train_df, test_df = self.processor.preprocess_data(df, **pp_cfg)
                self.train_dfs.append(train_df)
                self.test_dfs.append(test_df)

            from utils.data_io import save_dataframe_to_parquet
            data_cfg = load_config_section(self.general_cfg_path, "data")

            for df, name in zip(self.train_dfs + self.test_dfs, ["train_h1", "train_h2", "test_h1", "test_h2"]):
                save_dataframe_to_parquet(df, f"{self.exp_name}/{name}.parquet", config=data_cfg)
        return self.train_dfs, self.test_dfs

    def execute_training(self, train_dfs, test_dfs):
        from athena_analyze.models.base import ModelRegistry

        self.model_cfg = load_config_section(self.exp_cfg_path, "models")

        registry = ModelRegistry()
        registry.auto_discover()
        registry.list_models()

        each_model_cfg = [k for k in self.model_cfg.keys() if k in self.model_cfg["general"]["models"]]
        self.trained_models = []
        dataset_labels = ["h1", "h2"]

        for model_name, each_cfg in zip(self.model_cfg["general"]["models"], each_model_cfg):
            for di, ds_label in enumerate(dataset_labels):
                _log.info(f"Training model: {each_cfg} on {ds_label}")
                model_cls = registry.get_model(model_name)
                model = model_cls(config=self.model_cfg[each_cfg])

                if each_cfg == "sarima":
                    model.train(train_dfs[di])
                else:
                    target_col = self.model_cfg[each_cfg].get("target_col", "OT")
                    drop_cols = [target_col, "date"]
                    feature_cols = [c for c in train_dfs[di].columns if c not in drop_cols]

                    X_train = train_dfs[di][feature_cols]
                    y_train = train_dfs[di][target_col]
                    X_valid = test_dfs[di][feature_cols]
                    y_valid = test_dfs[di][target_col]

                    model.train(X_train, y_train, valid_data=X_valid, valid_label=y_valid)

                self.trained_models.append((each_cfg, ds_label, di, model))

        _log.info("Training complete.")
    
    def get_model_info(self):
        models_dir = self.root_dir / "models" / self.exp_name
        models_dir.mkdir(parents=True, exist_ok=True)

        for name, ds_label, _, model in self.trained_models:
            if name == "sarima":
                save_path = str(models_dir / f"{name}_{ds_label}.pkl")
            else:
                save_path = str(models_dir / f"{name}_{ds_label}.txt")

            model.save_model(save_path)
            info = model.get_info()
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

    def run_evaluation(self):
        for name, ds_label, di, model in self.trained_models:
            if name == "sarima":
                x_col = self.model_cfg[name].get("x_col", "date")
                y_col = self.model_cfg[name].get("y_col", "OT")
                X_test = self.test_dfs[di][[x_col]]
                y_test = self.test_dfs[di][y_col].values
                eval_result = model.evaluate(X_test, y_test)
            else:
                target_col = self.model_cfg[name].get("target_col", "OT")
                drop_cols = [target_col, "date"]
                feature_cols = [c for c in self.test_dfs[di].columns if c not in drop_cols]
                X_test = self.test_dfs[di][feature_cols]
                y_test = self.test_dfs[di][target_col]
                eval_result = model.evaluate(X_test, y_test)

            _log.info(f"--- {name} ({ds_label}) ---")
            for metric, value in eval_result.items():
                _log.info(f"  {metric}: {value:.4f}")

    def plot_results(self):
        reports_dir = self.root_dir / "reports" / self.exp_name
        reports_dir.mkdir(parents=True, exist_ok=True)

        for name, ds_label, di, model in self.trained_models:
            test_df = self.test_dfs[di]

            if name == "sarima":
                x_col = self.model_cfg[name].get("x_col", "date")
                y_col = self.model_cfg[name].get("y_col", "OT")
                preds = model.predict(len(test_df), test_df[[x_col]])
                actual = test_df[y_col].values
            else:
                target_col = self.model_cfg[name].get("target_col", "OT")
                drop_cols = [target_col, "date"]
                feature_cols = [c for c in test_df.columns if c not in drop_cols]
                preds = model.predict(test_df[feature_cols])
                actual = test_df[target_col].values

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

def main():
    exp_name = "exp_001"
    cwd = Path(__file__).parent.parent
    runner = ExperimentRunner(exp_name, root_dir=cwd)
    dfs = runner.setup()
    train_dfs, test_dfs = runner.load_and_preprocess(dfs)
    runner.execute_training(train_dfs, test_dfs)
    runner.get_model_info()
    runner.run_evaluation()
    runner.plot_results()

if __name__ == "__main__":
    
    main()
