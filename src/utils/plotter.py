import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.logging import setup_logging

_log = setup_logging()

class Plotter:
    """
    Plotter の Docstring

    Args:
        None
    """

    def __init__(self, output_fol: str):
        self.output_folder = Path(output_fol)
        os.makedirs(self.output_folder, exist_ok=True)
        _log.debug(f"Plotter initialized with output folder: {self.output_folder}")

    def save_plot(self, fig: Figure, file_name: str) -> None:
        """
        指定されたファイル名でプロットを保存します。

        Args:
            fig (Figure): 保存するプロットの Figure オブジェクト
            file_name (str): 保存するファイルの名前
        """
        file_path = self.output_folder / file_name
        _str_path = str(file_path)
        fig.savefig(_str_path)
        plt.close(fig)
        _log.debug(f"Plot saved to {_str_path}")
