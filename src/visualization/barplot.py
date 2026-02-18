import matplotlib
matplotlib.use('Agg')  # Essential for server-side/headless execution
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler
from typing import List


class BarPlot:
    """
    Handles the generation of bar charts for Variable Importance (VImp).
    """

    def __init__(self, output_dir: str = 'results'):
        """
        Initializes the BarPlot generator.

        Args:
            output_dir (str): Directory where plots will be saved.
        """
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.scaler = MaxAbsScaler()

    def _save_plot(self, 
                   variables: List[str], 
                   values: np.ndarray, 
                   title: str, 
                   filename: str, 
                   color: str, 
                   invert: bool = False) -> None:
        """
        Internal helper to create consistent single-sided bar plots.
        """
        if values is None or len(values) == 0:
            return

        # Dynamic figure width based on feature count to avoid overlap
        width = max(14, len(variables) * 0.3)
        plt.figure(figsize=(width, 7))

        display_values = -np.abs(values) if invert else np.abs(values)
        plt.bar(variables, display_values, color=color, alpha=0.7, edgecolor='black')

        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(title, fontsize=14)
        plt.ylabel('Importance Value')

        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()

        plt.savefig(self.output_path / f"{filename}.png", dpi=200)
        plt.close('all')

    def generate_importance_plots(self, 
                                  variables: List[str], 
                                  model_name: str, 
                                  dataset: str, 
                                  imp_vector: np.ndarray) -> None:
        """
        Generates raw and normalized bar plots for a given experiment run.

        Args:
            variables (List[str]): Feature names.
            model_name (str): ML model identifier (e.g., 'RandomForest').
            dataset (str): Dataset name.
            imp_vector (np.ndarray): Importance vector.
        """
        # 1. Data Preparation and Reshaping for Scikit-Learn
        imp_arr = np.asanyarray(imp_vector).reshape(-1, 1)

        # 2. Normalization (MaxAbsScaler preserves sign and sparsity)
        def scale_data(data: np.ndarray) -> np.ndarray:
            if np.ptp(data) > 0:
                return self.scaler.fit_transform(data).flatten()
            return data.flatten()

        norm_vimp = scale_data(imp_arr)
        raw_vimp = imp_arr.flatten()

        # 3. Raw and Normalized Plots
        self._save_plot(variables, raw_vimp, f'Raw GAwLL Importance - {model_name}',
                        f'{dataset}_{model_name}_Raw_GAwLL', '#2E8B57')
        self._save_plot(variables, norm_vimp, f'Normalized GAwLL Importance - {model_name}',
                        f'{dataset}_{model_name}_Normalized_GAwLL', '#2E8B57')