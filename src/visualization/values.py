import numpy as np
from pathlib import Path
from typing import List, Union, Tuple, Any

class SaveValues:
    """
    Handles the persistence of raw numerical results from the GAwLL algorithm.
    
    This class is responsible for saving importance vectors and interaction 
    matrices in CSV and TXT formats.
    """

    def __init__(self, output_dir: str = 'results'):
        """
        Initializes the persistence handler.

        Args:
            output_dir (str): Base directory where results will be stored.
        """
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def save_importances(self, importance_vector: np.ndarray, importance_name: str) -> None:
        """
        Saves the importance vector (VImp) as a CSV file.

        Args:
            importance_vector (np.ndarray): Array containing importance scores.
            importance_name (str): Name of the file (e.g., 'vimp_max').
        """
        filepath = self.output_path / f'{importance_name}.csv'

        # Using numpy for high precision (10 decimal places) and I/O speed
        np.savetxt(filepath, importance_vector, delimiter=',',
                   header='Importance', comments='', fmt='%.10f')

    def save_interaction_matrix(self, interaction_matrix: np.ndarray, name: str) -> None:
        """
        Saves the interaction matrix (vInt) as a CSV file with normalization.

        Args:
            interaction_matrix (np.ndarray): The raw adjacency matrix from Linkage Learning.
            name (str): Name of the file (e.g., 'matrix_max').
        """
        filepath = self.output_path / f'{name}.csv'

        matrix = np.asanyarray(interaction_matrix)

        np.savetxt(filepath, matrix, delimiter=',', fmt='%.10f')

    def save_top_importances(self, 
                             gawll_imp: np.ndarray, 
                             variables: List[str], 
                             model_name: str, 
                             ds_name: str, 
                             n: int) -> None:
        """
        Saves a human-readable TXT report of the Top N most important variables.

        Args:
            gawll_imp (np.ndarray): Importance scores.
            variables (List[str]): List of feature names.
            model_name (str): Name of the estimator used (e.g., 'DT', 'RF').
            ds_name (str): Name of the dataset.
            n (int): Number of top variables to export.
        """
        # Pair variables with their importance and sort descending
        variable_importances = list(zip(variables, gawll_imp))
        sorted_imps = sorted(variable_importances, key=lambda x: x[1], reverse=True)
        top_n = sorted_imps[:n]

        filename = f"{ds_name}_{model_name}_Top_{n}_Importances.txt"
        filepath = self.output_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Top {n} Attributes - GAwLL ({ds_name} | {model_name})\n")
            f.write("-" * 50 + "\n")
            for var, val in top_n:
                f.write(f"- {var}: {val:.10f}\n")

    def save_top_interactions(self, 
                              interaction_matrix: np.ndarray, 
                              variables: List[str], 
                              model_name: str, 
                              ds_name: str, 
                              n: int) -> None:
        """
        Saves a human-readable TXT report of the Top N variable interactions.

        Args:
            interaction_matrix (np.ndarray): The interaction matrix (vInt).
            variables (List[str]): List of feature names.
            model_name (str): Name of the estimator.
            ds_name (str): Name of the dataset.
            n (int): Number of top interactions to export.
        """
        weighted_edges: List[Tuple[Tuple[str, str], float]] = []
        num_vars = len(variables)

        # Efficient extraction of the upper triangle to avoid redundant (A,B) and (B,A)
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                weight = float(interaction_matrix[i][j])
                if abs(weight) > 1e-10:
                    weighted_edges.append(((variables[i], variables[j]), weight))

        # Sort by absolute interaction strength
        sorted_ints = sorted(weighted_edges, key=lambda x: abs(x[1]), reverse=True)
        top_n_ints = sorted_ints[:n]

        filename = f"{ds_name}_{model_name}_Top_{n}_Interactions.txt"
        filepath = self.output_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Top {n} Interactions - GAwLL ({ds_name} | {model_name})\n")
            f.write("-" * 50 + "\n")
            for (var1, var2), weight in top_n_ints:
                f.write(f"- {var1} <--> {var2}: {weight:.10f}\n")