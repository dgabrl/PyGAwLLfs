"""
Variable Importance (VImp) Module.

This module tracks the individual contribution of each feature to the model's 
performance. It maintains the empirical performance impact
observed during the evolutionary process.
"""

import numpy as np


class Importance:
    """
    Tracks the empirical importance (VImp) of each variable.

    By storing the historical performance impact for each gene,
    this class provides data for posterior XAI analysis.

    Attributes:
        degree (int): Number of features in the chromosome.
        max_importance (np.ndarray): Vector storing the highest observed impact per gene.
    """

    def __init__(self, degree: int):
        """
        Initializes the importance tracking vectors using sentinel values.

        Args:
            degree (int): The number of features (chromosome size).
        """
        self.degree: int = degree

        # Initialized importance vector with zeros
        self._max_importance: np.ndarray = np.zeros(degree, dtype=float)

    def add_importance(self, i: int, importance: float) -> None:
        """
        Updates the maximum importance for variable 'i'.

        Args:
            i: Index of the variable being updated.
            importance: The measured performance impact.
        """
        if importance > self._max_importance[i]:
            self._max_importance[i] = importance

    def export_importance_vector(self) -> np.ndarray:
        """
        Returns the VImp vector.

        Returns:
            np.ndarray: Max_Importance
        """
        return self._max_importance.copy()