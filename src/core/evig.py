"""
Empirical Variable Interaction Graph (eVIG) Module.

This module implements the storage and management of the Interaction Matrix (VInt).
It captures empirical evidence of relationships between features by recording
performance deltas observed during the Linkage Learning mutation process.
"""

import numpy as np


class eVIG:
    """
    Manages the Interaction Matrix (VInt) for feature linkage discovery.

    The eVIG maintains symmetric adjacency matrices that store the
    interaction weights observed between pairs of variables.

    Attributes:
        degree (int): The number of features (dimension of the square matrices).
    """

    def __init__(self, degree: int):
        """
        Initializes the interaction matrices with zeros.

        Args:
            degree (int): Total number of variables/features in the dataset.
        """
        self.degree: int = degree

        # _max_edge_weight: Stores the highest interaction.
        self._max_edge_weight: np.ndarray = np.zeros((degree, degree), dtype=float)

    def add_edge(self, a: int, b: int, w: float) -> None:
        """
        Records an interaction weight between feature 'a' and feature 'b'.

        This method updates the symmetric matrices if the new weight 'w'
        represents a new maximum for the pair.

        Args:
            a (int): Index of the first feature.
            b (int): Index of the second feature.
            w (float): The interaction weight.
        """
        # Symmetric update for the Maximum Interaction Matrix
        if w > self._max_edge_weight[a,b]:
            self._max_edge_weight[a, b] = w
            self._max_edge_weight[b, a] = w

    def export_interaction_matrix(self) -> np.ndarray:
        """
        Returns a copy of the internal Maximum Variable Interaction Matrix (vInt Max).

        This matrix represents the peak interaction strengths detected by
        the Linkage Learning mechanism during the evolution.

        Returns:
            np.ndarray: A 2D symmetric matrix of shape (n_features, n_features).
        """
        return self._max_edge_weight.copy()