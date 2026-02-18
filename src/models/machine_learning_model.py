"""
Machine Learning Model Wrappers.

This module provides an abstraction layer over Scikit-Learn models,
standardizing training, prediction, and performance evaluation
for both Classification and Regression tasks.
"""

import gc
import warnings
import numpy as np
from typing import Optional, Tuple, Any, Union
from src.utils.util import DatasetType

# Scikit-Learn Imports
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings to keep logs clean during GA evolution
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BaseModel:
    """
    Base class for machine learning models.

    Provides common utilities for data slicing and performance evaluation
    tailored for feature selection tasks.
    """

    def __init__(self, dataset_type: DatasetType):
        """
        Initialize the base model.

        Args:
            dataset_type (DatasetType): Enum for CLASSIFICATION or REGRESSION.
        """
        self.dataset_type: DatasetType = dataset_type
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.model: Any = None

    def set_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Stores the training dataset in the model's context.
        This data will be sliced and used during each GA evaluation.
        """
        self.X_train = X
        self.y_train = y

    def _prepare_data(self, X_test: np.ndarray, feature_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slices the training and test matrices based on selected features.

        Args:
            X_test: The testing feature matrix.
            feature_mask: Boolean mask of selected features.

        Returns:
            tuple: (sliced_X_train, sliced_X_test)
        """
        if feature_mask is not None:
            return self.X_train[:, feature_mask], X_test[:, feature_mask]
        return self.X_train, X_test

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, feature_mask: Optional[np.ndarray] = None) -> float:
        """
        Evaluates model performance on a specific subset of features.

        Args:
            X_test: Test features.
            y_test: Test targets.
            feature_mask: Boolean mask of features to use.

        Returns:
            float: Performance (Accuracy or 1-MSE).
        """
        # Edge case: no features selected
        if feature_mask is not None and not np.any(feature_mask):
            return 0.0

        y_pred = self.predict(X_test, feature_mask)

        if self.dataset_type == DatasetType.CLASSIFICATION:
            performance = float(accuracy_score(y_test, y_pred))
        else:
            performance = float(1.0 - mean_squared_error(y_test, y_pred))

        # Critical Memory Management: Explicitly clear model and trigger GC
        self.model = None
        gc.collect()

        return performance

    def predict(self, X: np.ndarray, feature_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the predict method.")
    
    def permutation_importances(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """
        Calculates feature importance using the Permutation Method.

        Args:
            X_test (np.ndarray): Test features matrix.
            y_test (np.ndarray): Test labels or target values.

        Returns:
            np.ndarray: Array containing the mean importance score for each feature.
        """
        # Ensure the model is fitted on the full feature set before permutation
        # Using feature_mask=None as per updated project guidelines (2026-02-04)
        self.predict(X_test, feature_mask=None)

        # Select appropriate scoring metric based on dataset type
        scoring_metric = (
            'accuracy' if self.dataset_type == DatasetType.CLASSIFICATION
            else 'neg_mean_squared_error'
        )

        # Perform permutation importance
        result = permutation_importance(
            self.model, 
            X_test, 
            y_test, 
            scoring=scoring_metric,
            random_state=42  # Seed for reproducibility across runs
        )

        # We return the mean importance. 
        return np.asanyarray(result.importances_mean)


class DT(BaseModel):
    """Decision Tree Wrapper."""

    def predict(self, X: np.ndarray, feature_mask: Optional[np.ndarray] = None) -> np.ndarray:
        X_tr, X_ts = self._prepare_data(X, feature_mask)

        self.model = (DecisionTreeClassifier() if self.dataset_type == DatasetType.CLASSIFICATION
                      else DecisionTreeRegressor())

        self.model.fit(X_tr, self.y_train)
        return self.model.predict(X_ts)
      
    def intrinsic_importances(self, X_test: np.ndarray) -> np.ndarray:
        self.predict(X_test, feature_mask=None)

        return np.asanyarray(self.model.feature_importances_)


class KNN(BaseModel):
    """K-Nearest Neighbors Wrapper."""

    def __init__(self, dataset_type: DatasetType, k: int, n_jobs: int):
        super().__init__(dataset_type)
        self.k = k
        self.n_jobs = n_jobs

    def predict(self, X: np.ndarray, feature_mask: Optional[np.ndarray] = None) -> np.ndarray:
        X_tr, X_ts = self._prepare_data(X, feature_mask)

        KClass = KNeighborsClassifier if self.dataset_type == DatasetType.CLASSIFICATION else KNeighborsRegressor
        self.model = KClass(n_neighbors=self.k, n_jobs=self.n_jobs)

        self.model.fit(X_tr, self.y_train)
        return self.model.predict(X_ts)


class MLP(BaseModel):
    """Neural Network (MLP) Wrapper."""

    def __init__(self, dataset_type: DatasetType, hidden_layer_sizes: Tuple):
        super().__init__(dataset_type)
        self.hidden_layer_sizes = hidden_layer_sizes

    def predict(self, X: np.ndarray, feature_mask: Optional[np.ndarray] = None) -> np.ndarray:
        X_tr, X_ts = self._prepare_data(X, feature_mask)

        params = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'max_iter': 500,
            'early_stopping': True,
            'random_state': 42  # Ensures deterministic results for GA stability
        }

        MLPClass = MLPClassifier if self.dataset_type == DatasetType.CLASSIFICATION else MLPRegressor
        self.model = MLPClass(**params)

        self.model.fit(X_tr, self.y_train)
        return self.model.predict(X_ts)


class RandomForest(BaseModel):
    """Random Forest Wrapper."""

    def __init__(self, dataset_type: DatasetType, n_jobs: int):
        super().__init__(dataset_type)
        self.n_jobs = n_jobs

    def predict(self, X: np.ndarray, feature_mask: Optional[np.ndarray] = None) -> np.ndarray:
        X_tr, X_ts = self._prepare_data(X, feature_mask)

        RFClass = RandomForestClassifier if self.dataset_type == DatasetType.CLASSIFICATION else RandomForestRegressor
        self.model = RFClass(n_jobs=self.n_jobs, max_depth=10, n_estimators=100)

        self.model.fit(X_tr, self.y_train)
        return self.model.predict(X_ts)
    
    def intrinsic_importances(self, X_test: np.ndarray) -> np.ndarray:
        self.predict(X_test, feature_mask=None)

        return np.asanyarray(self.model.feature_importances_)