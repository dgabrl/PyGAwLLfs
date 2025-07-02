from src.util.util import DatasetType
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
import warnings

class DT:
    def __init__(
        self,
        dataset_type,
        min_samples_split,
        random_state=42,
    ):
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.dataset_type = dataset_type

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, dimensions=None):
        if self.dataset_type == DatasetType.CLASSIFICATION:
            self.model = DecisionTreeClassifier(
                random_state=self.random_state, min_samples_split=self.min_samples_split
            )
        elif self.dataset_type == DatasetType.REGRESSION:
            self.model = DecisionTreeRegressor(
                random_state=self.random_state, min_samples_split=self.min_samples_split
            )

        if dimensions is not None:
            X_train = self.X_train[:, dimensions]
            X = X[:, dimensions]
        else:
            X_train = self.X_train

        self.model.fit(X_train, self.y_train)
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, dimensions=None):
        if not np.any(dimensions):
            return 0.0

        y_pred = self.predict(X_test, dimensions)

        if self.dataset_type == DatasetType.CLASSIFICATION:
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
        elif self.dataset_type == DatasetType.REGRESSION:
            mse = 1 - mean_squared_error(y_test, y_pred)
            return mse

    def permutation_importances(self, X_test, y_test):
      self.predict(self.X_train)
      self.model.fit(self.X_train, self.y_train)
      if self.dataset_type == DatasetType.CLASSIFICATION:
          scoring_metric = 'accuracy'
      elif self.dataset_type == DatasetType.REGRESSION:
          scoring_metric = 'neg_mean_squared_error'
      else:
          raise ValueError("Dataset type not allowed")
      importances = permutation_importance(self.model, X_test, y_test,
                                           scoring=scoring_metric)
      return np.array(importances.importances_mean)

    def intrinsic_importances(self):
      self.predict(self.X_train)
      self.model.fit(self.X_train, self.y_train)
      return np.array(self.model.feature_importances_)

class KNN:
    def __init__(self, dataset_type, k):
        self.k = k
        self.dataset_type = dataset_type

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, dimensions=None):
        if self.dataset_type == DatasetType.CLASSIFICATION:
            self.model = KNeighborsClassifier(n_neighbors=self.k)
        elif self.dataset_type == DatasetType.REGRESSION:
            self.model = KNeighborsRegressor(n_neighbors=self.k)

        if dimensions is not None:
            X_train = self.X_train[:, dimensions]
            X = X[:, dimensions]
        else:
            X_train = self.X_train

        self.model.fit(X_train, self.y_train)
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, dimensions=None):
        if not np.any(dimensions):
            return 0.0

        y_pred = self.predict(X_test, dimensions)

        if self.dataset_type == DatasetType.CLASSIFICATION:
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
        elif self.dataset_type == DatasetType.REGRESSION:
            mse = 1 - mean_squared_error(y_test, y_pred)
            return mse

    def permutation_importances(self, X_test, y_test):
      self.predict(self.X_train)
      self.model.fit(self.X_train, self.y_train)
      if self.dataset_type == DatasetType.CLASSIFICATION:
          scoring_metric = 'accuracy'
      elif self.dataset_type == DatasetType.REGRESSION:
          scoring_metric = 'neg_mean_squared_error'
      else:
          raise ValueError("Dataset type not allowed")
      importances = permutation_importance(self.model, X_test, y_test,
                                           scoring=scoring_metric)
      return np.array(importances.importances_mean)

class MLP:
    def __init__(
        self,
        dataset_type,
        hidden_layer_sizes,
        learning_rate_init,
        max_iter,
        random_state=42,
    ):
        self.random_state = random_state
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter

        self.dataset_type = dataset_type

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, dimensions=None):
        if self.dataset_type == DatasetType.CLASSIFICATION:
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
            )
        elif self.dataset_type == DatasetType.REGRESSION:
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )

        if dimensions is not None:
            X_train = self.X_train[:, dimensions]
            X = X[:, dimensions]
        else:
            X_train = self.X_train

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(X_train, self.y_train)

        return self.model.predict(X)

    def evaluate(self,X_test, y_test,dimensions=None):
        if not np.any(dimensions):
            return 0.0

        y_pred = self.predict(X_test, dimensions)

        if self.dataset_type == DatasetType.CLASSIFICATION:
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
        elif self.dataset_type == DatasetType.REGRESSION:
            mse = 1 - mean_squared_error(y_test, y_pred)
            return mse

    def permutation_importances(self, X_test, y_test):
      self.predict(self.X_train)
      self.model.fit(self.X_train, self.y_train)
      if self.dataset_type == DatasetType.CLASSIFICATION:
          scoring_metric = 'accuracy'
      elif self.dataset_type == DatasetType.REGRESSION:
          scoring_metric = 'neg_mean_squared_error'
      else:
          raise ValueError("Dataset type not allowed")
      importances = permutation_importance(self.model, X_test, y_test,
                                           scoring=scoring_metric)
      return np.array(importances.importances_mean)

class RandomForest:
  def __init__(self,dataset_type,max_depth, min_samples_split, random_state=42):
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.random_state = random_state
    self.dataset_type = dataset_type
    self.n_jobs = -1

  def fit(self, X, y):
        self.X_train = X
        self.y_train = y

  def predict(self, X, dimensions = None):
    if self.dataset_type == DatasetType.CLASSIFICATION:
      self.model = RandomForestClassifier(n_jobs=self.n_jobs,random_state= self.random_state)
    elif self.dataset_type == DatasetType.REGRESSION:
      self.model = RandomForestRegressor(n_jobs=self.n_jobs,max_depth=self.max_depth,
                                         min_samples_split = self.min_samples_split,
                                         random_state= self.random_state)
    if dimensions is not None:
      X_train = self.X_train[:, dimensions]
      X = X[:, dimensions]
    else:
      X_train = self.X_train

    self.model.fit(X_train, self.y_train)
    return self.model.predict(X)

  def evaluate(self, X_test, y_test, dimensions=None):
      if not np.any(dimensions):
          return 0.0

      y_pred = self.predict(X_test, dimensions)

      if self.dataset_type == DatasetType.CLASSIFICATION:
          accuracy = accuracy_score(y_test, y_pred)
          return accuracy
      elif self.dataset_type == DatasetType.REGRESSION:
          mse = 1 - mean_squared_error(y_test, y_pred)
          return mse

  def permutation_importances(self, X_test, y_test):
      self.predict(self.X_train)
      self.model.fit(self.X_train, self.y_train)
      if self.dataset_type == DatasetType.CLASSIFICATION:
          scoring_metric = 'accuracy'
      elif self.dataset_type == DatasetType.REGRESSION:
          scoring_metric = 'neg_mean_squared_error'
      else:
          raise ValueError("Dataset type not allowed")
      importances = permutation_importance(self.model, X_test, y_test,
                                           scoring=scoring_metric)
      return np.array(importances.importances_mean)

  def intrinsic_importances(self):
    self.predict(self.X_train)
    self.model.fit(self.X_train, self.y_train)
    return np.array(self.model.feature_importances_)