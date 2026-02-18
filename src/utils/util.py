"""
Utility Module for Data Loading and Preprocessing.

Handles the parsing of custom .dat files and manages the train/test splitting
logic for both classification and regression tasks.
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from src.config.datasets_config import DATASETS


class DatasetType(Enum):
    """Enumeration to distinguish between classification and regression problems."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Util:
    """
    Utility class providing static methods for data handling.
    """

    @staticmethod
    def load_dataset(dataset_name: str, train_ratio: float) -> Optional[Tuple]:
        """
        Parses a .dat file and prepares the training and testing sets.

        Args:
            dataset_name: The key corresponding to the dataset in DATASETS config.
            train_ratio: Proportion of the data (0.0 to 1.0) to use for training.

        Returns:
            A tuple containing:
            (dataset_type, chrom_size, x_train, y_train, x_test, y_test)
            Returns None if the dataset is not found or parsing fails.
        """
        file_path = DATASETS.get(dataset_name)

        if not file_path or not file_path.exists():
            print(f"[ERROR] Dataset file not found: {dataset_name}")
            return None

        # Metadata initialization
        prob_type: Optional[int] = None
        chrom_size: Optional[int] = None
        n_examples: Optional[int] = None

        # Data containers
        x_values: Optional[np.ndarray] = None
        y_values: Optional[np.ndarray] = None

        with open(file_path, "r") as fin:
            lines = fin.readlines()

        # Phase 1: Parsing Metadata and Headers
        for i, line in enumerate(lines):
            tokens = line.split()
            if not tokens:
                continue

            # Extract keyword (removing the trailing colon if present)
            keyword = tokens[0].rstrip(':')

            if keyword == "TYPE":
                prob_type = int(tokens[1])
            elif keyword == "N_ATTRIBUTES":
                chrom_size = int(tokens[1])
            elif keyword == "N_EXAMPLES":
                n_examples = int(tokens[1])
                # Initialize arrays once metadata is known
                x_values = np.zeros((n_examples, chrom_size))
                # prob_type 1: Classification, others: Regression
                y_dtype = int if prob_type == 1 else float
                y_values = np.zeros(n_examples, dtype=y_dtype)

            # Phase 2: Loading Raw Data
            elif keyword == "DATASET" and n_examples is not None:
                # Start reading from the next line after the keyword 'DATASET'
                for row_idx in range(n_examples):
                    data_tokens = lines[i + 1 + row_idx].split()

                    # Fill feature matrix
                    x_values[row_idx, :] = [float(val) for val in data_tokens[:chrom_size]]

                    # Fill target vector
                    y_values[row_idx] = y_dtype(data_tokens[chrom_size])
                break

        # Phase 3: Train/Test Split
        if x_values is None or y_values is None:
            print(f"[ERROR] Failed to load data for: {dataset_name}")
            return None

        # Classification uses stratification to keep class proportions balanced
        is_classification = (prob_type == 1)
        stratify_param = y_values if is_classification else None

        x_train, x_test, y_train, y_test = train_test_split(
            x_values,
            y_values,
            test_size=(1.0 - train_ratio),
            random_state=42,
            stratify=stratify_param
        )

        dataset_type = DatasetType.CLASSIFICATION if is_classification else DatasetType.REGRESSION

        return (
            dataset_type,
            chrom_size,
            x_train,
            y_train,
            x_test,
            y_test
        )