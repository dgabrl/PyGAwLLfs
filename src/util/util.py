import os
from enum import Enum
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASETS = {
    "boson": os.path.join(DATA_DIR, "boson.dat"),
}

class DatasetType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Util:
    @staticmethod
    def read_dataset(dataset_name,perc_train):
        try:
            with open(DATASETS[dataset_name], "r") as fin:
                lines = fin.readlines()
        except FileNotFoundError:
            print("file name error")
            return

        prob_type = None
        n_examples = None
        chrom_size = None
        x_dataset = None
        d_dataset = None
        d_dataset_regression = None
        x_trainset = None
        x_testset = None
        d_trainset = None
        d_testset = None
        x_trainset_regression = None
        x_testset_regression = None
        d_trainset_regression = None
        d_testset_regression = None

        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            keyword = tokens[0][0:-1]

            if keyword == "TYPE":
                try:
                    prob_type = int(tokens[1])
                except ValueError:
                    print("TYPE error")
                    return
            elif keyword == "N_ATTRIBUTES":
                try:
                    chrom_size = int(tokens[1])
                except ValueError:
                    print("N_ATTRIBUTES error")
                    return
            elif keyword == "N_EXAMPLES":
                try:
                    n_examples = int(tokens[1])
                    x_dataset = np.zeros((n_examples, chrom_size))
                    if prob_type == 1:
                        d_dataset = np.zeros(n_examples, dtype=int)
                    else:
                        d_dataset_regression = np.zeros(n_examples)
                except ValueError:
                    print("N_EXAMPLES error")
                    return
            elif keyword == "DATASET" and n_examples is not None:
                dataset_lines = lines[
                                lines.index(line) + 1: lines.index(line) + 1 + n_examples
                                ]
                for i, data_line in enumerate(dataset_lines):
                    data_tokens = data_line.split()
                    for j in range(chrom_size):
                        x_dataset[i][j] = float(data_tokens[j])
                    if prob_type == 1:
                        d_dataset[i] = int(data_tokens[chrom_size])
                    else:
                        d_dataset_regression[i] = float(data_tokens[chrom_size])
                break


        if prob_type == 1:
            x_trainset, x_testset, d_trainset, d_testset = train_test_split(
                x_dataset, d_dataset, test_size=(1 - perc_train), random_state=42, stratify=d_dataset
            )
            dataset_type = DatasetType.CLASSIFICATION
        else:
            x_trainset_regression, x_testset_regression, d_trainset_regression, d_testset_regression = train_test_split(
                x_dataset, d_dataset_regression, test_size=(1 - perc_train), random_state=42
            )
            dataset_type = DatasetType.REGRESSION

        return (
            dataset_type,
            chrom_size,
            x_trainset if prob_type == 1 else x_trainset_regression,
            d_trainset if prob_type == 1 else d_trainset_regression,
            x_testset if prob_type == 1 else x_testset_regression,
            d_testset if prob_type == 1 else d_testset_regression,
        )
