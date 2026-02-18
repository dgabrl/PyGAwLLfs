# PyGAwLLfs: Genetic Algorithm with Linkage Learning for Feature Selection in Python (V2)

**PyGAwLLfs** is a high-performance research framework designed for **Feature Selection** and **Explainable AI (XAI)**. It utilizes evolutionary computation to detect feature interactions (Linkage Learning), generating Interaction Matrix ($VInt$) and Importance Vectors ($VImp$).

## 🧬 Key Evolutionary Innovations (V2)

The V2 implementation introduces significant architectural and algorithmic upgrades focused on scalability and biological plausibility:

* **High-Intensity Elite Reset:** When the population reaches stagnation (`tau_reset`), the framework preserves the elite individual and generates a new population through **aggressive mutation** in this individual, rather than simple random initialization.
* **Batch Performance Evaluation:** Individuals are evaluated in optimized batches using parallel processing to maximize throughput.
* **RAM Escalator (Memory Safeguard):** A dynamic monitoring system implemented in the pipeline that manages worker lifespans (`maxtasksperchild`) based on RAM usage (90%, 94%, and 97% thresholds) to prevent memory leaks in Linux environments.

---

## 🏗 Modular Architecture

### 📂 `src/core/`
* **GA Operators:** Core logic for selection, crossover, and mutation.
* **Linkage Learning Mutation:** The engine that builds interaction graphs and estimates $VInt$ and $VImp$ during the mutation phase.
* **Pipeline:** Orchestrates independent runs, manages hardware resources, and implements the **RAM Escalator**.

### 📂 `src/models/`
* **Refactored ML Interface:** Features a `BaseModel` inheritance system. Specific models (DT, RF, KNN, MLP) inherit from this base, ensuring a consistent interface for the GA core.

### 📂 `src/config/` (User Configuration)
You can customize the framework behavior without changing the core logic:

| Module | Purpose | Key Adjustments |
| :--- | :--- | :--- |
| **Hardware** | Performance & Safety | CPU fraction, Cache limits, Parallelism rules. |
| **Datasets** | Data Path Management | Path resolution via `pathlib` and dataset registry. |
| **Variables**| XAI & Feature Mapping | Mapping chromosome indices to human-readable names. |

### 📂 `src/visualization/` & `src/utils/`
* **Visualization:** Generates Full/Reduced Interaction Graphs (NetworkX), Barplots (Raw/Normalized), and metrics in `.txt` and `.csv`.
* **Utils:** Data loading, train/test splitting, and `experiment_logger.py` for SQLite persistence of hardware metrics and timing.

---

## 📊 Data & Results
* **Input**: Datasets must be in `.dat` format. Use the `GAwLLfs-Dataset-Converter` to transform `.csv` files.
* **Outputs**: Results are saved in the `/results` folder, including interaction graphs, barplots, and the SQLite database.

---

## ⚙️ Mandatory Configuration (Before Running)

Since GAwLL is a research framework, you **must** configure the following files in `src/config/` to match your environment and dataset:

### 1. Hardware Setup (`hardware_config.py`)
Adjust the computational limits based on your machine's capacity:
* Set `CPU_USAGE_FRACTION` to define how much of your processor PyGAwLLfs can use.
* Ensure `CACHE_LIMIT` is compatible with your available RAM to prevent fragmentation.

### 2. Dataset Registration (`datasets_config.py`)
Add your `.dat` files to the `DATASETS` dictionary:
```python
DATASETS = {
    "my_experiment": DATA_DIR / "my_data.dat",
}
```

### 3. Feature Mapping (`variables_config.py`)
For XAI metrics to work, you must map the chromosome indices to human-readable names.
```python
configs = {
    'my_experiment': [
        "x_0", "x_1", "x_2"
    ]
}
```

---

## 🚀 Execution & Usage

You can execute GAwLL V2 in two ways: using the interactive orchestrator for standard runs or the Command Line Interface (CLI) for advanced research configurations.

### 1. Interactive Orchestrator
The project includes a Bash script that automates environment setup, dependency checks, and prompts you for the basic parameters.

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

> **Note:** This script will create the venv_gawll, install requirements.txt, and ask you which datasets and models to run.

### 2. Advanced CLI usage (Argparse)
For fine-grained control or cluster execution, call `src/main.py` directly.

**Required Arguments**
* `--datasets`: One or more datasets names (must exist in `datasets_config.py` datasets dictionary).
  * Example: `--datasets boson zoo`.
* `--models`: One or more ML models to test.
  * Options: `dt`(Decision or Regression Tree), `rf` (Random Forest), `mlp` (Multi-Layer Perceptron), `knn` (K-Nearest Neighbors).
  * Example: `--models dt mlp`.

**Optional Arguments (Evolutionary and XAI)**
| Argument | Default | Description |
| :--- | :----: | ---: |
| `--seed` | 42 | Base random seed for reproducibility. |
| `--n_runs` | 10 | Number of independent GA executions per experiment. |
| `--pop_size` | 50 | Size of the population. |
| `--max_gen` | 100 | Maximum generations per run. |
| `--cross_rate` | 0.4 | Crossover probability. |
| `--tau_reset` | 50 | Generations without improvement before triggering High-Intensity Reset. |
| `--no_linkage` | False | Use this to deactivate Linkage Learning (disables XAI outputs). |
| `--no_compare` | False | Use this to deactivate comparison with Permutation/Intrinsic methods. |
| `--top_n` | 10 | Number of top importances/interactions to display in the final reports. |

**Model Hyperparameters**
| Argument | Default | Description |
| :--- | :----: | ---: |
| `--mlp_hidden` | "(8,)" | Hidden layer architecture for MLP. |
| `--knn_k` | 3 | K neighbors for KNN. |

* Example of complex research command:
  
```bash
python3 src/main.py --datasets boson --models rf mlp --n_runs 30 --pop_size 100 --tau_reset 20 --mlp_hidden "(32,16,8)"
```
