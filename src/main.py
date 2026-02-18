"""
Main Entry Point for the GAwLL Research Framework.

Orchestrates experiments by parsing CLI arguments, managing environment
variables for Linux performance, and delegating execution to the Pipeline.
"""

import argparse
import ast
import sys
import gc
import os
from pathlib import Path

# Ensure absolute/relative imports work correctly from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config.variables_config import get_variables_names
from src.utils.statistic import Statistics
from src.models.machine_learning_model import DT, KNN, MLP, RandomForest
from src.visualization.barplot import BarPlot
from src.visualization.graph import Graph
from src.visualization.values import SaveValues
from src.core.pipeline import GAwLLPipeline
from src.config.hardware_config import MODEL_N_JOBS
from src.utils.experiment_logger import ExperimentLogger

# --- LINUX PERFORMANCE TUNING ---
# Disable multi-threading in low-level libraries to prevent CPU contention
# when running GA evaluations in parallel.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def get_model_instance(model_name: str, args: argparse.Namespace):
    """
    Maps a model identifier string to its class and respective hyperparameter arguments.
    """
    try:
        if model_name == 'dt':
            return DT, {}
        elif model_name == 'rf':
            return RandomForest, {'n_jobs': MODEL_N_JOBS}
        elif model_name == 'knn':
            return KNN, {'k': args.knn_k, 'n_jobs': MODEL_N_JOBS}
        elif model_name == 'mlp':
            hidden_layers = ast.literal_eval(args.mlp_hidden)
            return MLP, {'hidden_layer_sizes': hidden_layers}
        else:
            print(f"    [WARNING] Model '{model_name}' not recognized.")
            return None, None
    except Exception as e:
        print(f"    [ERROR] Configuration error for model {model_name}: {e}")
        return None, None


def main():
    """
    Parses CLI arguments and executes the experimental pipeline.
    """
    parser = argparse.ArgumentParser(description="GAwLL Framework: XAI-Enhanced Feature Selection")

    # --- Required Arguments ---
    parser.add_argument("--datasets", nargs='+', required=True,
                        help="Datasets to process (e.g., boson zoo)")
    parser.add_argument("--models", nargs='+', required=True,
                        help="ML models to evaluate (e.g., dt rf knn mlp)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for reproducibility")

    # --- Evolutionary Algorithm Parameters ---
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Number of independent GA executions per experiment")
    parser.add_argument("--pop_size", type=int, default=50,
                        help="Population size")
    parser.add_argument("--max_gen", type=int, default=100,
                        help="Maximum number of generations")
    parser.add_argument("--cross_rate", type=float, default=0.8,
                        help="Crossover probability")
    parser.add_argument("--tau_reset", type=int, default=50,
                        help="Generations without improvement before population reset")

    # --- XAI and Linkage Settings (Opt-out Logic) ---
    parser.add_argument("--no_linkage", action="store_false", dest="linkage_learning",
                        help="Deactivate Linkage Learning (default is True)")
    parser.set_defaults(linkage_learning=True)

    parser.add_argument("--no_compare", action="store_false", dest="compare_methods",
                        help="Deactivate comparative methods like Permutation/Intrinsic (default is True)")
    parser.set_defaults(compare_methods=True)

    # --- Output and Hyperparameters ---
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top features to display in reports")
    parser.add_argument("--mlp_hidden", type=str, default="(16,8)",
                        help="Hidden layer architecture for MLP")
    parser.add_argument("--knn_k", type=int, default=3,
                        help="K neighbors for KNN")

    args = parser.parse_args()

    # Shared SQL logger for hardware and timing metrics
    db_logger = ExperimentLogger()

    # Logical Synchronization: Comparison requires Linkage results to be meaningful
    effective_compare = args.compare_methods if args.linkage_learning else False

    # --- Main Experiment Loop ---
    for ds_name in args.datasets:
        # Fetch human-readable variable names
        vars_names = get_variables_names(ds_name)
        if not vars_names:
            print(f"    [SKIPPING] Dataset '{ds_name}' not found in variable config.")
            continue

        for m_name in args.models:
            # Instantiate Statistics with its visual and saving handlers
            stats = Statistics(BarPlot(), Graph(), SaveValues(), db_logger)
            pipeline = GAwLLPipeline(stats)

            model_class, model_args = get_model_instance(m_name, args)
            if model_class is None:
                continue

            config = {
                'seed': args.seed,
                'n_runs': args.n_runs,
                'model_label': m_name,
                'dataset_name': ds_name,
                'variables': vars_names,
                'top_n': args.top_n,
                'compare_methods': effective_compare,
                'ga_params': {
                    'pop_size': args.pop_size,
                    'max_generations': args.max_gen,
                    'crossover_rate': args.cross_rate,
                    'tau_reset': args.tau_reset,
                    'linkage_learning': args.linkage_learning
                }
            }

            print(f"\n" + "═" * 75)
            print(f" STARTING EXPERIMENT: {ds_name.upper()} ║ MODEL: {m_name.upper()}")
            print(f" CONFIG: Linkage={args.linkage_learning} | Compare={effective_compare}")
            print("═" * 75)

            try:
                pipeline.run_experiment(ds_name, model_class, model_args, config)
            except Exception as e:
                import traceback
                print(f"\n    [CRITICAL ERROR] Experiment failed for {ds_name}/{m_name}: {e}")
                traceback.print_exc()
                continue
            finally:
                # Force memory cleanup between different model/dataset pairs
                del pipeline
                del stats
                gc.collect()

    db_logger.close()
    print("\n" + "═" * 75)
    print(" ALL SCHEDULED EXPERIMENTS COMPLETED SUCCESSFULLY.")
    print("═" * 75)


if __name__ == "__main__":
    main()