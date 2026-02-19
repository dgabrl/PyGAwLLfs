"""
Pipeline Module for GAwLL Execution.

Orchestrates data loading, parallel execution environments,
and the integration between the Genetic Algorithm and Machine Learning models.
Includes dynamic RAM management and peak memory logging.
"""

import time
import multiprocessing
import gc
import traceback
import os
import psutil
import numpy as np
from typing import List, Dict, Any, Type
from src.core.gawll import GAwLL
from src.utils.util import Util
from src.config.hardware_config import SAFE_CORES


def worker_perf_task(task_data: tuple) -> float:
    """
    Independent worker function for parallel execution.
    Instantiates, trains, and evaluates a model on a specific feature mask.
    """
    feature_mask, model_class, dataset_type, model_args, data = task_data
    try:
        model = model_class(dataset_type=dataset_type, **model_args)
        # Using the refactored context setting method
        model.set_training_data(data['X_train'], data['y_train'])

        # Evaluate using the feature_mask (the selected columns)
        score = model.evaluate(data['X_test'], data['y_test'], feature_mask=feature_mask)

        model = None
        return float(score)
    except Exception:
        traceback.print_exc()
        return 0.0

class GAwLLPipeline:
    """
    Orchestrates the execution of GAwLL experiments across multiple datasets.
    Handles parallel pools, memory safety, and result persistence.
    """

    def __init__(self, stats_manager):
        """
        Initializes the pipeline with a statistics manager.
        """
        self.stats = stats_manager
        self.db_logger = stats_manager.db_logger

    def run_experiment(self, dataset_name: str, model_class: Type, model_args: Dict, config: Dict):
        """
        Executes the full experiment suite for a given dataset/model combination.
        """
        try:
            # Loading data
            (dataset_type, chrom_size, X_train, y_train, X_test, y_test) = (
                Util.load_dataset(dataset_name, train_ratio=0.70)
            )
        except Exception as e:
            print(f"    [ERROR] Failed to load {dataset_name}: {e}")
            return

        shared_data = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }

        # Accumulators for multi-run averages
        sum_vint, sum_vimp = None, None
        sum_perm, sum_intr = None, None
        count_success = 0
        best_ind_run = []

        # Start SQL experiment tracking
        id_group = self.db_logger.register_experiment(
            dataset_name, config['model_label'], config['n_runs']
        )

        is_linkage_enabled = config['ga_params']['linkage_learning']
        ctx = multiprocessing.get_context('spawn')

        for r in range(config['n_runs']):
            current_seed = config['seed'] + r
            print(f"\n >>> Run {r + 1}/{config['n_runs']} (Seed: {current_seed})\n")

            # Define the dynamic parallel pool
            current_max_tasks = None 
            pool = ctx.Pool(processes=SAFE_CORES, maxtasksperchild=current_max_tasks)
            
            def batch_evaluator(feature_masks: List[np.ndarray]) -> List[float]:
                """
                Parallel evaluator with dynamic RAM protection (Safety Lock).
                """
                nonlocal pool, current_max_tasks
                mem_percent = psutil.virtual_memory().percent
                
                if mem_percent > 90:
                    target_tasks = 50 if mem_percent <= 95 else (15 if mem_percent <= 97 else 5)
                    
                    if current_max_tasks is None or target_tasks < current_max_tasks:
                        print(f"\n [RAM] {mem_percent}% usage. Reconfiguring workers to {target_tasks}...")
                        
                        pool.close()
                        pool.terminate() 
                        pool.join()
                        gc.collect()     
                        time.sleep(3)    
                        
                        current_max_tasks = target_tasks
                        pool = ctx.Pool(processes=SAFE_CORES, maxtasksperchild=current_max_tasks)

                # Dispatching tasks
                tasks = [(fm, model_class, dataset_type, model_args, shared_data) for fm in feature_masks]
                try:
                    chunk_size = max(1, len(tasks) // (SAFE_CORES * 2))
                    result_async = pool.map_async(worker_perf_task, tasks, chunksize=chunk_size)
                    return result_async.get(timeout=3600)
                except Exception as e:
                    print(f" [ERROR] Mapping failed: {e}")
                    return [0.0] * len(feature_masks)

            # --- EXECUTION BLOCK ---
            try:
                  # 1. GAwLL Execution
                  ga = GAwLL(perf_batch=batch_evaluator,
                             chrom_size=chrom_size,
                             mutation_probability=1.0 / chrom_size,
                             **config['ga_params'])
  
                  start_gawll = time.time()
                  ga.run(current_seed)
                  t_gawll = time.time() - start_gawll
  
                  # 2. Comparative Methods
                  t_perm, t_intr = 0.0, 0.0
                  imp_perm, imp_intr = None, None
                  if is_linkage_enabled and config['compare_methods']:
                      # Model training to obtain permutation importance and intrinsic importance
                      model = model_class(dataset_type=dataset_type, **model_args)
                      model.set_training_data(X_train, y_train)
  
                      start_p = time.time()
                      imp_perm = model.permutation_importances(X_test, y_test)
                      t_perm = time.time() - start_p
  
                      if config['model_label'] in ['dt', 'rf']:
                          imp_intr = model.intrinsic_importances(X_test)
                          t_intr = time.time() - (start_p + t_perm)
  
                      del model
                      gc.collect()
  
                  # 4. Getting the data
                  # Results from the Feature Selection problem: Individual (chromosome, fitness, performance)
                  res_fs = ga.get_feature_selection_results()
                  best_ind_run.append(res_fs)
                  
                  # Results from the Linkage Learning Mutation (VInt and VImp)
                  if is_linkage_enabled:
                      vimp_vector = ga.importance.export_importance_vector()
                      vint_matrix = ga.evig.export_interaction_matrix()
  
                      # Sum the VImp and VInt to obtain the average of VInt and VImp from different methods
                      if sum_vimp is None:
                          sum_vimp, sum_vint = vimp_vector, vint_matrix
                          sum_perm = imp_perm if imp_perm is not None else None
                          sum_intr = imp_intr if imp_intr is not None else None
                      else:
                          np.add(sum_vimp, vimp_vector, out=sum_vimp)
                          np.add(sum_vint, vint_matrix, out=sum_vint)
                          if imp_perm is not None: np.add(sum_perm, imp_perm, out=sum_perm)
                          if imp_intr is not None: np.add(sum_intr, imp_intr, out=sum_intr)
                      # Counter to perform the average
                      count_success += 1
                  
                  try:
                      peak_ram_usage = psutil.Process().memory_info().rss / (1024 * 1024)
                  except:
                      peak_ram_usage = 0.0
                      
                  # SQL Logging
                  run_data = {
                      'run': r + 1, 'time_gawll': t_gawll,
                      'time_perm': t_perm, 'time_intr': t_intr,
                      'peak_ram': peak_ram_usage
                  }
                  self.db_logger.log_run(id_group, run_data)
                  
                  ga.cleanup()
                  batch_func = None
                  del ga          
                  for _ in range(3):
                      gc.collect()
                  time.sleep(3)
                
                
            except Exception as e:
                print(f"    [ERROR Run {r + 1}] {dataset_name}: {e}")
                traceback.print_exc()
            
            finally:
                if 'pool' in locals() and pool is not None:
                    try:
                        pool.terminate() 
                        pool.join()      
                    except:
                        pass
                
                gc.collect()
                time.sleep(3) 

        # --- Save Outputs ---
        if not is_linkage_enabled:
            self.stats.save_feature_selection_statistics(best_ind_run, config)
            self.stats.db_logger.finalize_summary(id_group)
        
        elif is_linkage_enabled and count_success > 0:
            aggregates = {
                'id_group': id_group,
                'vint_gawll': sum_vint / count_success,
                'vimp_gawll': sum_vimp / count_success,
                'vimp_perm': sum_perm / count_success if sum_perm is not None else None,
                'vimp_intr': sum_intr / count_success if sum_intr is not None else None
            }
            self.stats.save_all_results(best_ind_run, aggregates, config)      
