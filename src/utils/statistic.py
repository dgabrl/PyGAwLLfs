import numpy as np
from pathlib import Path

class Statistics:
    """
    Manages the evolutionary data, XAI metrics, and statistical analysis across runs.
    Centralizes result persistence in text reports, visualizations, and SQL.
    """

    def __init__(self, plotter, grapher, saver, db_logger):
        """
        Initializes the Statistics manager.

        Args:
            plotter: Object for generating importance plots.
            grapher: Object for interaction matrices visualizations.
            saver: Object for file system operations (CSV, TXT).
            db_logger: Object for SQL database persistence.
        """
        self.plotter = plotter
        self.grapher = grapher
        self.saver = saver
        self.db_logger = db_logger

        # Directory handling - uses saver's path if available
        self.output_path = getattr(saver, 'output_path', Path('results'))
        self.output_path.mkdir(parents=True, exist_ok=True)

    def save_all_results(self, best_ind_run, aggregates, config):
        """
        Main entry point to save all experiment outputs after completion.
        Note: Execution times are handled via SQL during the runs.

        Args:
            best_ind_run (list): Results per run (chromosome, fitness, performance).
            aggregates (dict): Calculated means for VInt, VImp and comparative methods.
            config (dict): Experiment configuration parameters.
        """
        # 1. Save XAI Metrics (Matrices and Plots)
        self.save_xai_metrics(config, **aggregates)

        # 2. Save Feature Selection Statistics (Text Report)
        self.save_feature_selection_statistics(best_ind_run, config)

        # 3. Finalize SQL Summary (Aggregates performance in DB)
        self.db_logger.finalize_summary(aggregates['id_group'])

    def save_xai_metrics(self, config, vint_gawll, vimp_gawll, vimp_perm, vimp_intr, **kwargs):
        """Saves interaction matrices, importance vectors, and generates plots."""
        prefix = f"{config['dataset_name']}_{config['model_label']}"
        vars_names = list(config['variables'])

        # CSV Data (Raw matrices for future analysis)
        self.saver.save_interaction_matrix(vint_gawll, f"{prefix}_Mean_VInt_GAwLL")
        self.saver.save_importances(vimp_gawll, f"{prefix}_Mean_VImp_GAwLL")

        if vimp_perm is not None: self.saver.save_importances(vimp_perm, f"{prefix}_Mean_VImp_Permutation")
        if vimp_intr is not None: self.saver.save_importances(vimp_intr, f"{prefix}_Mean_VImp_Intrinsic")

        # Top Rankings (Detailed TXT for quick inspection)
        self.saver.save_top_importances(vimp_gawll, vars_names, config['model_label'], config['dataset_name'],
                                        config['top_n'])
        self.saver.save_top_interactions(vint_gawll, vars_names, config['model_label'], config['dataset_name'],
                                         config['top_n'])

        # Visualizations (PNG/PDF)
        self.grapher.save_all_visualizations(vimp_gawll, vint_gawll, vars_names, prefix)
        self.plotter.generate_importance_plots(vars_names, config['model_label'],
                                               config['dataset_name'], vimp_gawll)

    def save_feature_selection_statistics(self, best_ind_run, config):
        """
        Generates a consolidated text report with detailed statistics on fitness,
        accuracy, and feature subset sizes.

        Args:
            best_ind_run (list): Results from each run (individual, fitness, performance).
            config (dict): Experiment configuration including variables names and model label.
        """
        vars_names = list(config['variables'])

        # Extraction of metrics and feature counts per run
        all_fitness = np.array([ind[1] for ind in best_ind_run])
        all_perfs = np.array([ind[2] for ind in best_ind_run])
        all_counts = np.array([int(np.sum(ind[0])) for ind in best_ind_run])

        # Identifying critical indices for extreme cases
        idx_max_fit = np.argmax(all_fitness)
        idx_min_fit = np.argmin(all_fitness)
        idx_max_acc = np.argmax(all_perfs)
        idx_min_acc = np.argmin(all_perfs)

        # Global best solution based on Max Fitness
        best_chrom = best_ind_run[idx_max_fit][0]
        selected_vars = [vars_names[i] for i, val in enumerate(best_chrom) if val == 1]

        report = [
            "=" * 75,
            f"CONSOLIDATED REPORT: {config['dataset_name'].upper()} | {config['model_label'].upper()}",
            "=" * 75 + "\n",
            "1. RESULTS PER RUN",
            "-" * 35
        ]

        for i, (chrom, fit, perf) in enumerate(best_ind_run):
            report.append(f"Run {i:02d} | Fitness: {fit:.5f} | Acc: {perf:.5f} | Vars: {all_counts[i]}")

        report.extend([
            "\n2. GLOBAL BEST SOLUTION (Max Fitness)",
            "-" * 35,
            f"Best Fitness:      {all_fitness[idx_max_fit]:.5f}",
            f"Best Performance:  {all_perfs[idx_max_fit]:.5f}",
            f"Features Selected: {all_counts[idx_max_fit]}",
            f"Names: {', '.join(selected_vars)}",

            "\n3. EXTREME METRICS",
            "-" * 35,
            f"Max Accuracy: {all_perfs[idx_max_acc]:.5f} (Vars: {all_counts[idx_max_acc]})",
            f"Min Accuracy: {all_perfs[idx_min_acc]:.5f} (Vars: {all_counts[idx_min_acc]})",
            f"Max Fitness:  {all_fitness[idx_max_fit]:.5f} (Vars: {all_counts[idx_max_fit]})",
            f"Min Fitness:  {all_fitness[idx_min_fit]:.5f} (Vars: {all_counts[idx_min_fit]})",

            "\n4. GENERAL STATISTICS",
            "-" * 35,
            f"Fitness  -> Mean: {np.mean(all_fitness):.5f} | Std: {np.std(all_fitness):.5f}",
            f"Accuracy -> Mean: {np.mean(all_perfs):.5f} | Std: {np.std(all_perfs):.5f} ",
            f"Features -> Mean: {np.mean(all_counts):.2f} | Std: {np.std(all_counts):.2f}"
        ])

        file_path = self.saver.output_path / f"Final_Report_{config['dataset_name']}_{config['model_label']}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

class PopulationAnalyzer:
    """
    Static tool to get the population fittest individual.
    """
    @staticmethod
    def get_fittest_individual(population):
        """Identifies the individual with the highest fitness value."""
        if not population:
            return None

        return max(population, key=lambda ind: ind.fitness)