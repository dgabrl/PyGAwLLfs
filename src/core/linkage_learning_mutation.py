"""
Linkage Learning Mutation Module.

This module implements the core logic for detecting feature interactions (VInt)
and individual importance (VImp) through targeted bit-flip perturbations.
"""

import random
import numpy as np
from typing import List, Callable


class LinkageLearning:
    """
    Implements Linkage Learning mechanisms to estimate variable interactions.

    By analyzing performance changes when specific genes (g, h) are flipped,
    this class builds an empirical model of the search space's topology.
    """

    def __init__(self, evig, importance, eval_perf_batch: Callable, calc_fitness: Callable):
        """
        Initializes the Linkage Learning module.

        Args:
            evig: Instance of eVIG to store interactions.
            importance: Instance of Importance to store variable impact.
            eval_perf_batch: Callable that evaluates a batch of chromosomes.
            calc_fitness: Callable that evaluates the fitness of chromosomes.
        """
        self.evig = evig
        self.importance = importance
        self.eval_perf = eval_perf_batch
        self.eval_fit = calc_fitness

    def mutation_ll(self, parent) -> List[np.ndarray]:
        """
        Performs mutation guided by Linkage Learning.
        Generates 3 offspring to probe the search space for interactions.

        Returns:
            List[np.ndarray]: [xg_chrom, xh_chrom, xgh_chrom] for the GA engine.
        """
        EPS2 = 1.0e-10
        parent_chrom = parent.chromosome
        chrom_size = len(parent_chrom)
        fx = parent.fitness

        # 1. Random Selection of two distinct genes
        g, h = random.sample(range(chrom_size), 2)

        # 2. Generating perturbed chromosomes
        # Using boolean NOT for fast flipping
        xg_chrom = parent_chrom.copy()
        xg_chrom[g] = not xg_chrom[g]

        xh_chrom = parent_chrom.copy()
        xh_chrom[h] = not xh_chrom[h]

        xgh_chrom = xg_chrom.copy()
        xgh_chrom[h] = not xgh_chrom[h]

        # 3. Batch Evaluation
        # We send all 3 to the batch evaluator at once
        # and evaluate their fitness
        pxg, pxh, pxgh = self.eval_perf([xg_chrom, xh_chrom, xgh_chrom])

        fxg = self.eval_fit(xg_chrom, pxg)
        fxh = self.eval_fit(xh_chrom, pxh)
        fxgh = self.eval_fit(xgh_chrom, pxgh)

        # 4. Update VInt and VImp
        # --- Variable Importance (VImp) ---
        # Impact of g alone
        df_g = abs(fxg - fx)
        if df_g > EPS2:
            self.importance.add_importance(g, df_g)

        # Impact of h alone
        df_h = abs(fxh - fx)
        if df_h > EPS2:
            self.importance.add_importance(h, df_h)

        # --- Variable Interaction (VInt) ---
        # Interaction is the difference between the joint impact and sum of individual impacts
        df_interaction = abs(fxgh - fxg - fxh + fx)

        if df_interaction > EPS2:
            # Symmetrically update the eVIG matrix
            self.evig.add_edge(g, h, df_interaction)

            # Conditional Importance
            df_g_cond = abs(fxgh - fxh)
            if df_g_cond > EPS2:
                self.importance.add_importance(g, df_g_cond)

            df_h_cond = abs(fxgh - fxg)
            if df_h_cond > EPS2:
                self.importance.add_importance(h, df_h_cond)

        return [xg_chrom, xh_chrom, xgh_chrom]