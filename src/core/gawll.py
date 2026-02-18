"""
GAwLL: Genetic Algorithm with Linkage Learning Engine.

This module implements the core evolutionary logic, integrating parsimony
pressure, stagnation-based resets (Tau-Reset), and Linkage Learning
capabilities for high-dimensional feature selection.
"""

import gc
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable

# Project-specific imports
from src.core.evig import eVIG
from src.core.importance import Importance
from src.utils.statistic import PopulationAnalyzer
from src.core.ga_operators import Selection, Crossover, Mutation
from src.core.linkage_learning_mutation import LinkageLearning
from src.config.hardware_config import CACHE_LIMIT


class Individual:
    """
    Represents a single solution in the evolutionary population.

    Using __slots__ optimizes memory by preventing the creation of __dict__
    for each instance, which is critical for GA performance in Python.

    Attributes:
        chromosome (np.ndarray): Binary vector representing selected features.
        fitness (Optional[float]): Scalar value (Performance + Parsimony).
        f_perf (Optional[float]): Raw performance metric (e.g., accuracy).
    """
    __slots__ = ['chromosome', 'fitness', 'f_perf']

    def __init__(
            self,
            chromosome: np.ndarray,
            fitness: Optional[float] = None,
            f_perf: Optional[float] = None
    ):
        self.chromosome: np.ndarray = chromosome
        self.fitness: Optional[float] = fitness
        self.f_perf: Optional[float] = f_perf


class GAwLL:
    """
    Main Engine for the Genetic Algorithm with Linkage Learning (GAwLL).

    This class orchestrates the evolutionary lifecycle: initialization,
    evaluation, selection, recombination, and mutation.

    Attributes:
        EPSILON (float): Small constant for robust floating-point comparisons.
    """
    EPSILON: float = 1.0e-10

    def __init__(
            self,
            *,
            perf_batch: Callable[[List[np.ndarray]], List[float]],
            chrom_size: int,
            mutation_probability: float,
            max_generations: int,
            pop_size: int,
            crossover_rate: float,
            tau_reset: int,
            linkage_learning: bool,
    ):
        """
        Initializes the GAwLL instance with evolutionary hyperparameters.

        Args:
            perf_batch: Function to evaluate a batch of chromosomes.
            chrom_size: Length of the binary chromosome.
            mutation_probability: Rate of bit-flip mutation.
            max_generations: Maximum number of generations to run.
            pop_size: Size of the population.
            crossover_rate: Probability of crossover occurrence.
            tau_reset: Generations of stagnation before triggering a reset.
            linkage_learning: Boolean flag to enable Linkage Learning mutation.
        """
        # Evaluation settings
        self.perf_batch: Callable = perf_batch
        self.chrom_size: int = chrom_size

        # GA hyperparameters
        self.mutation_probability: float = mutation_probability
        self.max_generations: int = max_generations
        self.population_size: int = pop_size
        self.crossover_rate: float = crossover_rate
        self.tau_reset_generations: int = tau_reset
        self.linkage_learning: bool = linkage_learning

        # Component placeholders
        self.population_analyzer: Optional[PopulationAnalyzer] = None
        self.selection: Optional[Selection] = None
        self.crossover: Optional[Crossover] = None
        self.mutation: Optional[Mutation] = None
        self.mutation_ll: Optional[LinkageLearning] = None

        # Evolutionary state
        self.population: List[Individual] = []
        self.evig: Optional[eVIG] = None
        self.importance: Optional[Importance] = None
        self.final_fittest: Optional[Individual] = None

        # Performance cache (FIFO strategy)
        self.cache: Dict[bytes, float] = {}

    def _calculate_fitness_value(self, chromosome: np.ndarray, f_perf: float) -> float:
        """
        Calculates the combined fitness (98% Performance, 2% Parsimony).

        Args:
            chromosome: The binary individual array.
            f_perf: Raw performance score from the ML model.

        Returns:
            float: The calculated fitness score.
        """
        parsimony = 1.0 - np.mean(chromosome)
        return float(0.98 * f_perf + 0.02 * parsimony)

    def _evaluate_individuals(self, individuals: List[Individual]) -> None:
        """
        Evaluates a list of individuals using batching and FIFO caching.

        Args:
            individuals: List of Individual objects to be evaluated.
        """
        to_evaluate: List[Individual] = []

        # 1. Check cache first to avoid redundant evaluations
        for ind in individuals:
            chrom_hash = ind.chromosome.tobytes()
            if chrom_hash in self.cache:
                ind.f_perf = self.cache[chrom_hash]
                ind.fitness = self._calculate_fitness_value(ind.chromosome, ind.f_perf)
            elif ind.f_perf is None:
                to_evaluate.append(ind)

        if not to_evaluate:
            return

        # 2. Perform Batch Evaluation (CPU/GPU Bottleneck)
        chromosomes = [ind.chromosome for ind in to_evaluate]
        scores = self.perf_batch(chromosomes)

        # 3. Update Individuals and populate Cache
        for ind, score in zip(to_evaluate, scores):
            ind.f_perf = score
            ind.fitness = self._calculate_fitness_value(ind.chromosome, score)

            # Manage Cache size using FIFO (First-In, First-Out)
            chrom_hash = ind.chromosome.tobytes()
            if len(self.cache) >= CACHE_LIMIT:
                # Remove the oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[chrom_hash] = score

    def _initialize_population(self) -> None:
        """Initializes the population with random binary chromosomes."""
        bits = np.random.randint(0, 2, size=(self.population_size, self.chrom_size)).astype(bool)
        self.population = [Individual(bits[i]) for i in range(self.population_size)]
        self._evaluate_individuals(self.population)

    def _apply_elitism(self, new_population: List[Individual], fittest_individual: Individual) -> None:
        """
        Protects the best found solution by injecting it into the new generation.

        Args:
            new_population: The generation currently being formed.
            fittest_individual: The best individual from the previous state.
        """
        # Find the index of the worst individual in the new population
        # We use -1.0 as fallback for uninitialized fitness
        worst_idx = min(range(len(new_population)), key=lambda i: new_population[i].fitness or -1.0)

        # Inject a copy of the fittest individual
        self.population[worst_idx] = Individual(
            fittest_individual.chromosome.copy(),
            fittest_individual.fitness,
            fittest_individual.f_perf
        )

    def _generation(self, fittest_individual: Individual) -> List[Individual]:
        """
        Performs a standard evolutionary generation cycle.

        Args:
            fittest_individual: The current global best for elitism.

        Returns:
            List[Individual]: The newly formed population.
        """
        new_population: List[Individual] = []

        while len(new_population) < self.population_size:
            parent1 = self.selection.tournament_selection(self.population)

            if len(new_population) < self.population_size - 1:
                parent2 = self.selection.tournament_selection(self.population)

                off1_chrom, off2_chrom = parent1.chromosome.copy(), parent2.chromosome.copy()

                if random.random() < self.crossover_rate:
                    off1_chrom, off2_chrom = self.crossover.uniform_crossover(off1_chrom, off2_chrom)

                self.mutation.bit_flip_mutation(off1_chrom, self.mutation_probability)
                self.mutation.bit_flip_mutation(off2_chrom, self.mutation_probability)

                new_population.extend([Individual(off1_chrom), Individual(off2_chrom)])
            else:
                off_chrom = parent1.chromosome.copy()
                self.mutation.bit_flip_mutation(off_chrom, self.mutation_probability)
                new_population.append(Individual(off_chrom))

        self._evaluate_individuals(new_population)
        self._apply_elitism(new_population, fittest_individual)
        return new_population

    def _generation_ll(self, fittest_individual: Individual) -> List[Individual]:
        """
        Evolutionary generation cycle enhanced with Linkage Learning mutation.

        Args:
            fittest_individual: The current global best for elitism.

        Returns:
            List[Individual]: The newly formed population.
        """
        new_population: List[Individual] = []

        while len(new_population) < self.population_size:
            parent1 = self.selection.tournament_selection(self.population)

            # 1. Exploratory Crossover
            num_crossover = int(self.population_size * self.crossover_rate)
            if len(new_population) < num_crossover:
                parent2 = self.selection.tournament_selection(self.population)
                off1_chrom, off2_chrom = self.crossover.uniform_crossover(
                    parent1.chromosome.copy(), parent2.chromosome.copy()
                )
                self.mutation.bit_flip_mutation(off1_chrom, self.mutation_probability)
                self.mutation.bit_flip_mutation(off2_chrom, self.mutation_probability)
                new_population.extend([Individual(off1_chrom), Individual(off2_chrom)])

            # 2. Linkage Learning Guided Mutation
            elif len(new_population) < self.population_size - 2:
                chroms = self.mutation_ll.mutation_ll(parent1)
                new_population.extend([Individual(c) for c in chroms])

            # 3. Residual filling
            else:
                off_chrom = parent1.chromosome.copy()
                self.mutation.bit_flip_mutation(off_chrom, self.mutation_probability)
                new_population.append(Individual(off_chrom))

        self._evaluate_individuals(new_population)
        self._apply_elitism(new_population, fittest_individual)
        return new_population

    def run(self, seed: int) -> None:
        """
        Executes the full GAwLL evolutionary process.

        Args:
            seed: Random seed for reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)

        # Component Initialization
        self._initialize_population()
        self.evig = eVIG(self.chrom_size)
        self.importance = Importance(self.chrom_size)
        self.population_analyzer = PopulationAnalyzer()
        self.selection = Selection()
        self.crossover = Crossover()
        self.mutation = Mutation()
        self.mutation_ll = LinkageLearning(self.evig, self.importance,
                                           self.perf_batch,self._calculate_fitness_value)

        last_improvement_gen = 0
        best_fitness_track = 0.0

        for generation in range(self.max_generations):
            fittest = self.population_analyzer.get_fittest_individual(self.population)

            # Stagnation tracking
            if fittest.fitness > best_fitness_track + self.EPSILON:
                best_fitness_track = fittest.fitness
                last_improvement_gen = generation

            # Tau-Reset logic: Triggered if no improvement is found
            if generation - last_improvement_gen > self.tau_reset_generations:
                print(f"    [RESTART] Generation {generation}: Perturbing population...")
                reset_pop = []
                for _ in range(self.population_size):
                    c = fittest.chromosome.copy()
                    self.mutation.bit_flip_mutation(c, 0.30)  # High mutation for reset
                    reset_pop.append(Individual(c))

                self._evaluate_individuals(reset_pop)
                self.population = reset_pop
                last_improvement_gen = generation

            # Evolution Step
            if self.linkage_learning:
                self.population = self._generation_ll(fittest)
            else:
                self.population = self._generation(fittest)

            gc.collect()

        self.final_fittest = self.population_analyzer.get_fittest_individual(self.population)

    def get_feature_selection_results(self) -> Tuple[np.ndarray, float, float]:
        """
        Extracts the results from the best individual found.

        Returns:
            Tuple[np.ndarray, float, float]: (Best Chromosome, Fitness, Raw Performance)
        """
        if not self.final_fittest:
            raise RuntimeError("GAwLL must be executed via .run() before fetching results.")

        return (
            self.final_fittest.chromosome.copy(),
            float(self.final_fittest.fitness),
            float(self.final_fittest.f_perf)
        )

    def cleanup(self) -> None:
        """
        Releases system resources and breaks circular references.
        Critical for avoiding OOM (Out Of Memory) in long-running experiments.
        """
        self.perf_batch = None
        self.mutation_ll = None
        self.selection = None
        self.crossover = None
        self.mutation = None

        self.cache.clear()
        self.population = []
        self.evig = None
        self.importance = None

        gc.collect()