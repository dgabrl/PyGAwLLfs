"""
Genetic Operators Module.

Provides core mechanisms for evolutionary computation, including 
tournament selection, uniform crossover, and vectorized bit-flip mutation.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Any


class Selection:
    """Provides selection mechanisms for identifying parents for the next generation."""

    @staticmethod
    def tournament_selection(population: List[Any], tournament_size: int = 3) -> Optional[Any]:
        """
        Selects the fittest individual from a random subset of the population.

        Args:
            population: List of individual objects with a 'fitness' attribute.
            tournament_size: Number of individuals to compete in the tournament.

        Returns:
            The winning individual (highest fitness). Returns None if population is empty.
        """
        if not population:
            return None

        participants = random.sample(population, min(tournament_size, len(population)))

        # Fallback to -1.0 if fitness is None to avoid comparison errors
        return max(participants, key=lambda ind: ind.fitness if ind.fitness is not None else -1.0)


class Crossover:
    """Provides genetic recombination operators to create offspring."""

    @staticmethod
    def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs uniform crossover using a random binary mask for gene exchange.

        Args:
            parent1: First parent chromosome (binary array).
            parent2: Second parent chromosome (binary array).

        Returns:
            A tuple (offspring1, offspring2) of mutated chromosomes.
        """
        # Generates a boolean mask (50% probability for each gene exchange)
        mask = np.random.random(len(parent1)) < 0.5

        # np.where(condition, x, y) returns x if condition is True, else y.
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(~mask, parent1, parent2)

        return offspring1, offspring2


class Mutation:
    """Provides stochastic operators to maintain genetic diversity."""

    @staticmethod
    def bit_flip_mutation(chromosome: np.ndarray, mutation_probability: float) -> np.ndarray:
        """
        Applies bit-flip mutation by XORing the chromosome with a random mask.

        Args:
            chromosome: The binary chromosome to mutate.
            mutation_probability: Probability (0 to 1) of flipping each bit.

        Returns:
            The mutated chromosome as a boolean/int8 array.
        """
        # Generate a boolean mask where True means 'flip this bit'
        mutation_mask = np.random.random(len(chromosome)) < mutation_probability

        # Logical XOR handles bit flipping (0^0=0, 0^1=1, 1^0=1, 1^1=0)
        # We ensure it returns a boolean array for consistency with Individual class
        return np.logical_xor(chromosome, mutation_mask).astype(bool)