import random
import numpy as np
from cachetools import LRUCache
from src.core.evig import eVIG
from src.core.importance import Importance
from src.util.statistic import Statistics, PopulationAnalyzer
from src.util.feature_selection import FrequencyFeatureSelection
from src.core.ga_operators import Selection, Crossover, Mutation
from src.core.linkage_learning_mutation import LinkageLearning

class Individual:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness


class GAwLL:
    POPULATION_SIZE = 50
    CROSSOVER_RATE = 0.4

    TAU_RESET_GENERATIONS = 50

    EPSILON = 1.0e-10

    FITNESS_FUNCTION_CACHE_LIMIT = 10e4

    def __init__(
            self,
            *,
            fitness_function,
            chrom_size,
            mutation_probability,
            max_generations,
            linkage_learning=True,
            selection=Selection(),
            crossover=Crossover(),
            mutation=Mutation(),
            mutation_ll=LinkageLearning,
    ):
        cache = LRUCache(maxsize=self.FITNESS_FUNCTION_CACHE_LIMIT)

        self.fitness_function = lambda chromosome: cache.get(
            tuple(chromosome),
            cache.setdefault(tuple(chromosome), fitness_function(chromosome)),
        )

        self.chrom_size = chrom_size
        self.mutation_probability = mutation_probability
        self.max_generations = max_generations

        self.linkage_learning = linkage_learning

        self.statistics = Statistics()
        self.population_analyzer = PopulationAnalyzer()

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.mutation_ll = mutation_ll(self)

        self.variables_frequency = FrequencyFeatureSelection(
            num_features=self.chrom_size,
            feature_names=[f"x{i}" for i in range(self.chrom_size)],
            max_generations=self.max_generations,
        )

        self.population = None
        self.evig = None
        self.importance = None

    def initialize_population(self, fittest_individual=None):
        self.population = []

        for _ in range(self.POPULATION_SIZE):
            chromosome = np.random.randint(0, 2, size=self.chrom_size).astype(bool)
            self.population.append(self.generate_individual(chromosome))

        if fittest_individual is not None:
            self.population[0] = fittest_individual

    def generate_individual(self, chromosome):
        fitness = self.fitness_function(chromosome)
        return Individual(chromosome, fitness)

    def generation(self, fittest_individual):
        new_population = [self.generate_individual(fittest_individual.chromosome.copy())]

        while len(new_population) < len(self.population):
            offspring1 = parent1 = self.selection.tournament_selection(self.population)

            if len(new_population) < len(self.population) - 1:
                offspring2 = parent2 = self.selection.tournament_selection(self.population)

                if random.random() < self.CROSSOVER_RATE:
                    offspring1, offspring2 = self.crossover.uniform_crossover(parent1, parent2)

                self.mutation.bit_flip_mutation(offspring1, self.mutation_probability)
                self.mutation.bit_flip_mutation(offspring2, self.mutation_probability)

                new_population.append(self.generate_individual(offspring1))
                new_population.append(self.generate_individual(offspring2))
            else:
                self.mutation.bit_flip_mutation(offspring1, self.mutation_probability)
                new_population.append(self.generate_individual(offspring1))

        return new_population

    def generation_ll(self, fittest_individual):
        new_population = [self.generate_individual(fittest_individual.chromosome.copy())]

        while len(new_population) < len(self.population):

            offspring1 = parent1 = self.selection.tournament_selection(self.population)

            if len(new_population) < int(self.POPULATION_SIZE * self.CROSSOVER_RATE):
                offspring2 = parent2 = self.selection.tournament_selection(self.population)

                if random.random() < self.CROSSOVER_RATE:
                    offspring1, offspring2 = self.crossover.uniform_crossover(parent1, parent2)

                self.mutation.bit_flip_mutation(offspring1, self.mutation_probability)
                self.mutation.bit_flip_mutation(offspring2, self.mutation_probability)

                new_population.append(self.generate_individual(offspring1))
                new_population.append(self.generate_individual(offspring2))

            elif len(new_population) < len(self.population) - 2:
                offspring1, offspring2, offspring3 = self.mutation_ll.mutation_ll(parent1)

                new_population.append(offspring1)
                new_population.append(offspring2)
                new_population.append(offspring3)

            else:
                self.mutation.bit_flip_mutation(offspring1, self.mutation_probability)
                new_population.append(self.generate_individual(offspring1))

        return new_population

    def run(self, seed):
        random.seed(seed)

        self.initialize_population()

        self.evig = eVIG(self.chrom_size)
        self.importance = Importance(self.chrom_size)

        self.statistics.initial_bfi_per_run.append(self.population_analyzer.get_fittest_individual(self.population))
        self.statistics.initial_mean_fitness_per_run.append(self.population_analyzer.get_average_fitness(self.population))

        last_change_generation = 0
        last_change_highest_fitness = 0

        for generation in range(self.max_generations):
            self.variables_frequency.update_counts(self.population, generation)

            fittest_individual = self.population_analyzer.get_fittest_individual(self.population)

            if fittest_individual.fitness > last_change_highest_fitness + self.EPSILON:
                last_change_generation = generation
                last_change_highest_fitness = fittest_individual.fitness

            if generation - last_change_generation > self.TAU_RESET_GENERATIONS:
                last_change_generation = generation
                self.initialize_population(fittest_individual=fittest_individual)

            if self.linkage_learning:
                self.population = self.generation_ll(fittest_individual)
            else:
                self.population = self.generation(fittest_individual)

        self.statistics.best_individual_per_run.append(self.population_analyzer.get_fittest_individual(self.population))
        self.statistics.evig_per_run.append(self.evig)

    def best_individual(self,model_name):
        n_individuals = len(self.statistics.best_individual_per_run)
        best_individual = self.statistics.best_individual_per_run[n_individuals - 1]
        self.statistics.save_best_individual(model_name,best_individual.chromosome,best_individual.fitness)

    def feature_selection(self,model_name):
        self.variables_frequency.save_variables_frequency(model_name)