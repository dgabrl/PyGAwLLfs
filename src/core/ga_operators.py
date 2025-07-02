import random
import numpy as np

class Selection:
    @staticmethod
    def tournament_selection(population):
        TOURNAMENT_SIZE = 3

        population_size = len(population)
        if not population:
            return None

        participants = random.sample(population, min(TOURNAMENT_SIZE, population_size))
        winner = max(participants, key=lambda individual: individual.fitness)
        return winner.chromosome


class Crossover:
    @staticmethod
    def uniform_crossover(parent1, parent2):
        chromosome1 = []
        chromosome2 = []
        chrom_size = len(parent1)

        for gene in range(chrom_size):
            if parent1[gene] == parent2[gene]:
                perform_crossover = False
            else:
                perform_crossover = random.choice([False, True])

            if perform_crossover:
                chromosome1.append(parent2[gene])
                chromosome2.append(parent1[gene])
            else:
                chromosome1.append(parent1[gene])
                chromosome2.append(parent2[gene])

        return chromosome1, chromosome2


class Mutation:
    @staticmethod
    def bit_flip_mutation(offspring, mutation_probability):
        mutation_mask = np.random.rand(len(offspring)) < mutation_probability
        offspring = np.logical_xor(offspring, mutation_mask)

        return offspring