import os

class Statistics:
    def __init__(self, output_dir='results'):
        self.best_individual_per_run = []
        self.evig_per_run = []

        self.initial_bfi_per_run = []
        self.initial_mean_fitness_per_run = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def save_best_individual(self, model_name, best_chrom, best_fitness):

        best_chromosome_binary = [1 if gene else 0 for gene in best_chrom]

        output_filename = f"{model_name}-Best Chromosome and Fitness.txt"
        filepath = os.path.join(self.output_dir, output_filename)
        with open(filepath, "w") as f:
            f.write(f"\nBest individual (GAwLL - {model_name}):\n")
            f.write(f"- Chromosome: {best_chromosome_binary}\n")
            f.write(f"- Fitness: {best_fitness}\n")
        print(f"File '{output_filename}' saved.")


class PopulationAnalyzer:
    @staticmethod
    def get_average_fitness(population):
        if not population:
            return 0
        return sum(individual.fitness for individual in population) / len(population)

    @staticmethod
    def get_fittest_individual(population):
        if not population:
            return None
        return max(population, key=lambda individual: individual.fitness)