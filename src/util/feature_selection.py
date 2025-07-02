import os

class FrequencyFeatureSelection:
    def __init__(self, num_features, feature_names, max_generations, output_dir='results'):
        self.num_features = num_features
        self.feature_names = feature_names
        self.max_generations = max_generations
        self.generation_counts = {name: [0] * max_generations for name in feature_names}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def update_counts(self, population, generation_index):
        for individual in population:
            for i, gene in enumerate(individual.chromosome):
                if gene == 1:
                    self.generation_counts[self.feature_names[i]][generation_index] += 1

    def get_total_counts(self):
        return {name: sum(counts) for name, counts in self.generation_counts.items()}

    def save_variables_frequency(self, model_name):
        total_counts = self.get_total_counts()
        sorted_counts = sorted(total_counts.items(), key=lambda item: item[1], reverse=True)

        output_filename = f"{model_name}-Frequency Variables.txt"
        filepath = os.path.join(self.output_dir, output_filename)
        with open(filepath, "w") as f:
            f.write(f"Frequency of variables (GAwLL - {model_name}):\n")
            for feature, count in sorted_counts:
                f.write(f"- {feature} = {count} times (across {self.max_generations} generations)\n")

            f.write("\nFrequency per generation:\n")
            for feature, count in sorted_counts:
                f.write(f"- {feature}: {self.generation_counts[feature]}\n")

        print(f"File '{output_filename}' saved.")