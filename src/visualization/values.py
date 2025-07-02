import csv
import os

class SaveValues:
    def __init__(self, output_dir = 'results'):
        self.output_dir = output_dir

    def save_importances(self,importance_vector,importance_name):
        output_filename = f'{importance_name}.csv'
        filepath = os.path.join(self.output_dir,output_filename)

        with open(filepath,'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Importance'])
            for importance in importance_vector:
                writer.writerow([f'{importance:.10f}'])
        print(f'File {output_filename} saved')

    def save_interaction_matrix(self,interaction_matrix, name):
        output_filename = f'{name}.csv'
        filepath = os.path.join(self.output_dir,output_filename)

        with open(filepath, 'w', newline ='') as csvfile:
            writer = csv.writer(csvfile)
            for row in interaction_matrix:
               writer.writerow([f'{weight: .10f}' for weight in row])
        print(f'File {output_filename} saved')

    def save_top_importances(self,gawll_imp,variables, model_name, n):
        variable_importances = list(enumerate(gawll_imp))
        sorted_importances = sorted(variable_importances, key=lambda item: item[1], reverse=True)
        top_importances_index = [i for i, _ in sorted_importances[:n]]
        top_importance_variables = [variables[i] for i in top_importances_index]
        top_importances_values = [importance for _, importance in sorted_importances[:n]]

        output_filename = f"{model_name}-GAwLL top importances.txt"
        filepath = os.path.join(self.output_dir, output_filename)
        with open(filepath, "w") as f:
            f.write(f"Top attributes with highest importance (GAwLL - {model_name}):\n")
            for variable, importance in zip(top_importance_variables, top_importances_values):
                f.write(f"- {variable}: {importance:.4f}\n")
        print(f"File '{output_filename}' saved.")

    def save_top_interactions(self, interaction_matrix, variables, model_name, n):
        weighted_edges = []
        for i in range (len(variables)):
            for j in range (i+1, len(variables)):
                if interaction_matrix[i][j] != 0:
                    weighted_edges.append(((variables[i],variables[j]),interaction_matrix[i][j]))

        sorted_interactions = sorted(weighted_edges, key = lambda item: item[1], reverse = True)
        top_interactions = sorted_interactions[:n]

        output_filename = f"{model_name} top {n} interactions.txt"
        filepath = os.path.join(self.output_dir, output_filename)
        with open(filepath, "w") as f:
            f.write(f"\nTop {n} attributes with highest interaction (GAwLL - {model_name}):\n")
            for (var1, var2), weight in top_interactions[:n]:
                f.write(f"- {var1} -- {var2}: {weight:.4f}\n")
        print(f"File '{output_filename}' saved.")