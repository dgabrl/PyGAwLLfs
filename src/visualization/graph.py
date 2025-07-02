import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Graph:
    def __init__(self, output_dir = 'results'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

        self.nx_graph = nx.Graph()

    def graph(self, interaction_matrix,variables, model_name):
        num_nodes = len(variables)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if interaction_matrix[i][j] != 0:
                    self.nx_graph.add_edge(variables[i], variables[j], weight=interaction_matrix[i][j])
        circ = nx.circular_layout(self.nx_graph)

        edge_weights = [d['weight'] for _, _, d in self.nx_graph.edges(data=True)]
        edge_widths = [weight / max(edge_weights) * 5 for weight in edge_weights]

        nx.draw_networkx(self.nx_graph, circ, font_size=6, width=edge_widths)
        plt.savefig(os.path.join(self.output_dir, f"{model_name} Graph.png"))
        plt.close()
        return self.nx_graph

    def reduced_graph(self, graph,model_name):
        def vis_transf_v_int(x):
            a = 0.0
            b = 5.0
            y = a + b * np.array(x)
            return y

        def outlier_upper_range(datacolumn):
            q05, q95 = np.percentile(datacolumn, [5, 95])
            upper_bound = q95
            return upper_bound

        edges = graph.edges()
        weights = [graph[u][v]['weight'] for u, v in edges]
        weights_reduced = np.array(weights)
        weights_reduced = np.delete(weights_reduced, np.where(weights_reduced == 0.0))

        if weights_reduced.size > 0:
            upper_range = outlier_upper_range(weights_reduced)
            elarge = [(u, v) for (u, v, w) in graph.edges(data=True) if w["weight"] >= upper_range]
            weights_elarge = [graph[u][v]['weight'] for u, v in elarge]
            max_weight_elarge = max(weights_elarge)
            w_new = [weight / max_weight_elarge * 3 for weight in weights_elarge]
            pos = nx.circular_layout(graph)
            nx.draw(graph, pos=pos, with_labels=False, font_weight='bold', width=w_new,
                    edgelist=elarge, edge_color = 'black')
            labels = nx.draw_networkx_labels(graph, pos, font_size=10)
            for label in labels.values():
                label.set_rotation(45)
            plt.savefig(os.path.join(self.output_dir, f"{model_name} Reduced Graph.png"))
            plt.close()
        else:
            print(f"Error: there is no edges with non-zero weights for {model_name}")
