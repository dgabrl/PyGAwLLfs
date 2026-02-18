import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
from typing import List, Tuple

class Graph:
    """
    Visualizes the empirical Variable Interaction Graph (eVIG) using a circular layout.
    Ideal for identifying 'hubs' of interaction within the feature space.
    """

    def __init__(self):
        # Directory handling
        self.output_path = Path('results')
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _prepare_nx_graph(self, matrix: np.ndarray, variables: List[str]) -> nx.Graph:
        """Converts an adjacency matrix to a NetworkX graph structure (undirected)."""
        G = nx.Graph()
        G.add_nodes_from(variables)

        matrix = np.asanyarray(matrix)
        rows, cols = np.where(np.abs(matrix) > 1e-10)

        for i, j in zip(rows, cols):
            if i < j:  # Upper triangle only to avoid duplicate edges
                G.add_edge(variables[i], variables[j], weight=matrix[i, j])
        return G

    def _get_vimp_styles(self, vimp_vector: np.ndarray
                         ) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
        """Maps the Variable Importance Vector (vImp) to node sizes and colors."""

        # 1. Robust Min-Max Normalization
        min_val, max_val = np.nanmin(vimp_vector), np.nanmax(vimp_vector)

        if min_val == max_val:
            norm_values = np.zeros_like(vimp_vector)
            node_sizes = np.full_like(vimp_vector, 500.0)
        else:
            norm_values = (vimp_vector - min_val) / (max_val - min_val)
            # Size scale for Matplotlib (300 to 1500 units)
            node_sizes = 300 + (1200 * norm_values)

        # 2. Palette Selection
        colors = ['#F9FBE7', '#CDDC39', '#827717']

        cmap = mcolors.LinearSegmentedColormap.from_list(f"vimp_gawll", colors)
        node_colors = [cmap(val) for val in norm_values]

        return node_sizes, node_colors

    def _get_vint_styles(self,
                         edge_weights: np.ndarray
                         ) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
        """Maps Interaction Matrix weights (vInt) to edge widths and colors."""

        min_w, max_w = np.nanmin(edge_weights), np.nanmax(edge_weights)

        if min_w == max_w:
            edge_widths = np.full_like(edge_weights, 2.0)
            norm_w = np.zeros_like(edge_weights)
        else:
            norm_w = (edge_weights - min_w) / (max_w - min_w)
            # Width scale (1.0 to 6.0 pixels)
            edge_widths = 1 + (5 * norm_w)

        # 2. Palette Selection for Edges
        colors = ['#90EE90', '#006400']

        cmap = mcolors.LinearSegmentedColormap.from_list(f"vint_gawll", colors)
        edge_colors = [cmap(val) for val in norm_w]

        return edge_widths, edge_colors

    def _add_circular_labels(self, pos: dict, nodes: list, fontsize: int = 10):
        """Helper to place labels outside the circular layout with correct rotation."""
        for node in nodes:
            x, y = pos[node]
            angle = np.degrees(np.arctan2(y, x))
            ha = "right" if (angle > 90 or angle < -90) else "left"
            rot = angle + 180 if (angle > 90 or angle < -90) else angle
            plt.text(1.1 * x, 1.1 * y, str(node), ha=ha, va="center",
                     fontsize=fontsize, rotation=rot, rotation_mode="anchor")
        
    def save_all_visualizations(self, 
                                vimp_vector: np.ndarray,
                                vint_matrix: np.ndarray,
                                variables: List[str], 
                                model_name: str):
        """Generates all eVIG visual perspectives using a consistent circular layout."""
        G = self._prepare_nx_graph(vint_matrix, variables)

        # Circular layout based on the list of variables to keep consistency
        pos = nx.circular_layout(variables)

        # Save graph and reduced graph
        if G.number_of_edges() > 0:
            self._save_graph(G, pos, variables, vimp_vector, f"{model_name}_GAwLL_Interactions")
            self._save_reduced_graph(G, pos, variables, vimp_vector, f"{model_name}_GAwLL_Reduced_Interactions")

    def _save_graph(self, G: nx.Graph, pos: dict, variables: List[str],
                    importances: np.ndarray, filename: str):
        """
        Plots a styled graph with dual colorbars for XAI interpretation.

        This method integrates Variable Importance (nodes) and
        Interaction Strength (edges) into a single visual perspective.
        """
        plt.figure(figsize=(14, 12))

        # 1. Styling Engine: Map VImp and VInt to visual attributes
        node_sizes, node_colors = self._get_vimp_styles(importances)

        if G.number_of_edges() > 0:
            weights = np.array([abs(d['weight']) for _, _, d in G.edges(data=True)])
            edge_widths, edge_colors = self._get_vint_styles(weights)
        else:
            weights, edge_widths, edge_colors = np.array([]), [], []

        # 2. Map styles to current graph nodes
        nodes_in_g = list(G.nodes())
        var_to_idx = {var: i for i, var in enumerate(variables)}
        current_node_sizes = [node_sizes[var_to_idx[n]] for n in nodes_in_g]
        current_node_colors = [node_colors[var_to_idx[n]] for n in nodes_in_g]

        # 3. Draw Elements
        nx.draw_networkx_nodes(G, pos,
                               node_size=current_node_sizes,
                               node_color=current_node_colors,
                               edgecolors='black',
                               linewidths=1.2)

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6)

        # 4. Labeling and Layout adjustments
        self._add_circular_labels(pos, G.nodes())

        # Extend limits to prevent label clipping
        plt.xlim(-1.4, 1.4)
        plt.ylim(-1.4, 1.4)

        plt.axis('off')
        plt.savefig(self.output_path / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close('all')

    def _save_reduced_graph(self, G: nx.Graph, pos: dict, variables: List[str],
                            importances: np.ndarray, filename: str):
        """Filters top 5% of edges for critical linkage analysis."""
        weights = [abs(d['weight']) for _, _, d in G.edges(data=True)]
        if not weights: return

        threshold = np.percentile(weights, 95)
        # Create a subgraph with only strong edges
        G_reduced = nx.Graph()
        G_reduced.add_nodes_from(variables)
        strong_edges = [(u, v, d) for u, v, d in G.edges(data=True) if abs(d['weight']) >= threshold]
        G_reduced.add_edges_from(strong_edges)

        self._save_graph(G_reduced, pos, variables, importances, filename)