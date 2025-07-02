import os

class eVIGEdge:
    def __init__(self, vertex, weight):
        self.vertex = vertex
        self.weight = weight


class eVIGNode:
    def __init__(self):
        self.edges = []

    def add_or_replace_edge(self, vertex, weight):
        found = False

        for edge in self.edges:
            if edge.vertex == vertex:
                found = True
                edge.weight = weight

        if not found:
            self.edges.append(eVIGEdge(vertex, weight))


class eVIG:
    def __init__(self, degree):
        self.degree = degree

        self._max_edge_weight = [[0 for _ in range(degree)] for _ in range(degree)]
        self._nodes = [eVIGNode() for _ in range(degree)]

    def add_edge(self, a, b, w):
        self._max_edge_weight[a][b] = max(self._max_edge_weight[a][b], w)

        self._max_edge_weight[b][a] = self._max_edge_weight[a][b]

        self._nodes[a].add_or_replace_edge(b, w)
        self._nodes[b].add_or_replace_edge(a, w)

    def interaction_matrix(self):
        interaction_matrix = [[0.0 for _ in range(self.degree)] for _ in range(self.degree)]
        for i in range(self.degree):
            for j in range(i):
                if self._max_edge_weight[i][j] != 0:
                    interaction_matrix[i][j] = self._max_edge_weight[i][j]
                    interaction_matrix[j][i] = interaction_matrix[i][j]
        return interaction_matrix