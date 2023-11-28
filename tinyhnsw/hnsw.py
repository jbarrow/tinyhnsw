from __future__ import annotations
from tinyhnsw.index import Index
from tinyhnsw.knn import FullNNIndex
from matplotlib.patches import Rectangle

import numpy
import math
import random
import networkx
import matplotlib.pyplot as plt


class HNSWIndex(Index):
    def __init__(
        self, d: int, M: int, ef_construction: int = 32, ef_search: int = 32
    ) -> None:
        super().__init__(d)

        self.M = M
        self.M_max = M
        self.M_max0 = 2 * M

        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # m_L
        self.m_L = 1.0 / math.log(M)

        self.graphs = [networkx.Graph()]
        self.entry_point = None
        self.L = 0
        self.current_ix = 0

        self.temp_index = FullNNIndex(self.d)

    def add(self, vectors: numpy.ndarray) -> None:
        for vector in vectors:
            self.add_one(vector)
            self.temp_index.add(numpy.expand_dims(vector, axis=0))

    def add_one(self, vector: numpy.ndarray) -> None:
        level = self.assign_level()

        if level > self.L:
            for i in range(self.L + 1, level):
                G = networkx.Graph()
                G.add_node(self.current_ix)
                self.entry_point = self.current_ix
                self.graphs.append(G)

        for i in range(min(level, self.L) + 1):
            self.graphs[i].add_node(self.current_ix)
            self.set_edges(self.current_ix, vector, i)
            neighbors = list(self.graphs[i][self.current_ix])

            for neighbor in neighbors:
                if len(self.graphs[i][neighbor]) > self.M_max:
                    self.set_edges(neighbor, self.temp_index.vectors[neighbor], i)

        self.L = level
        self.current_ix += 1

    def set_edges(self, node, vector, layer):
        # remove all connected edges
        self.graphs[layer].remove_node(node)
        self.graphs[layer].add_node(node)

        _, neighbors = self.search(numpy.expand_dims(vector, axis=0), k=self.M)
        self.graphs[layer].add_edges_from([(self.current_ix, n) for n in neighbors[0]])

    def search(
        self, query: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        return self.temp_index.search(query, k)

    def assign_level(self) -> int:
        level = min(3, math.floor(-math.log(random.random()) * self.m_L))
        print(level)
        return level


def visualize_hnsw_index(index: HNSWIndex):
    """
    Use this to visualize the different layers of HNSW graphs. The nodes
    maintain consistent locations between layers, and the layers are
    plotted next to each other.
    """
    _, axs = plt.subplots(1, len(index.graphs), figsize=(len(index.graphs) * 5, 5))

    layout = networkx.spring_layout(index.graphs[0])

    for i, graph in enumerate(index.graphs):
        graph_layout = {k: v for k, v in layout.items() if k in graph}
        networkx.draw(graph, graph_layout, ax=axs[i])
        axs[i].set_title(f"Layer {i}")

        # Add a border around each subplot
        rect = Rectangle(
            (0, 0),
            1,
            1,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            transform=axs[i].transAxes,
        )
        axs[i].add_patch(rect)

    plt.show()


if __name__ == "__main__":
    index = HNSWIndex(10, 3)
    vectors = numpy.random.randn(10, 10)
    index.add(vectors)

    visualize_hnsw_index(index)
