from __future__ import annotations
from tinyhnsw.index import Index

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

        self.graphs = []
        self.entry_point = None
        self.L = -1
        self.current_ix = 0

    def add(self, vectors: numpy.ndarray) -> None:
        for vector in vectors:
            self.add_one(vector)

    def add_one(self, vector: numpy.ndarray) -> None:
        level = self.assign_level()

        # for current_level in range(self.L, -1, -1):
        #     if current_level > level:
        #         distance, point = self.search(vector, k=1)

        if level > self.L:
            for i in range(self.L+1, level+1):
                G = networkx.Graph()
                G.add_node(self.current_ix)
                self.entry_point = self.current_ix
                #self.graphs.append(G)
                self.graphs.append(networkx.complete_graph(5-i))
            self.L = level
        
        self.current_ix += 1



    def search(
        self, query: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        return ()

    def assign_level(self) -> int:
        return math.floor(-math.log(random.random()) * self.m_L)


def visualize_hnsw_index(index: HNSWIndex):
    fig, axs = plt.subplots(1, len(index.graphs), figsize=(len(index.graphs)*5, 5))
    
    layout = networkx.spring_layout(index.graphs[0])
    for i, graph in enumerate(index.graphs):
        graph_layout = {k: v for k, v in layout.items() if k in graph}
        networkx.draw(graph, graph_layout, ax=axs[i])
        axs[i].set_title(f'Layer {i}')
    
    plt.show()


if __name__ == "__main__":
    index = HNSWIndex(100, 3)
    vectors = numpy.random.randn(10, 5)
    index.add(vectors)

    visualize_hnsw_index(index)


