from __future__ import annotations
from tinyhnsw.index import Index
from tinyhnsw.knn import FullNNIndex, cosine_similarity
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

        self.graphs = []
        self.entry_point = None
        self.L = -1
        self.current_ix = 0

        self.temp_index = FullNNIndex(self.d)

    def add(self, vectors: numpy.ndarray) -> None:
        for vector in vectors:
            self.temp_index.add(numpy.expand_dims(vector, axis=0))
            self.add_one(vector)

    def distance(self, q, v):
        c = cosine_similarity(numpy.expand_dims(q, axis=0), numpy.expand_dims(v, axis=0))
        return c

    def add_one(self, vector: numpy.ndarray) -> None:
        level = self.assign_level()

        if level > self.L:
            for i in range(self.L + 1, level + 1):
                G = networkx.Graph()
                self.graphs.append(G)
            self.L = level

        # Initialize the node in all layers up to its level
        for i in range(level + 1):
            if self.current_ix not in self.graphs[i]:
                self.graphs[i].add_node(self.current_ix)

        # Set the entry point if this is the highest level node so far
        if level == self.L:
            self.entry_point = self.current_ix

        ep = self.entry_point

        # Start from the top layer and work down to layer 0
        for lc in range(self.L, -1, -1):
            if lc > level:
                continue

            # Search for the ef closest elements in the current layer
            W = self.search_layer(vector, ep, ef=self.ef_construction, layer=lc)
            neighbors = self.select_neighbors(vector, W, self.M, lc)

            # Add edges between the current node and its neighbors in the graph
            for neighbor in neighbors:
                self.graphs[lc].add_edge(self.current_ix, neighbor)

                # Check if the neighbor has too many connections and prune if necessary
                eConn = list(self.graphs[lc][neighbor])
                if len(eConn) > self.M_max:
                    if lc == 0:
                        Mmax = self.M_max0
                    else:
                        Mmax = self.M_max
                    eNewConn = self.select_neighbors(self.temp_index.vectors[neighbor], eConn, Mmax, lc)
                    # Update the neighbor's connections
                    self.graphs[lc].remove_node(neighbor)
                    self.graphs[lc].add_node(neighbor)
                    for new_neighbor in eNewConn:
                        self.graphs[lc].add_edge(neighbor, new_neighbor)

            # Update the entry point for the next lower layer
            if len(W) > 0:
                ep = min(W, key=lambda x: self.distance(vector, self.temp_index.vectors[x]))

        self.current_ix += 1

    def search_layer(
        self, vector: numpy.ndarray, entry_point: int, layer: int, ef: int = 1
    ) -> tuple[list[float], list[int]]:
        visited = set([self.entry_point])
        candidates = {
            self.entry_point: self.distance(vector, self.temp_index.vectors[entry_point].squeeze())
        }
        neighbors = {
            self.entry_point: self.distance(vector, self.temp_index.vectors[entry_point].squeeze())
        }

        while len(candidates) > 0:
            c, c_distance = min(candidates.items(), key=lambda x: x[1])
            f, f_distance = max(neighbors.items(), key=lambda x: x[1])

            if c_distance > f_distance:
                break

            for e in self.graphs[layer][c]:
                if e not in visited:
                    visited.add(e)
                    e_distance = self.distance(vector, self.temp_index.vectors[e])

                    f, f_distance = max(neighbors.items(), key=lambda x: x[1])
                    if e_distance < f_distance or len(neighbors) < ef:
                        candidates[e] = e_distance
                        neighbors[e] = e_distance

                        if len(neighbors) > ef:
                            furthest = max(neighbors, key=neighbors.get)
                            del neighbors[furthest]

            del candidates[c]
        print(neighbors)
        return list(neighbors.keys())

    def select_neighbors(
        self,
        q: numpy.ndarray,
        C: set,
        M: int,
        lc: int,
        extendCandidates: bool = False,
        keepPrunedConnections: bool = False,
    ) -> set:
        R = set()
        W = C.copy()

        if extendCandidates:
            for e in C:
                for e_adj in self.graphs[lc][e]:
                    if e_adj not in W:
                        W.add(e_adj)

        W_d = set()

        while len(W) > 0 and len(R) < M:
            e = min(W, key=lambda x: self.distance(q, self.temp_index.vectors[x]))
            W.remove(e)

            if len(R) == 0 or self.distance(q, self.temp_index.vectors[e]) < max(
                [self.distance(q, self.temp_index.vectors[r]) for r in R]
            ):
                R.add(e)
            else:
                W_d.add(e)

        if keepPrunedConnections:
            while len(W_d) > 0 and len(R) < M:
                e = min(W_d, key=lambda x: self.distance(q, self.temp_index.vectors[x]))
                W_d.remove(e)
                R.add(e)

        return R

    def assign_level(self) -> int:
        return math.floor(-math.log(random.random()) * self.m_L)


def visualize_hnsw_index(index: HNSWIndex):
    """
    Use this to visualize the different layers of HNSW graphs. The nodes
    maintain consistent locations between layers, and the layers are
    plotted next to each other.
    """
    _, axs = plt.subplots(1, len(index.graphs), figsize=(len(index.graphs) * 5, 5))

    layout = networkx.spring_layout(index.graphs[0])
    node_color = [
        "r" if index.entry_point == node else "c" for node, _ in layout.items()
    ]

    for i, graph in enumerate(index.graphs):
        graph_layout = {k: v for k, v in layout.items() if k in graph}
        graph_node_color = [node_color[k] for k, _ in graph_layout.items()]
        networkx.draw(
            graph, graph_layout, ax=axs[i], node_size=25, node_color=graph_node_color
        )
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
