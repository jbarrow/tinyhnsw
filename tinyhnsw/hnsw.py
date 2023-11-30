from __future__ import annotations
from tinyhnsw.index import Index
from tinyhnsw.knn import cosine_similarity
from tinyhnsw.utils import load_sift, evaluate
from dataclasses import dataclass
from heapq import nlargest, nsmallest, heappop, heappush
from tqdm import tqdm

import numpy
import math
import random
import networkx
import matplotlib.pyplot as plt


@dataclass
class HNSWConfig:
    M: int
    M_max: int
    M_max0: int
    m_L: float
    ef_construction: int


DEFAULT_CONFIG = HNSWConfig(
    M=3, M_max=3, M_max0=6, m_L=(1.0 / math.log(3)), ef_construction=32
)


class HNSWIndex(Index):
    def __init__(self, d: int, config: HNSWConfig = DEFAULT_CONFIG) -> None:
        super().__init__(d)

        self.config = config
        self.vectors = None

        self.layers = [HNSWLayer(self, 0)]
        self.ep = 0
        self.L = 0
        self.ix = 0

    def assign_level(self) -> int:
        return math.floor(-math.log(random.random()) * self.config.m_L)

    def add(self, vectors: numpy.ndarray) -> None:
        assert vectors.shape[1] == self.d

        if self.vectors is None:
            self.vectors = vectors
            self.is_trained = True
        else:
            self.vectors = numpy.append(self.vectors, vectors, axis=0)

        self.ntotal = self.vectors.shape[0]

        for vector in tqdm(vectors):
            self.insert_into_graph(vector)

    def insert_into_graph(self, q: numpy.ndarray):
        l = self.assign_level()
        L = self.L
        ep = self.ep
        ix = self.ix

        for layer in range(L, l, -1):
            _, W = self.layers[layer].search(q, ep, ef=1)
            ep = W[0]

        for layer in range(min(L, l), -1, -1):
            self.layers[layer].insert(q, ix, ep)

        if l > self.L:
            for l_new in range(L + 1, l + 1):
                self.layers.append(HNSWLayer(self, l_new, self.ix))
            self.L = l
            self.ep = ix

        self.ix += 1

    def distance(self, q: numpy.ndarray, v: numpy.ndarray) -> numpy.ndarray:
        if len(q.shape) == 1:
            q = numpy.expand_dims(q, axis=0)

        if len(v.shape) == 1:
            v = numpy.expand_dims(v, axis=0)

        return 1.0 - cosine_similarity(q, v)

    def search(self, q: numpy.ndarray, k: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        ep = self.ep
        for lc in range(self.L, 0, -1):
            ep = self.layers[lc].search(q, ep, 1)[1][0]

        return self.layers[0].search(q, ep, k)


class HNSWLayer:
    def __init__(self, index: HNSWIndex, lc: int, ep: int | None = None) -> None:
        self.G = networkx.Graph()
        self.index = index
        self.config = self.index.config

        if ep is not None:
            self.G.add_node(ep)

        if lc == 0:
            self.M_max = self.config.M_max0
        else:
            self.M_max = self.config.M_max

    def distance_to_node(self, q: numpy.ndarray, e: int) -> float:
        v = self.index.vectors[e]
        d = self.index.distance(q, v)[0][0]
        return d

    def search(
        self, q: numpy.ndarray, ep: int, ef: int
    ) -> tuple[list[float], list[int]]:
        ep_dist = self.distance_to_node(q, ep)

        v = {ep}
        C = [(ep_dist, ep)]
        W = [(ep_dist, ep)]

        while len(C) > 0:
            d_c, c = heappop(C)
            d_f, f = nlargest(1, W, key=lambda x: x[0])[0]

            if d_c > d_f:
                break

            for e in self.G[c]:
                if e not in v:
                    v.add(e)
                    d_f, f = nlargest(1, W, key=lambda x: x[0])[0]
                    d_e = self.distance_to_node(q, e)

                    if d_e < d_f or len(W) < ef:
                        heappush(C, (d_e, e))
                        heappush(W, (d_e, e))

                        if len(W) > ef:
                            W = nsmallest(ef, W, key=lambda x: x[0])

        return tuple(zip(*W))

    def insert(self, q: numpy.ndarray, node: int, ep: int) -> None:
        if len(self.G) == 0:
            self.G.add_node(node)
            return

        D, W = self.search(q, ep, self.config.ef_construction)
        neighbors = self.select_neighbors(q, D, W, self.config.M)
        self.G.add_edges_from([(e, node, {"distance": float(d)}) for d, e in neighbors])

        for d, e in neighbors:
            if len(self.G[e]) > self.M_max:
                D, W = list(zip(*[(self.G[e][n]["distance"], n) for n in self.G[e]]))
                new_conn = self.select_neighbors(
                    self.index.vectors[e], D, W, self.M_max
                )
                self.G.remove_edges_from([(e, e_n) for e_n in self.G[e]])
                self.G.add_edges_from(
                    [(e, e_n, {"distance": d_n}) for d_n, e_n in new_conn]
                )

    def select_neighbors(
        self, q: numpy.ndarray, D: list[float], W: list[int], M: int
    ) -> list[tuple[float, int]]:
        return nlargest(M, zip(D, W), key=lambda x: x[0])


def visualize_hnsw_index(index: HNSWIndex):
    """
    Use this to visualize the different layers of HNSW graphs. The nodes
    maintain consistent locations between layers, and the layers are
    plotted next to each other.
    """

    _, axs = plt.subplots(1, len(index.layers), figsize=(len(index.layers) * 5, 5))

    layout = networkx.spring_layout(index.layers[0].G)
    node_color = ["r" if index.ep == node else "c" for node, _ in layout.items()]

    for i, layer in enumerate(index.layers):
        graph = layer.G

        graph_layout = {k: v for k, v in layout.items() if k in graph}
        graph_node_color = [node_color[k] for k, _ in graph_layout.items()]
        networkx.draw(
            graph, graph_layout, ax=axs[i], node_size=25, node_color=graph_node_color
        )
        axs[i].set_title(f"Layer {i}")

    plt.show()


if __name__ == "__main__":
    # index = HNSWIndex(2)
    # vectors = numpy.random.randn(10, 2)
    # index.add(vectors)

    # for ix, v in enumerate(vectors):
    #     print(ix, index.search(v, 1))

    # visualize_hnsw_index(index)
    config = HNSWConfig(
        M=40, M_max=40, M_max0=40, m_L=(1.0 / math.log(40)), ef_construction=64
    )

    data, queries, labels = load_sift()

    index = HNSWIndex(128, config=config)
    index.add(data)

    I = []
    for q in queries:
        D_q, I_q = index.search(q, k=1)
        I.append(I_q[0])


    print(f"Recall@1: {evaluate(labels, I)}")
