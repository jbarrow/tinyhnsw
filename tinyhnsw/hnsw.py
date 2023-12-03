from __future__ import annotations
from tinyhnsw.index import Index
from tinyhnsw.utils import load_sift, evaluate
from dataclasses import dataclass
from heapq import nlargest, nsmallest, heappop, heappush, heapify
from tqdm import tqdm

import numpy
import math
import random
import networkx


random.seed(1337)


@dataclass
class HNSWConfig:
    M: int
    M_max: int
    M_max0: int
    m_L: float
    ef_construction: int
    ef_search: int

    neighbors: str = "simple"
    extend_candidates: bool = False
    keep_pruned_connections: bool = True


DEFAULT_CONFIG = HNSWConfig(
    M=16,
    M_max=16,
    M_max0=32,
    m_L=(1.0 / math.log(16)),
    ef_construction=32,
    ef_search=32,
)


class HNSWIndex(Index):
    def __init__(
        self, d: int, distance: str = "cosine", config: HNSWConfig = DEFAULT_CONFIG
    ) -> None:
        super().__init__(d, distance)

        self.config = config
        self.vectors = None

        self.ep = 0
        self.L = 0
        self.ix = 0
        self.layers = [self.layer_factory(0, self.ep)]

    def layer_factory(self, lc: int, ep: int | None = None) -> HNSWLayer:
        ep = ep or self.ep
        return HNSWLayer(self, lc, ep)

    def assign_level(self) -> int:
        return math.floor(-math.log(random.random()) * self.config.m_L)

    def add(self, vectors: numpy.ndarray) -> None:
        super().add(vectors)

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
                self.layers.append(self.layer_factory(l_new, self.ix))
            self.L = l
            self.ep = ix

        self.ix += 1

    def distance(self, q: numpy.ndarray, v: numpy.ndarray) -> numpy.ndarray:
        if len(q.shape) == 1:
            q = numpy.expand_dims(q, axis=0)

        if len(v.shape) == 1:
            v = numpy.expand_dims(v, axis=0)

        return self.f_distance(q, v)

    def search(self, q: numpy.ndarray, k: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        ep = self.ep
        ef = max(k, self.config.ef_search)
        for lc in range(self.L, 0, -1):
            ep = self.layers[lc].search(q, ep, 1)[1][0]

        W = list(zip(*self.layers[0].search(q, ep, ef)))
        neighbors = nsmallest(k, W, lambda x: x[0])
        return list(zip(*neighbors))


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

        if self.config.neighbors == "simple":
            self.f_neighbors = self.select_neighbors
        else:
            self.f_neighbors = self.select_neighbors_heuristic

    def distance_to_node(self, q: numpy.ndarray, e: int) -> float:
        v = self.index.vectors[e]
        d = self.index.distance(q, v)[0]
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
                if e in v:
                    continue

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
        neighbors = self.f_neighbors(D, W, self.config.M)
        self.G.add_edges_from([(e, node, {"distance": float(d)}) for d, e in neighbors])

        for d, e in neighbors:
            if len(self.G[e]) > self.M_max:
                D, W = list(zip(*[(self.G[e][n]["distance"], n) for n in self.G[e]]))
                new_conn = self.f_neighbors(D, W, self.M_max)
                self.G.remove_edges_from([(e, e_n) for e_n in self.G[e]])
                self.G.add_edges_from(
                    [(e, e_n, {"distance": d_n}) for d_n, e_n in new_conn]
                )

    def select_neighbors(
        self, D: list[float], W: list[int], M: int
    ) -> list[tuple[float, int]]:
        """
        Uses the "simple" way to select neighbors.
        """
        return nsmallest(M, zip(D, W), key=lambda x: x[0])

    def select_neighbors_heuristic(
        self, D: list[float], W: list[int], M: int
    ) -> list[tuple[float, int]]:
        """
        The "heuristic" method of selecting neighbors, which works
        better for clustered data.
        """
        R = []
        W_d = []
        # ensure we don't clobber the pointer
        h = list(zip(D, W))
        heapify(h)

        while len(h) > 0 and len(R) < M:
            d_e, e = heappop(h)

            if len(R) == 0 or (
                d_e
                < min([self.distance_to_node(self.index.vectors[e], n) for _, n in R])
            ):
                R.append((d_e, e))
            else:
                W_d.append((d_e, e))

        if self.config.keep_pruned_connections and len(R) < M:
            R.extend(nsmallest(M - len(R), W_d, key=lambda x: x[0]))

        return nsmallest(M, R, key=lambda x: x[0])


if __name__ == "__main__":
    # config = HNSWConfig(
    #     M=3, M_max=3, M_max0=6, m_L=(1.0 / math.log(3)), ef_construction=32, ef_search=32
    # )
    # index = HNSWIndex(2, config=config)
    # vectors = numpy.random.randn(10, 2)
    # index.add(vectors)

    # for ix, v in enumerate(vectors):
    #     print(ix, index.search(v, 1))

    # visualize_hnsw_index(index)

    data, queries, labels = load_sift()

    index = HNSWIndex(128, distance="l2", config=DEFAULT_CONFIG)
    index.add(data)

    I = []
    for q in queries:
        D_q, I_q = index.search(q, k=1)
        I.append(I_q[0])

    print(f"Recall@1: {evaluate(labels, I)}")
