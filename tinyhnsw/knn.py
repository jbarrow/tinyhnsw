from __future__ import annotations
from tinyhnsw.index import Index
from tinyhnsw.utils import load_sift, evaluate

import numpy


class FullNNIndex(Index):
    """
    A full nearest-neighbors index. It uses cosine similarity as
    the default similarity measure -- we can swap it with inner product
    if we make the assumption that the matrices are pre-normed.
    """

    def __init__(self, d: int, distance: str = "cosine") -> None:
        super().__init__(d, distance)

    def search(
        self, query: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        if not self.is_trained:
            return ([[]], [[]])

        similarity = self.f_distance(self.vectors, query)
        indices = similarity.argsort(axis=0)[:k].T
        scores = numpy.array([similarity[indices[i]] for i in range(len(indices))])

        return scores, indices


if __name__ == "__main__":
    data, queries, labels = load_sift()

    index = FullNNIndex(128, distance='l2')
    index.add(data)

    I = []
    for q in queries:
        q = numpy.expand_dims(q, axis=0)
        D_q, I_q = index.search(q, k=1)
        I.append(I_q[0])

    print(f"Recall@1: {evaluate(labels, numpy.array(I).squeeze())}")
