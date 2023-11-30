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

    def __init__(self, d: int) -> None:
        super().__init__(d)

    def search(
        self, query: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        if not self.is_trained:
            return ([[]], [[]])

        # similarity = cosine_similarity(self.vectors, query)
        similarity = l2_distance(self.vectors, query)
        # indices = (-similarity).argsort(axis=0)[:k, :].T
        indices = similarity.argsort(axis=0)[:k].T
        scores = numpy.array(
            [similarity[indices[i]] for i in range(len(indices))]
        )

        return scores, indices


def _normalize(X: numpy.ndarray) -> numpy.ndarray:
    return X / numpy.expand_dims(numpy.linalg.norm(X, axis=1), axis=1)


def cosine_similarity(X: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray:
    X = _normalize(X)
    Y = _normalize(Y)

    return numpy.dot(X, Y.T)


def l2_distance(X: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray:
    return numpy.linalg.norm(X - Y, axis=1)


if __name__ == "__main__":
    data, queries, labels = load_sift()

    index = FullNNIndex(128)
    index.add(data)

    I = []
    for q in queries:
        q = numpy.expand_dims(q, axis=0)
        D_q, I_q = index.search(q, k=1)
        I.append(I_q[0])

    print(f"Recall@1: {evaluate(labels, numpy.array(I))}")
