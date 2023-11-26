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
        self.is_trained = True
        self.vectors = None

    def add(self, vectors: numpy.ndarray) -> None:
        assert vectors.shape[1] == self.d

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = numpy.append(self.vectors, vectors, axis=0)

        self.ntotal = self.vectors.shape[0]

    def search(
        self, query: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        similarity = _cosine_similarity(self.vectors, query)
        indices = (-similarity).argsort(axis=0)[:k, :].T
        scores = numpy.array(
            [similarity[indices[i, :], i] for i in range(len(indices))]
        )

        return scores, indices


def _normalize(X: numpy.ndarray) -> numpy.ndarray:
    return X / numpy.expand_dims(numpy.linalg.norm(X, axis=1), axis=1)


def _cosine_similarity(X: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray:
    X = _normalize(X)
    Y = _normalize(Y)

    return numpy.dot(X, Y.T)


if __name__ == "__main__":
    data, queries, labels = load_sift()

    index = FullNNIndex(128)
    index.add(data)

    D, I = index.search(queries, k=10)

    print(f"Recall@1: {evaluate(labels, I[:, 0])}")
