# Nearest Neighbor Search

- what is nearest neighbor search
- why is nearest neighbor search
- how is nearest neighbor search (cosine similarity, sentence bert, etc.)
- limitations of nearest neighbor search (speed, memory, etc.)


```python
import numpy


class Index:
    is_trained: bool
    ntotal: int

    def __init__(self, d: int) -> None:
        self.ntotal = 0
        self.d = d

    def add(self, vectors: numpy.ndarray) -> None:
        raise NotImplementedError()

    def search(
        self, query: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        raise NotImplementedError()
```

```python
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
```