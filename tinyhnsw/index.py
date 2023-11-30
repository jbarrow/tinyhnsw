from __future__ import annotations

import numpy


class Index:
    def __init__(self, d: int) -> None:
        self.ntotal = 0
        self.vectors = None
        self.is_trained = False
        self.d = d

    def add(self, vectors: numpy.ndarray) -> None:
        assert vectors.shape[1] == self.d

        if self.vectors is None:
            self.vectors = vectors
            self.is_trained = True
        else:
            self.vectors = numpy.append(self.vectors, vectors, axis=0)

        self.ntotal = self.vectors.shape[0]

    def search(
        self, query: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        raise NotImplementedError()
    
    def save(self, file: str) -> None:
        pass

    @classmethod
    def from_file(cls, file: str) -> Index:
        pass