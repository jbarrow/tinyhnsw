from __future__ import annotations
from tinyhnsw.index import Index

import numpy


class FullNNIndex(Index):
    def __init__(self, d: int) -> None:
        super().__init__(d)
        self.is_trained = True

    def add(self, vectors: numpy.ndarray) -> None:
        pass

    def search(
        self, query: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        pass
