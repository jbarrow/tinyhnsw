from __future__ import annotations

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
