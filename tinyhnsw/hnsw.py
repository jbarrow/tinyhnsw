from __future__ import annotations
from tinyhnsw.index import Index

import numpy
import math
import random


class HNSWIndex(Index):
    def __init__(self, d: int, M: int) -> None:
        super().__init__(d)

        self.M = M
        self.M_max = M
        self.M_max0 = 2*M

        # m_L
        self.m_L = 1. / math.log(M)

    def add(self, vectors: numpy.ndarray) -> None:
        return

    def search(
        self, query: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        return ()

    def assign_level(self) -> int:
        return math.floor(-math.log(random.random())*self.m_L)


if __name__ == '__main__':
    index = HNSWIndex(100, 3)
    for i in range(15):
        print(index.assign_level())