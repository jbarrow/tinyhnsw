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