from tinyhnsw.knn import _cosine_similarity

import numpy


def test_cosine_similarity():
    # X is a matrix of 10 vectors with dimension 100
    X = numpy.random.randn(10, 100)
    # test that self-similarity results in 1.
    self_sim = _cosine_similarity(X, X)
    assert numpy.allclose(numpy.diag(self_sim), numpy.ones(10))
