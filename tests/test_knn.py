from tinyhnsw.knn import cosine_similarity
from tinyhnsw import FullNNIndex

import numpy


def test_cosine_similarity():
    # X is a matrix of 10 vectors with dimension 100
    X = numpy.random.randn(10, 100)
    # test that self-similarity results in 1.
    self_sim = cosine_similarity(X, X)
    assert numpy.allclose(numpy.diag(self_sim), numpy.ones(10))


def test_add():
    X = numpy.random.randn(10, 100)
    index = FullNNIndex(100)
    index.add(X)
    assert index.ntotal == 10
    index.add(X)
    assert index.ntotal == 20


def test_search_shapes():
    X = numpy.random.randn(10, 100)
    index = FullNNIndex(100)
    index.add(X)
    D, I = index.search(X, 5)

    assert D.shape == (10, 5)
    assert I.shape == (10, 5)


def test_self_search():
    X = numpy.random.randn(10, 100)
    index = FullNNIndex(100)
    index.add(X)
    D, I = index.search(X, 1)

    assert D.shape == (10, 1)
    assert I.shape == (10, 1)

    assert numpy.allclose(D[:, 0], numpy.ones_like(D[:, 0]))
    assert numpy.allclose(I, numpy.expand_dims(numpy.arange(10), axis=1))
