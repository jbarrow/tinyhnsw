import os
import numpy
import shutil
import tarfile
import urllib.request as request

from contextlib import closing


DATA_PATH = os.path.join("data", "siftsmall", "siftsmall_base.fvecs")
QUERY_PATH = os.path.join("data", "siftsmall", "siftsmall_query.fvecs")
LABEL_PATH = os.path.join("data", "siftsmall", "siftsmall_groundtruth.ivecs")


def download_sift() -> None:
    """
    Download the ANN_SIFT10K dataset, with code modified from:
        https://www.pinecone.io/learn/series/faiss/vector-indexes/
    """
    output = os.path.join("data", "siftsmall.tar.gz")

    with closing(
        request.urlopen("ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz")
    ) as r:
        with open(output, "wb") as f:
            shutil.copyfileobj(r, f)

    tar = tarfile.open(output, "r:gz")
    tar.extractall("data")


def read_vecs(path: str, ivecs: bool = False) -> numpy.ndarray:
    a = numpy.fromfile(path, dtype="int32")
    d = a[0]
    matrix = a.reshape(-1, d + 1)[:, 1:].copy()

    if not ivecs:
        matrix = matrix.view("float32")

    return matrix


def evaluate(gold: numpy.ndarray, predictions: numpy.ndarray) -> float:
    """
    Compute Recall@1;
        - gold: array of shape (k,) -- integers
        - predictions: array of shape (k,) -- integers
    """
    return sum(gold==predictions)


def load_sift() -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    if not os.path.exists(DATA_PATH):
        download_sift()

    return (
        read_vecs(DATA_PATH),
        read_vecs(QUERY_PATH),
        read_vecs(LABEL_PATH, ivecs=True)[:, 0],
    )


if __name__ == '__main__':
    load_sift()