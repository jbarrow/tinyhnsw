import os
import numpy
import shutil
import tarfile
import urllib.request as request

from contextlib import closing


DATA_PATH = os.path.join('data', 'siftsmall', 'siftsmall_base.fvecs')

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


def read_fvecs(path: str) -> numpy.ndarray:
    a = numpy.fromfile(path, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')


if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        download_sift()
    
    vectors = read_fvecs(DATA_PATH)
    print(vectors.shape)
