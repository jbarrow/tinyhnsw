# `tinyhnsw`

TinyHNSW is a tiny, simple implementation of HNSW in Python with minimal dependencies.
It has an associated set of tutorials that build up to understanding how HNSW works.

⚠️ This tutorial is not yet complete -- it's merely something that I'm working on putting together. I will update the README as chapters are completed.

# Tutorial Outline

1. [ ] [Introduction](chapters/0_introduction.md)
2. [ ] [Nearest Neighbor Search](chapters/1_nearest_neighbor_search.md)
3. [ ] [An Overview of HNSW](chapters/2_hnsw_overview.md)
4. [ ] [Skip Lists](chapters/3_skip_lists.md)
5. [ ] [Navigable Small Worlds](chapters/4_navigable_small_worlds.md)
6. [ ] [HNSW](chapters/5_hnsw.md)
7. [ ] [Limitations](chapters/6_limitations.md)

## Code Completeness

1. [x] Introduction
2. [x] Nearest Neighbor Search
3. [ ] An Overview of HNSW
4. [x] Skip Lists
5. [ ] Navigable Small Worlds
6. [ ] HNSW
7. [ ] Limitations

# Library Usage

⚠️ `tinyhnsw` is NOT production-quality.
Compared to literally *any* standard implementation, it's slow and probably buggy.
If you want to use an Approximate Nearest Neighbor library in a real application, consider something like FAISS.

With that disclaimer out of the way, here is how you use it/set it up.

## Installation

To install `tinyhnsw`, run the following command:

```sh
pip install -Ue .
```

This will install the library and all its dependencies (`numpy`, `networkx`, `scipy`, `tqdm`).

## Usage

### HNSW Index

```python
from tinyhnsw import HNSWIndex

import numpy

vectors = numpy.random.randn(100, 10)

index = HNSWIndex(d=10)
index.add(vectors)

print(index.ntotal)
# => 100
```

#### HNSW Visualizations

You can also visualize each layer of the HNSW graph using the following code:

```
from tinyhnsw.hnsw import visualize_hnsw_index

# ... set up index here

visualize_hnsw_index(index)
```

Which will generate a representation like the following:

![HNSW layers visualization](chapters/figures/visualization.png)

### Full NN Index

You can evaluate the full nearest neighbors index with the following command:

```python
from tinyhnsw.utils import load_sift, evaluate
from tinyhnsw import FullNNIndex

data, queries, labels = load_sift()

index = FullNNIndex(128)
index.add(data)

D, I = index.search(queries, k=10)

print(f"Recall@1: {evaluate(labels, I[:, 0])}")
```

On my M2 MacBook Air with `numpy=1.26.2`, that runs in 0.25s and results in a recall@1 of 98%.

### Skip Lists

📝 As part of understanding how HNSW works, the tutorial walks you through how skip lists work and how to implement one. 
However, this implementation is not particularly robust and only works with integer keys.
It's there for teaching purposes, as understanding skip lists will really help understand how HNSW works.

You can use the skip lists as follows:

```python
from tinyhnsw.skip_list import SkipList

list = [3, 2, 1, 7, 14, 9, 6]
s = SkipList(list)
print(s)
```

Which will return something like the following (but not exactly, it's a random data structure after all):

```
2 |   2 3     9
1 |   2 3 6   9 14
0 | 1 2 3 6 7 9 14
```

You have a few basic operations:

```python
s.find(3)
# => Node(value=3, pointers=[...])
s.delete(3)
# => None ; removes item from skiplist
s.insert(5)
# => None ; inserts the element 5
s.tolist()
# => [1, 2, 5, 6, 7, 9, 14]
```

### Navigable Small Worlds (NSWs)

📝 The second part of understanding how HNSW works is understanding how NSWs work.
Again, we provide a teaching implementation in this repo, but it's not meant for much more than teaching.

## Testing

There are a few different kinds of tests in this repo:

1. correctness tests
2. timing tests
3. accuracy tests

### Correctness Tests

To run the correctness tests, simply run:

```sh
poetry run pytest
```

### Timing Tests

Make sure you've downloaded the data:

```sh
python tinyhnsw/utils.py
```

Which will download the [SIFT10K dataset](http://corpus-texmex.irisa.fr) to the `data/` folder.

### Accuracy Tests

