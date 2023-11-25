# `tinyhnsw`

TinyHNSW is a tiny, simple implementation of HNSW in Python with minimal dependencies.
It has an associated set of tutorials that build up to understanding how HNSW works.

⚠️ This tutorial is not yet complete -- it's merely something that I'm working on putting together. I will update the README as chapters are completed.

# Tutorial Outline

1. [ ] [Introduction](chapters/0_introduction.md)
2. [ ] [Nearest Neighbor Search](chapters/1_nearest_neighbor_search.md)
3. [ ] [An Overview of HNSW](chapters/2_hnsw_overview.md)
4. [ ] [Skip-Lists](chapters/3_skip_lists.md)
5. [ ] [Navigable Small Worlds](chapters/4_navigable_small_worlds.md)
6. [ ] [HNSW](chapters/5_hnsw.md)
7. [ ] [Limitations](chapters/6_limitations.md)

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

```python
from tinyhnsw import HNSWIndex

import numpy

vectors = numpy.random.randn(10, 100)

index = HNSWIndex(d=10)
index.add(vectors)
```

### Skip Lists
You can use the skip-lists as follows:

```python
from tinyhnsw.skip_list import SkipList

list = [3, 2, 1, 7, 14, 9, 6]
skiplist = SkipList(list)
print(skiplist)
```

Which will return something like the following (but not exactly, it's a random data structure after all):

```
2 |   2 3     9
1 |   2 3 6   9 14
0 | 1 2 3 6 7 9 14
```