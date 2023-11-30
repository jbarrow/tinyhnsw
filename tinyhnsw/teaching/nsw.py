import numpy as np

from tinyhnsw.utils import evaluate, load_sift
from tinyhnsw.knn import cosine_similarity


class NSWIndex:
    def __init__(self):
        self.graph = {}  # Dictionary to store the graph
        self.data = []  # List to store the actual data items

    def add_item(self, item, f=10, w=5):
        # Add item to data list
        self.data.append(item)
        idx = len(self.data) - 1

        # If first item, just initialize its entry in the graph
        if idx == 0:
            self.graph[idx] = set()
            return

        # Perform k-NN search to find nearest neighbors
        neighbors = self.k_nn_search(item, w, f)

        # Connect the new item with its neighbors
        self.graph[idx] = set(neighbors)
        for neighbor in neighbors:
            self.graph[neighbor].add(idx)

    def k_nn_search(self, query, m, k):
        temp_res = set()
        candidates = set()
        visited_set = set()
        result = set()

        for i in range(m):
            candidates.add(self.get_random_entry_point())
            temp_res.clear()

            while candidates:
                c = min(candidates, key=lambda x: self.distance(query, self.data[x]))
                candidates.remove(c)

                if c in visited_set or (
                    result
                    and self.distance(query, self.data[c])
                    > self.distance(
                        query,
                        self.data[
                            max(
                                result, key=lambda x: self.distance(query, self.data[x])
                            )
                        ],
                    )
                ):
                    break

                visited_set.add(c)
                temp_res.add(c)
                for e in self.graph.get(c, []):
                    if e not in visited_set:
                        visited_set.add(e)
                        candidates.add(e)
                        temp_res.add(e)

            result.update(temp_res)

        return sorted(result, key=lambda x: self.distance(query, self.data[x]))[:k]

    def greedy_search(self, query, entry_point):
        v_curr = entry_point
        delta_min = self.distance(query, self.data[v_curr])
        v_next = None

        for v_friend in self.graph.get(v_curr, []):
            delta_fr = self.distance(query, self.data[v_friend])
            if delta_fr < delta_min:
                delta_min = delta_fr
                v_next = v_friend

        if v_next is None:
            return v_curr
        else:
            return self.greedy_search(query, v_next)

    @staticmethod
    def distance(a, b):
        # Adjusting the distance function for cosine similarity with single vectors
        return 1 - cosine_similarity([a], [b])[0, 0]

    def get_random_entry_point(self):
        import random

        return random.choice(list(self.graph.keys())) if self.graph else None


# Test the NSWIndex with some data
index = NSWIndex()

data, queries, labels = load_sift()

for i, vector in enumerate(data):
    index.add_item(vector)

print("added")

idxs = []
for vector in queries:
    idxs.append(index.greedy_search(vector, index.get_random_entry_point()))

print(idxs)
print(labels)

print(f"Recall@1: {evaluate(labels, np.array(idxs))}")
