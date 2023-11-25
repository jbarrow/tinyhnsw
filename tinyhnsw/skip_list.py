"""
Simple implementation of skip-lists in python, one of the two important algorithms to
understand to implement/understand HNSW.
"""

import random


class Node:
    def __init__(self, value: int, level: int) -> None:
        self.value = value
        self.pointers = [None for _ in range(level + 1)]

    def __repr__(self) -> str:
        return str(self.value)


class SkipList:
    def __init__(self, lst: list[int] = [], max_level: int = 2, p: float = 0.5) -> None:
        assert max_level >= 0

        self.max_level = max_level  # note: max_level is 0-indexed (0 means 1 level, 1 means 2 levls, etc.)
        self.level = 0
        self.p = p
        self.header = Node(value=-1, level=self.max_level)

        for value in lst:
            self.insert(value)

    def _random_level(self) -> int:
        """
        Coin-flipping implementation for sampling -- we could make this more
        efficient using the alias method.
        """
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def insert(self, value: int) -> None:
        # list of all nodes that might need to update their forward pointer
        update = [self.header for _ in range(self.max_level + 1)]
        # step 1 is to traverse the skip-list and make a list of all the
        # nodes that need to be updated
        current = self.header

        for level in range(self.level, -1, -1):
            while (
                current.pointers[level] is not None
                and current.pointers[level].value < value
            ):
                current = current.pointers[level]
            update[level] = current

        current = current.pointers[0]

        if current is None or current.value != value:
            # sample the level for the current node, and...
            level = self._random_level()
            # ...update the current level if necessary
            self.level = max(level, self.level)

            new_node = Node(value, level)

            for i in range(level, -1, -1):
                node = update[i]
                new_node.pointers[i] = node.pointers[i]
                node.pointers[i] = new_node

    def find(self, value: int) -> Node | None:
        current = self.header
        level = self.level

        while level >= 0 and current.pointers[0] is not None:
            next = current.pointers[level]
            # move down a level
            if next is None or next.value > value:
                level -= 1
                continue
            # closest node is current node -- return it
            if next.value == value:
                return next
            # move over one pointer
            if next.value < value:
                current = current.pointers[level]

    def delete(self, value: int) -> None:
        node = self.find(value)

        if node is None:
            return

        # list of all nodes that might need to update their forward pointer
        update = [None for _ in range(self.max_level + 1)]
        # step 1 is to traverse the skip-list and make a list of all the
        # nodes that need to be updated
        current = self.header

        for level in range(self.level, -1, -1):
            while (
                current.pointers[level] is not None
                and current.pointers[level].value < value
            ):
                current = current.pointers[level]

            if (
                current
                and current.pointers[level]
                and current.pointers[level].value == value
            ):
                update[level] = current

        for level, parent in enumerate(update):
            if parent is None:
                continue

            parent.pointers[level] = node.pointers[level]

        del node

    """
    Everything below here are just helper functions for testing and pretty-printing:
    """

    def tolist(self, level: int = 0) -> list[int]:
        output = []
        current = self.header
        while current.pointers[level] is not None:
            current = current.pointers[level]
            output.append(current.value)
        return output

    def __repr__(self) -> str:
        output = ""
        # we're going to use this to get the spacing correct
        list = self.tolist()

        for level in range(self.level, -1, -1):
            current = self.header
            values = []
            ix = -1

            while current.pointers[level] is not None:
                current = current.pointers[level]
                spacing = [" " for _ in range(1, list.index(current.value) - ix)]
                values.extend([*spacing, str(current)])
                ix = list.index(current.value)

            output += f"{level} | {' '.join(values)}\n"

        return output


def test_random_level():
    for i in range(5):
        s = SkipList(max_level=i, p=0.5)
        for j in range(10):
            level = s._random_level()
            print(i, level)
            assert level <= i


# test_random_level()

list = [3, 2, 1, 7, 14, 9, 6]
skiplist = SkipList(list)
print(skiplist)
print(skiplist.tolist())

print(skiplist.find(6) is not None)
print(skiplist.find(25) is not None)
print(skiplist.find(-1) is not None)

skiplist.delete(7)

print(skiplist)
