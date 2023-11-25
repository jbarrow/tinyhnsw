import random


class Node:
    def __init__(self, value: int, level: int) -> None:
        self.value = value
        self.pointers = [None for _ in range(level + 1)]


class SkipList:
    def __init__(self, lst: list[int] = [], max_level: int = 3, p: float = 0.5) -> None:
        assert max_level > 0

        self.max_level = max_level
        self.level = 0
        self.p = p
        self.header = Node(value=-1, level=self.max_level)

        for value in lst:
            self.insert(value)

    def _random_level(self) -> int:
        level = 0
        while random.random() < self.p and level < self.max_level-1:
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

        if current is None or current.value == value:
            # sample the level for the current node, and...
            level = self._random_level()
            print(level)
            # ...update the current level if necessary
            self.level = max(level, self.level)

            new_node = Node(value, level)

            for i in range(level):
                node = update[i]
                print(node.pointers, new_node.pointers, i)
                new_node.pointers[i] = node.pointers[i]
                node.pointers[i] = new_node

    def find(self, value: int) -> bool:
        # we can rely on our closest node algorithm
        closest_node = self.closest_node(value)
        return closest_node.value == value

    def closest_node(self, value: int) -> Node:
        pass

    def delete(self, value: int) -> None:
        pass

    def __repr__(self) -> str:
        output = ""
        for level in range(self.level, -1, -1):
            current = self.header
            values = []

            while current.pointers[level] is not None:
                current = current.pointers[level]
                values.append(str(current.value))

            output += f"{level} {' '.join(values)}\n"

        return output


def test_random_level():
    for i in range(1, 5):
        s = SkipList(max_level=i, p=0.5)
        for j in range(10):
            level = s._random_level()
            # print(i, level)
            assert level <= i - 1


test_random_level()

list = [1]
print(SkipList(list))
