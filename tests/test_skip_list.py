from tinyhnsw.teaching.skip_list import SkipList


def test_random_level():
    for i in range(5):
        s = SkipList(max_level=i, p=0.5)
        for j in range(10):
            level = s._random_level()
            assert level <= i


def test_set_ordering():
    s = SkipList([3, 1, 6, 2])
    assert s.tolist() == [1, 2, 3, 6]


def test_insertion():
    s = SkipList()
    s.insert(3)
    s.insert(1)
    s.insert(6)
    s.insert(2)
    assert s.tolist() == [1, 2, 3, 6]


def test_find_element_exists():
    s = SkipList([3, 1, 6, 2])
    assert s.find(3) and s.find(3).value == 3
    assert s.find(1) and s.find(1).value == 1
    assert s.find(6) and s.find(6).value == 6
    assert s.find(2) and s.find(2).value == 2


def test_find_element_missing():
    s = SkipList([3, 1, 6, 2])
    assert s.find(-1) is None
    assert s.find(-100) is None
    assert s.find(4) is None
    assert s.find(100) is None


def test_delete_all():
    s = SkipList([3, 1, 6, 2])
    s.delete(3)
    assert s.tolist() == [1, 2, 6]
    s.delete(2)
    assert s.tolist() == [1, 6]
    s.delete(6)
    assert s.tolist() == [1]
    s.delete(1)
    assert s.tolist() == []


def test_delete_nonexistent():
    s = SkipList([3, 1, 6, 2])
    s.delete(4)
    assert s.tolist() == [1, 2, 3, 6]
    s.delete(-4)
    assert s.tolist() == [1, 2, 3, 6]
    s.delete(9)
    assert s.tolist() == [1, 2, 3, 6]


def test_delete_reinsert():
    s = SkipList([3, 1, 6, 2])
    s.delete(1)
    s.insert(1)
    assert s.tolist() == [1, 2, 3, 6]
    s.delete(6)
    s.insert(19)
    assert s.tolist() == [1, 2, 3, 19]
