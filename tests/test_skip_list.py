from tinyhnsw.skip_list import SkipList


def test_random_level():
    for i in range(5):
        s = SkipList(max_level=i, p=0.5)
        for j in range(10):
            level = s._random_level()
            assert level <= i
