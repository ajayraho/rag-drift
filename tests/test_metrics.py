from pipeline.metrics import first_hit_rank, hit_at_k, mean, reciprocal_rank


def test_first_hit_rank():
    assert first_hit_rank([7, 3, 9], [9]) == 3
    assert first_hit_rank([7, 3, 9], [3, 9]) == 2   # first of several acceptable answers
    assert first_hit_rank([7, 3, 9], [1]) is None
    assert first_hit_rank([], [1]) is None


def test_hit_at_k():
    assert hit_at_k([1, 2, 3, 4, 5, 6], [5], k=5) == 1.0
    assert hit_at_k([1, 2, 3, 4, 5, 6], [6], k=5) == 0.0
    assert hit_at_k([1, 2], [9], k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([4, 2], [4]) == 1.0
    assert reciprocal_rank([4, 2], [2]) == 0.5
    assert reciprocal_rank([4, 2], [9]) == 0.0


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0
    assert mean([1.0, 0.0, 0.5]) == 0.5
