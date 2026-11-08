"""
Retrieval metrics for the golden-set eval. Deliberately just two, both
standard and easy to explain:

  Hit@k  -- "did ANY expected issue show up in the top k results?"
            Averaged over questions it's the fraction of questions where
            search found the right issue at all.
  MRR    -- mean reciprocal rank: 1 if the expected issue is ranked
            first, 1/2 if second, 1/3 if third... 0 if it's not in the
            results. Rewards putting the right issue near the top, which
            matters because the agent only reads the first few results.

`ranked` is the list of issue numbers in the order search returned them;
`expected` is the set of issue numbers that count as a correct answer.
"""

from collections.abc import Iterable, Sequence


def first_hit_rank(ranked: Sequence[int], expected: Iterable[int]) -> int | None:
    """1-based rank of the first expected issue in `ranked`, or None."""
    expected = set(expected)
    for position, number in enumerate(ranked, start=1):
        if number in expected:
            return position
    return None


def hit_at_k(ranked: Sequence[int], expected: Iterable[int], k: int) -> float:
    rank = first_hit_rank(ranked, expected)
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(ranked: Sequence[int], expected: Iterable[int]) -> float:
    rank = first_hit_rank(ranked, expected)
    return 1.0 / rank if rank is not None else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
