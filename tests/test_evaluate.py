"""Tests for the metrics and the significance machinery.

The metrics are checked against rankings whose scores can be written down by
hand.  A metric that is subtly wrong -- an off-by-one in the nDCG discount, a
MAP that divides by the number of hits instead of the number of relevant
documents -- produces numbers that still look reasonable, so these are the
tests that stop a whole report being quietly meaningless.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ir.data import Qrels
from ir.evaluate import (average_precision, dcg_at_k, evaluate, f_measure_at_k,
                         ndcg_at_k, paired_bootstrap, precision_at_k,
                         queries_needed, recall_at_k)

RELEVANT = {1, 2, 3}


def test_precision_and_recall_at_k() -> None:
    ranking = [1, 9, 2, 8, 7]           # relevant at ranks 1 and 3
    assert precision_at_k(ranking, RELEVANT, 1) == 1.0
    assert precision_at_k(ranking, RELEVANT, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranking, RELEVANT, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranking, RELEVANT, 5) == pytest.approx(2 / 3)


def test_recall_is_zero_when_nothing_is_relevant() -> None:
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert average_precision([1, 2], set()) == 0.0


def test_f_measure_is_the_harmonic_mean_at_beta_one() -> None:
    ranking = [1, 9, 2, 8, 7]
    p, r = precision_at_k(ranking, RELEVANT, 3), recall_at_k(ranking, RELEVANT, 3)
    assert f_measure_at_k(ranking, RELEVANT, 3) == pytest.approx(2 * p * r / (p + r))
    # beta > 1 weights recall, so with p == r the value is unchanged...
    assert f_measure_at_k(ranking, RELEVANT, 3, beta=2) == pytest.approx(p)


def test_average_precision_divides_by_relevant_not_by_hits() -> None:
    """Finding one relevant document at rank 1 and missing the other two is
    not a perfect result."""
    assert average_precision([1, 9, 8], RELEVANT) == pytest.approx(1 / 3)
    assert average_precision([1, 2, 3], RELEVANT) == 1.0
    # (1/1 + 2/3) / 3
    assert average_precision([1, 9, 2], RELEVANT) == pytest.approx((1 + 2 / 3) / 3)


def test_average_precision_rewards_earlier_ranks() -> None:
    assert average_precision([1, 2, 9, 9], RELEVANT) > average_precision([9, 1, 2, 9], RELEVANT)


def test_dcg_discount_is_log2_of_rank_plus_one() -> None:
    assert dcg_at_k([4], 1) == pytest.approx(4 / math.log2(2))          # rank 1: no discount
    assert dcg_at_k([4, 4], 2) == pytest.approx(4 + 4 / math.log2(3))
    assert dcg_at_k([4, 4, 4], 2) == dcg_at_k([4, 4], 2)                # cutoff respected


def test_ndcg_is_one_for_the_ideal_ranking() -> None:
    qrels = Qrels({1: {10: 1, 20: 2, 30: 4}})       # grades: 1 best, 4 worst
    assert ndcg_at_k([10, 20, 30], qrels, 1, 3) == pytest.approx(1.0)
    # Reversing puts the worst document first, so nDCG must drop below 1.
    assert ndcg_at_k([30, 20, 10], qrels, 1, 3) < 1.0


def test_ndcg_respects_the_inverted_grade_scale() -> None:
    """If the grades were read the wrong way round, the reversed ranking would
    score 1.0 and the correct one would not."""
    qrels = Qrels({1: {10: 1, 20: 4}})
    assert ndcg_at_k([10, 20], qrels, 1, 2) > ndcg_at_k([20, 10], qrels, 1, 2)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    qrels = Qrels({1: {10: 1}})
    assert ndcg_at_k([99, 98], qrels, 1, 2) == 0.0
    assert ndcg_at_k([99], Qrels({}), 1, 2) == 0.0      # no judgements at all


# ---------------------------------------------------------------- harness


class _StubRetriever:
    """Returns a fixed ranking, so the harness is tested and not a model."""

    name = "stub"

    def __init__(self, ranking: list[int]):
        self.ranking = ranking

    def search(self, query: str, k: int = 10):
        return [(d, 1.0 / (i + 1)) for i, d in enumerate(self.ranking[:k])]


class _Q:
    def __init__(self, qid: int, text: str):
        self.query_id, self.text = qid, text


def test_evaluate_reports_one_score_per_query() -> None:
    qrels = Qrels({1: {5: 1}, 2: {6: 1}})
    result = evaluate(_StubRetriever([5, 6]), [_Q(1, "a"), _Q(2, "b")], qrels, ks=(1,))
    assert result.model == "stub"
    assert result["P@1"].tolist() == [1.0, 0.0]         # query 1 hit, query 2 missed
    assert result.means["P@1"] == pytest.approx(0.5)


# ----------------------------------------------------------- significance


def test_bootstrap_finds_no_difference_between_identical_systems() -> None:
    x = np.random.default_rng(0).random(200)
    assert paired_bootstrap(x, x.copy()) > 0.9


def test_bootstrap_detects_a_large_consistent_difference() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(200)
    b = np.clip(a - 0.3, 0, 1)          # b is worse on every single query
    assert paired_bootstrap(a, b) < 0.01


def test_bootstrap_p_value_is_never_zero() -> None:
    """With ``n`` resamples the smallest attainable p-value is 1/(n+1);
    reporting p = 0 from a finite resample would be a lie."""
    a, b = np.ones(50), np.zeros(50)
    assert paired_bootstrap(a, b, n_resamples=999) == pytest.approx(1 / 1000)


def test_bootstrap_requires_paired_input() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(5), np.zeros(6))


def test_queries_needed_grows_as_the_effect_shrinks() -> None:
    """Sample size scales as (sigma/delta)^2, so a tenth of the effect at the
    same noise level costs a hundred times the queries."""
    rng = np.random.default_rng(0)
    base = rng.random(200)
    noise = rng.normal(0, 0.2, 200)                 # per-query variance in the difference
    big = queries_needed(base + 0.10 + noise, base)
    small = queries_needed(base + 0.01 + noise, base)
    assert small > 10 * big
    assert queries_needed(base, base.copy()) == -1      # no effect to detect


def test_queries_needed_is_consistent_with_the_bootstrap() -> None:
    """A difference the bootstrap already calls significant at n=200 must not
    be reported as needing more than 200 queries, and vice versa."""
    rng = np.random.default_rng(1)
    base = rng.random(200)
    clear = base + 0.15 + rng.normal(0, 0.2, 200)
    assert paired_bootstrap(clear, base) < 0.05 and queries_needed(clear, base) < 200

    murky = base + 0.01 + rng.normal(0, 0.3, 200)
    assert paired_bootstrap(murky, base) > 0.05 and queries_needed(murky, base) > 200
