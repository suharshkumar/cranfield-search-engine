"""Metrics, plus the statistics needed to believe them.

The metrics are the standard set: precision, recall, F-measure, MAP, nDCG. The
statistics are the part usually left out. A table of means with no variance
estimate can't tell you whether A beating B by 0.01 nDCG means anything, and
with 225 queries it usually doesn't, so paired_bootstrap puts a p-value on
every comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import WORST_GRADE, Qrels

# --------------------------------------------------------------- set metrics


def precision_at_k(ranking: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranking[:k]) & relevant) / k


def recall_at_k(ranking: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def f_measure_at_k(ranking: list[int], relevant: set[int], k: int, beta: float = 1.0) -> float:
    """F-beta.  Note this is *not* the harmonic mean when beta != 1: it weights
    recall beta^2 times as heavily as precision."""
    p = precision_at_k(ranking, relevant, k)
    r = recall_at_k(ranking, relevant, k)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def average_precision(ranking: list[int], relevant: set[int]) -> float:
    """Mean of the precisions measured at each relevant document's rank.

    Unretrieved relevant documents contribute 0, which is why AP is divided by
    ``|relevant|`` and not by the number of hits found -- a system that
    retrieves one relevant document perfectly is not a perfect system.
    """
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


# ------------------------------------------------------------------ graded


def dcg_at_k(gains: list[float], k: int) -> float:
    """``sum g_i / log2(i + 1)`` -- the linear-gain form.

    Cranfield's grades are a 4-point judgement scale, not a utility in a ratio
    sense, so the exponential-gain variant ``(2^g - 1)`` would assert that a
    grade-4 document is 15x a grade-1 one.  Linear gain is the honest reading.
    """
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranking: list[int], qrels: Qrels, query_id: int, k: int) -> float:
    """DCG of the returned ranking over DCG of the best possible ranking."""
    gains = [qrels.gain(query_id, d) for d in ranking[:k]]
    ideal = sorted(qrels.grades.get(query_id, {}).values())
    ideal_gains = [(WORST_GRADE + 1) - g for g in ideal]   # grade 1 -> gain 4
    ideal_gains.sort(reverse=True)
    best = dcg_at_k(ideal_gains, k)
    return dcg_at_k(gains, k) / best if best > 0 else 0.0


# ----------------------------------------------------------------- harness


@dataclass
class Result:
    """Per-query scores for one model, plus their means.

    Keeping the per-query vectors (not just the means) is what makes the
    significance tests and the failure analysis possible.
    """

    model: str
    per_query: dict[str, np.ndarray]     # metric name -> (n_queries,) array

    @property
    def means(self) -> dict[str, float]:
        return {m: float(v.mean()) for m, v in self.per_query.items()}

    def __getitem__(self, metric: str) -> np.ndarray:
        return self.per_query[metric]


def evaluate(retriever, queries, qrels: Qrels, ks: tuple[int, ...] = (1, 5, 10),
             depth: int = 100) -> Result:
    """Run every query and score the ranking.

    ``depth`` is how deep the ranking is materialised.  MAP is computed over
    this depth, so it is really MAP@100; with 1400 documents and a mean of 8
    relevant per query, ranking the full collection changes it by <0.001 and
    costs 14x the time.
    """
    metrics: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        metrics.setdefault(name, []).append(value)

    for q in queries:
        ranking = [doc_id for doc_id, _ in retriever.search(q.text, k=depth)]
        relevant = qrels.relevant(q.query_id)
        for k in ks:
            add(f"P@{k}", precision_at_k(ranking, relevant, k))
            add(f"R@{k}", recall_at_k(ranking, relevant, k))
            add(f"F@{k}", f_measure_at_k(ranking, relevant, k))
            add(f"nDCG@{k}", ndcg_at_k(ranking, qrels, q.query_id, k))
        add("MAP", average_precision(ranking, relevant))

    return Result(model=getattr(retriever, "name", type(retriever).__name__),
                  per_query={m: np.asarray(v) for m, v in metrics.items()})


# -------------------------------------------------------------- significance


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000,
                     random_state: int = 0) -> float:
    """Two-sided p-value for "``a`` and ``b`` have the same mean", by resampling.

    The pairing matters: the same 225 queries are run through both systems, and
    query difficulty varies far more than the systems do.  An unpaired test
    would drown the system difference in that query variance.

    The null is imposed by centring both samples on the pooled mean before
    resampling, then counting how often a resample reproduces a difference at
    least as large as the observed one.  No normality assumption -- IR metrics
    are bounded in [0, 1] and badly skewed, so a t-test is a poor fit.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired test needs one score per query from each system")
    diff = a - b
    observed = abs(diff.mean())
    centred = diff - diff.mean()          # shift so the null (mean 0) holds
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    resampled = centred[idx].mean(axis=1)
    # +1 in numerator and denominator: the observed sample is itself one draw
    # from the null, so a p-value of exactly 0 is not attainable.
    return float((np.sum(np.abs(resampled) >= observed) + 1) / (n_resamples + 1))


def queries_needed(a: np.ndarray, b: np.ndarray, alpha: float = 0.05,
                   power: float = 0.8) -> int:
    """How many queries a test set would need to resolve the difference seen here.

    Standard paired-sample size formula, ``n = (z_a/2 + z_b)^2 * sigma_d^2 /
    delta^2``, applied to the observed per-query difference.  It answers the
    question a null result should always prompt: *was the experiment even
    capable of detecting this?*  On Cranfield the answer is usually no --
    per-query nDCG variance dwarfs the between-system difference, so a few
    hundred queries can only resolve effects of about 0.05 and up.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    delta, sigma = abs(d.mean()), d.std(ddof=1)
    if delta == 0:
        return -1                     # no effect to detect at any sample size
    z_alpha, z_beta = 1.959963985, 0.841621234      # two-sided 0.05, power 0.80
    if (alpha, power) != (0.05, 0.8):
        from scipy.stats import norm
        z_alpha, z_beta = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return int(math.ceil(((z_alpha + z_beta) * sigma / delta) ** 2))


def compare(results: list[Result], metric: str = "nDCG@10",
            baseline: str | None = None) -> list[tuple[str, float, float, float]]:
    """``(model, mean, delta vs baseline, p-value)`` rows, best model first."""
    ordered = sorted(results, key=lambda r: -r.means[metric])
    base = next((r for r in results if r.model == baseline), ordered[-1])
    rows = []
    for r in ordered:
        if r.model == base.model:
            rows.append((r.model, r.means[metric], 0.0, float("nan")))
        else:
            rows.append((r.model, r.means[metric],
                         r.means[metric] - base.means[metric],
                         paired_bootstrap(r[metric], base[metric])))
    return rows
