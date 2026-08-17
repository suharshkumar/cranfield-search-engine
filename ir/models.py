"""Retrieval models. Each ranks the whole collection for a query.

    VectorSpaceModel     cosine similarity over TF-IDF vectors
    BM25                 probabilistic scoring with length normalisation
    LatentSemanticModel  cosine in a truncated-SVD concept space
    RocchioFeedback      pseudo-relevance feedback around any of the above

All four share one interface, search(query, k) -> ranked (doc_id, score) pairs,
so the evaluation harness can treat them interchangeably.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from .index import InvertedIndex


class Retriever(ABC):
    name: str

    def __init__(self, index: InvertedIndex):
        self.index = index

    def analyse(self, query: str) -> list[str]:
        """Queries go through exactly the pipeline the documents went through.

        If they did not, a stemmed index would never match an unstemmed query
        and every score would be zero -- the single most common way to get a
        silently broken IR system.
        """
        return self.index.preprocessor(query)

    @abstractmethod
    def score(self, tokens: list[str]) -> dict[int, float]:
        """Return ``{doc_id: score}`` for documents with any evidence."""

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.score(self.analyse(query))
        # Ties broken by doc_id so runs are reproducible across platforms.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]


# --------------------------------------------------------------- vector space


class VectorSpaceModel(Retriever):
    """Cosine similarity between log-TF-IDF vectors (Salton's SMART ``ltc.ltc``).

    Scoring walks the postings lists of the query terms only, so the cost is
    proportional to the postings touched rather than to the collection size --
    the reason an inverted index exists at all.  Document norms are precomputed
    once at construction.
    """

    name = "tfidf"

    def __init__(self, index: InvertedIndex):
        super().__init__(index)
        norms: dict[int, float] = {}
        for term, posting in index.postings.items():
            idf = index.idf(term)
            for doc_id, tf in posting.items():
                w = (1.0 + math.log(tf)) * idf
                norms[doc_id] = norms.get(doc_id, 0.0) + w * w
        self._norm = {d: math.sqrt(v) for d, v in norms.items() if v > 0}

    def score(self, tokens: list[str]) -> dict[int, float]:
        idx = self.index
        q_weights: dict[str, float] = {}
        for term, tf in Counter(tokens).items():
            if term in idx.postings:
                q_weights[term] = (1.0 + math.log(tf)) * idx.idf(term)
        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm == 0:
            return {}

        acc: dict[int, float] = {}
        for term, qw in q_weights.items():
            idf = idx.idf(term)
            for doc_id, tf in idx.postings[term].items():
                acc[doc_id] = acc.get(doc_id, 0.0) + qw * (1.0 + math.log(tf)) * idf
        return {d: s / (q_norm * self._norm[d]) for d, s in acc.items() if self._norm.get(d)}


# ----------------------------------------------------------------------- BM25


class BM25(Retriever):
    """Okapi BM25 (Robertson et al., TREC-3).

    Two ideas TF-IDF cosine lacks:

    * **Term-frequency saturation.**  ``tf/(k1 + tf)`` approaches 1, so the
      tenth occurrence of a term adds almost nothing.  Cosine's ``1 + log tf``
      also damps, but never saturates.
    * **Explicit length normalisation.**  ``b`` interpolates between no
      normalisation (0) and full (1).  Cosine normalises by the vector norm,
      which over-penalises long documents that are long because they are
      genuinely about more things.

    Defaults ``k1=1.2, b=0.75`` are the values tuned on TREC and used
    unchanged here; ``experiments/tuning.py`` sweeps them on Cranfield.
    """

    name = "bm25"

    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75):
        super().__init__(index)
        self.k1, self.b = k1, b

    def score(self, tokens: list[str]) -> dict[int, float]:
        idx = self.index
        k1, b, avg = self.k1, self.b, idx.avg_doc_length
        acc: dict[int, float] = {}
        for term, qtf in Counter(tokens).items():
            posting = idx.postings.get(term)
            if not posting:
                continue
            idf = idx.bm25_idf(term)
            for doc_id, tf in posting.items():
                dl = idx.doc_length[doc_id]
                denom = tf + k1 * (1.0 - b + b * dl / avg)
                acc[doc_id] = acc.get(doc_id, 0.0) + qtf * idf * (tf * (k1 + 1.0)) / denom
        return acc


# ------------------------------------------------------------ latent semantic


class LatentSemanticModel(Retriever):
    """Latent Semantic Analysis (Deerwester et al. 1990): rank-``k`` SVD of the
    term-document matrix, then cosine in the reduced space.

    The claim is that truncation merges synonyms ("lift" / "buoyancy") onto
    shared concept dimensions and so fixes the vocabulary mismatch that kills
    exact-match retrieval.  Measured on Cranfield it is the best of the four
    models (nDCG@10 0.371 against BM25's 0.353) but the margin does not clear
    a significance test on 112 queries -- see the README.

    ``n_components`` matters far more than the choice of model does: nDCG@10
    swings from 0.30 to 0.38 across the sweep, peaking near 150 and falling
    away on both sides.  Too few dimensions and distinct topics collapse
    together; too many and the trailing singular vectors re-introduce exactly
    the term-level noise the truncation was meant to discard.
    """

    name = "lsa"

    def __init__(self, index: InvertedIndex, n_components: int = 300,
                 random_state: int = 0):
        super().__init__(index)
        self.n_components = n_components
        matrix: csr_matrix = index.term_document_matrix("tfidf")
        # svds returns the k largest singular triplets of a sparse matrix
        # without ever forming the dense (|V| x N) factorisation.
        rng = np.random.default_rng(random_state)
        v0 = rng.standard_normal(min(matrix.shape))
        U, s, Vt = svds(matrix, k=min(n_components, min(matrix.shape) - 1), v0=v0)
        order = np.argsort(-s)                      # svds returns ascending
        self.U, self.s, self.Vt = U[:, order], s[order], Vt[order]

        self._doc_vecs = (np.diag(self.s) @ self.Vt).T          # (n_docs, k)
        self._doc_norms = np.linalg.norm(self._doc_vecs, axis=1)
        self._doc_norms[self._doc_norms == 0] = 1.0
        self.explained = float((self.s ** 2).sum())

    def _fold_in(self, tokens: list[str]) -> np.ndarray:
        """Project a query into concept space: ``q_hat = S^-1 U^T q``."""
        q = self.index.query_vector(tokens, "tfidf")
        return (self.U.T @ q) / self.s

    def score(self, tokens: list[str]) -> dict[int, float]:
        q = self._fold_in(tokens)
        qn = np.linalg.norm(q)
        if qn == 0:
            return {}
        sims = (self._doc_vecs @ q) / (self._doc_norms * qn)
        return {doc_id: float(sims[i]) for i, doc_id in enumerate(self.index.doc_ids)}


# ------------------------------------------------------- relevance feedback


class RocchioFeedback(Retriever):
    """Pseudo-relevance feedback: retrieve, assume the top ``n_feedback`` hits
    are relevant, push the query toward them, retrieve again.

    Rocchio's update, with the negative term dropped (standard for the
    pseudo-relevance case -- with no true judgements the "non-relevant"
    centroid is mostly noise):

        q' = alpha * q + beta * mean(top-n document vectors)

    The failure mode is query drift: if the first pass is wrong, the second
    pass is confidently wrong.  Expansion is capped at ``n_terms`` to limit it.
    """

    name = "bm25+prf"

    def __init__(self, base: Retriever, n_feedback: int = 10, n_terms: int = 20,
                 alpha: float = 1.0, beta: float = 0.6):
        super().__init__(base.index)
        self.base, self.n_feedback, self.n_terms = base, n_feedback, n_terms
        self.alpha, self.beta = alpha, beta
        self.name = f"{base.name}+prf"

    def score(self, tokens: list[str]) -> dict[int, float]:
        first = self.base.score(tokens)
        if not first:
            return first
        top = sorted(first.items(), key=lambda kv: (-kv[1], kv[0]))[:self.n_feedback]

        idx = self.index
        centroid: Counter[str] = Counter()
        for doc_id, _ in top:
            dl = idx.doc_length[doc_id] or 1
            for term, tf in idx.document_terms(doc_id).items():
                centroid[term] += (tf / dl) * idx.idf(term)

        # Terms already in the query are not "expansion" -- re-adding them here
        # would just double their weight and crowd out the new evidence.
        seen = set(tokens)
        expansion = [t for t, _ in centroid.most_common() if t not in seen][:self.n_terms]
        # alpha/beta act as integer repeat counts: the bag-of-words scorers
        # take term frequency as the weight, so repeating a term weights it.
        expanded = tokens * max(1, round(self.alpha * 2))
        expanded += expansion * max(1, round(self.beta * 2))
        return self.base.score(expanded)


def build_all(index: InvertedIndex, lsa_components: int = 300) -> list[Retriever]:
    """The models compared in ``experiments/compare.py``, in reporting order."""
    bm25 = BM25(index)
    return [VectorSpaceModel(index), bm25,
            LatentSemanticModel(index, n_components=lsa_components),
            RocchioFeedback(bm25)]
