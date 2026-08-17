"""The inverted index.

Maps each term to the postings list of documents containing it. TF-IDF, BM25
and LSA are all just different ways of scoring the same postings, so all three
take an InvertedIndex and none of them re-reads the corpus.

It also exposes the collection as a sparse term-document matrix, because LSA
needs the matrix form and building it from the postings costs nothing.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .text import DEFAULT, Preprocessor


@dataclass
class InvertedIndex:
    """Term -> {doc_id -> term frequency}, plus the statistics scorers need."""

    postings: dict[str, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    doc_ids: list[int] = field(default_factory=list)
    doc_length: dict[int, int] = field(default_factory=dict)   # tokens per document
    preprocessor: Preprocessor = DEFAULT

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, doc_ids: list[int], texts: list[str],
              preprocessor: Preprocessor = DEFAULT) -> "InvertedIndex":
        idx = cls(postings=defaultdict(dict), doc_ids=list(doc_ids),
                  preprocessor=preprocessor)
        for doc_id, tokens in zip(doc_ids, preprocessor.batch(texts)):
            counts = Counter(tokens)
            idx.doc_length[doc_id] = len(tokens)
            for term, tf in counts.items():
                idx.postings[term][doc_id] = tf
        idx._finalise()
        return idx

    def _finalise(self) -> None:
        self.postings = dict(self.postings)
        self._vocab = sorted(self.postings)
        self._term_pos = {t: i for i, t in enumerate(self._vocab)}
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        self._avg_len = (sum(self.doc_length.values()) / len(self.doc_ids)
                         if self.doc_ids else 0.0)
        # Forward index (doc -> {term: tf}).  Redundant with the postings, but
        # relevance feedback needs to read a document's terms and doing that
        # from the postings alone is a scan of the entire vocabulary.
        forward: dict[int, dict[str, int]] = {d: {} for d in self.doc_ids}
        for term, posting in self.postings.items():
            for doc_id, tf in posting.items():
                forward[doc_id][term] = tf
        self._forward = forward

    def document_terms(self, doc_id: int) -> dict[str, int]:
        """``{term: term frequency}`` for one document."""
        return self._forward.get(doc_id, {})

    # ------------------------------------------------------------ statistics

    @property
    def vocabulary(self) -> list[str]:
        return self._vocab

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def avg_doc_length(self) -> float:
        return self._avg_len

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency, ``log(N / df)`` with ``df >= 1``.

        A term absent from the collection gets ``idf = log(N)``; it contributes
        nothing anyway because its postings list is empty.
        """
        df = self.document_frequency(term) or 1
        return math.log(self.n_docs / df)

    def bm25_idf(self, term: str) -> float:
        """Robertson-Sparck Jones IDF, the probabilistic form BM25 assumes.

        ``log(1 + (N - df + 0.5) / (df + 0.5))``.  The ``1 +`` is the standard
        guard: without it, a term in more than half the documents scores
        negative and adding it to a query *lowers* a matching document's score.
        """
        df = self.document_frequency(term)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    # ------------------------------------------------------------ matrix form

    def term_document_matrix(self, weighting: str = "tfidf") -> csr_matrix:
        """Sparse ``(n_terms, n_docs)`` matrix.

        ``weighting='tfidf'`` uses log-normalised tf, i.e. ``1 + log(tf)``,
        times idf.  Sub-linear scaling matters here: a Cranfield abstract that
        says "flow" nine times is not nine times more about flow.
        """
        rows, cols, vals = [], [], []
        for term, posting in self.postings.items():
            r = self._term_pos[term]
            idf = self.idf(term)
            for doc_id, tf in posting.items():
                rows.append(r)
                cols.append(self._doc_pos[doc_id])
                vals.append((1.0 + math.log(tf)) * idf if weighting == "tfidf" else float(tf))
        return csr_matrix((vals, (rows, cols)),
                          shape=(len(self._vocab), self.n_docs), dtype=np.float64)

    def query_vector(self, tokens: list[str], weighting: str = "tfidf") -> np.ndarray:
        """Dense ``(n_terms,)`` query vector in the same space as the matrix."""
        v = np.zeros(len(self._vocab))
        for term, tf in Counter(tokens).items():
            j = self._term_pos.get(term)
            if j is None:
                continue        # out-of-vocabulary query term: no evidence either way
            v[j] = (1.0 + math.log(tf)) * self.idf(term) if weighting == "tfidf" else tf
        return v

    def doc_position(self, doc_id: int) -> int:
        return self._doc_pos[doc_id]

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        return pickle.loads(Path(path).read_bytes())

    def __getstate__(self) -> dict:
        # The derived lookup tables are cheap to rebuild and bulky to store.
        return {"postings": self.postings, "doc_ids": self.doc_ids,
                "doc_length": self.doc_length, "preprocessor": self.preprocessor}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._finalise()
