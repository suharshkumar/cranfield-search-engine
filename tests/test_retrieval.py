"""Tests for the index and the retrieval models.

Built on a five-document toy corpus small enough that every expected score can
be worked out by hand, so a failure points at a formula rather than at a
plausible-looking number that happens to be wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ir.index import InvertedIndex
from ir.models import BM25, LatentSemanticModel, RocchioFeedback, VectorSpaceModel
from ir.text import Preprocessor

TOY = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are animals",
    "aerodynamic flow over a wing",
    "flow separation over a delta wing at high angle of attack",
]
# No stemming, no stopwords: keeps the hand computation transparent.
PRE = Preprocessor(segmenter="naive", tokeniser="naive", normaliser="none", stopwords=None)


@pytest.fixture
def index() -> InvertedIndex:
    return InvertedIndex.build(list(range(1, len(TOY) + 1)), TOY, PRE)


# ------------------------------------------------------------------- index


def test_postings_hold_term_frequencies(index: InvertedIndex) -> None:
    assert index.postings["the"] == {1: 2, 2: 2}
    assert index.postings["flow"] == {4: 1, 5: 1}
    assert index.document_frequency("wing") == 2
    assert index.document_frequency("nonexistent") == 0


def test_document_lengths_count_tokens_not_types(index: InvertedIndex) -> None:
    assert index.doc_length[1] == 6          # "the" counted twice
    assert index.n_docs == 5
    assert index.avg_doc_length == pytest.approx(sum(index.doc_length.values()) / 5)


def test_idf_matches_the_formula(index: InvertedIndex) -> None:
    assert index.idf("the") == pytest.approx(math.log(5 / 2))
    assert index.idf("mat") == pytest.approx(math.log(5 / 1))
    # A term in every document carries no information.
    assert index.idf("cat") > index.idf("the")


def test_bm25_idf_never_goes_negative(index: InvertedIndex) -> None:
    """The guard that plain RSJ IDF lacks: a term in most documents would
    otherwise score negative and penalise the documents that contain it."""
    stuffed = InvertedIndex.build([1, 2, 3], ["flow aa", "flow bb", "flow cc"], PRE)
    assert stuffed.document_frequency("flow") == 3        # in every document
    assert stuffed.bm25_idf("flow") > 0
    assert stuffed.idf("flow") == 0.0                     # plain idf bottoms out at zero


def test_forward_index_agrees_with_postings(index: InvertedIndex) -> None:
    for doc_id in index.doc_ids:
        for term, tf in index.document_terms(doc_id).items():
            assert index.postings[term][doc_id] == tf


def test_index_survives_a_save_load_round_trip(index: InvertedIndex, tmp_path) -> None:
    path = tmp_path / "index.pkl"
    index.save(path)
    other = InvertedIndex.load(path)
    assert other.postings == index.postings
    assert other.avg_doc_length == index.avg_doc_length
    # The derived tables are rebuilt, not stored -- check they actually were.
    assert other.document_terms(1) == index.document_terms(1)
    assert BM25(other).search("cat", 1) == BM25(index).search("cat", 1)


# ------------------------------------------------------------------ models


@pytest.mark.parametrize("model", [VectorSpaceModel, BM25])
def test_exact_match_ranks_first(index: InvertedIndex, model) -> None:
    top, _ = model(index).search("aerodynamic flow over a wing", k=1)[0]
    assert top == 4


def test_query_terms_go_through_the_document_pipeline() -> None:
    """A stemmed index with an unstemmed query silently returns nothing.
    'flows' must reach the postings list of 'flow'."""
    stemmed = InvertedIndex.build([1, 2], ["flow separation", "cat mat"],
                                  Preprocessor("naive", "naive", "stem", None))
    assert stemmed.postings.keys() == {"flow", "separ", "cat", "mat"}
    assert BM25(stemmed).search("flows", k=1)[0][0] == 1


def test_unmatched_query_returns_nothing(index: InvertedIndex) -> None:
    assert VectorSpaceModel(index).search("quantum chromodynamics") == []
    assert BM25(index).search("quantum chromodynamics") == []


def test_cosine_scores_stay_in_range(index: InvertedIndex) -> None:
    for _, score in VectorSpaceModel(index).search("flow over a wing", k=5):
        assert -1e-9 <= score <= 1 + 1e-9


def test_bm25_b_zero_ignores_document_length(index: InvertedIndex) -> None:
    """With b=0 the length-normalisation term vanishes, so two documents with
    the same term frequency must score identically however long they are."""
    idx = InvertedIndex.build([1, 2], ["wing", "wing " + "filler " * 50], PRE)
    unnormalised = BM25(idx, b=0.0).score(["wing"])
    assert unnormalised[1] == pytest.approx(unnormalised[2])
    normalised = BM25(idx, b=1.0).score(["wing"])
    assert normalised[1] > normalised[2]        # the short document now wins


def test_bm25_saturates_in_term_frequency(index: InvertedIndex) -> None:
    """Doubling a term's count must give strictly less than double the score."""
    idx = InvertedIndex.build([1, 2], ["wing " * 1, "wing " * 8], PRE)
    s = BM25(idx, b=0.0).score(["wing"])
    assert s[2] > s[1]
    assert s[2] < 8 * s[1]


def test_lsa_projects_into_the_requested_rank(index: InvertedIndex) -> None:
    lsa = LatentSemanticModel(index, n_components=3)
    assert lsa.U.shape[1] == 3
    assert lsa._doc_vecs.shape == (index.n_docs, 3)
    assert np.all(np.diff(lsa.s) <= 1e-12)      # singular values, descending


def test_lsa_scores_every_document(index: InvertedIndex) -> None:
    """Unlike the sparse models, LSA gives even a zero-overlap document a
    score -- that is the whole point, and also its main failure mode."""
    scores = LatentSemanticModel(index, n_components=3).score(["wing"])
    assert set(scores) == set(index.doc_ids)


def test_feedback_expands_the_query_without_dropping_it(index: InvertedIndex) -> None:
    prf = RocchioFeedback(BM25(index), n_feedback=2, n_terms=5)
    assert prf.name == "bm25+prf"
    first = BM25(index).score(["wing"])
    after = prf.score(["wing"])
    assert set(first) <= set(after)             # feedback can only add documents
    assert after[4] > 0 and after[5] > 0


def test_ranking_is_deterministic_under_ties(index: InvertedIndex) -> None:
    """Two documents with identical scores must always come back in the same
    order, or every metric becomes irreproducible across runs."""
    idx = InvertedIndex.build([7, 3, 5], ["wing", "wing", "wing"], PRE)
    assert [d for d, _ in BM25(idx).search("wing", k=3)] == [3, 5, 7]
