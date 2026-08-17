"""Tests for the preprocessing pipeline.

The pipeline is where a retrieval system fails silently: a broken stage still
produces tokens, the index still builds, queries still return results, and
every score is just quietly worse.  These tests pin the behaviour of each
stage independently.
"""

from __future__ import annotations

from ir.text import (Preprocessor, frequency_stopwords, lemmatise,
                     nltk_stopwords, segment_naive, segment_punkt, stem,
                     tokenise_naive, tokenise_treebank)


def test_naive_segmenter_splits_on_terminal_punctuation() -> None:
    assert segment_naive("One thing. Another thing!") == ["One thing.", "Another thing!"]


def test_punkt_only_rescues_abbreviations_it_knows() -> None:
    """Punkt fixes the abbreviations in its trained model ('Dr.') and nothing
    else: 'fig.' and 'no.', the ones Cranfield's prose actually uses, still
    split.  This is why the ablation finds the two segmenters produce an
    identical vocabulary on this collection -- an unsophisticated result, but
    the measured one."""
    assert segment_punkt("Dr. Smith arrived early. He left.") == \
        ["Dr. Smith arrived early.", "He left."]
    assert len(segment_naive("Dr. Smith arrived early. He left.")) == 3

    for missed in ("See fig. 3 for the result.", "Ref. no. 12 applies here."):
        assert segment_punkt(missed) == segment_naive(missed)


def test_treebank_tokeniser_drops_punctuation_tokens() -> None:
    assert "," not in tokenise_treebank("flow, separation, and wings")
    assert tokenise_treebank("flow, separation") == ["flow", "separation"]


def test_tokenisers_lowercase_and_keep_internal_hyphens() -> None:
    for tokenise in (tokenise_naive, tokenise_treebank):
        assert "boundary-layer" in tokenise("Boundary-Layer effects")


def test_bare_numbers_are_not_terms() -> None:
    """Terms must start with a letter: bare numbers ('1958', '2') are
    document-specific noise that inflates the vocabulary and match nothing.
    Digits *inside* a term are kept -- 'x15' is an aircraft."""
    assert tokenise_naive("mach 2 flow in 1958") == ["mach", "flow", "in"]
    assert tokenise_naive("the x15 airframe") == ["the", "x15", "airframe"]


def test_porter_stemming_conflates_inflections() -> None:
    assert stem("flows") == stem("flowing") == stem("flow")
    assert stem("boundary") == "boundari"        # not a word -- that is fine


def test_lemmatisation_keeps_real_words() -> None:
    assert lemmatise("boundaries") == "boundary"
    assert lemmatise("flying") == "fly"
    assert lemmatise("wings") == "wing"


def test_pipeline_applies_every_stage() -> None:
    pre = Preprocessor("punkt", "treebank", "stem", nltk_stopwords())
    tokens = pre("The flows over the delta wings were measured.")
    assert "the" not in tokens and "were" not in tokens      # stopped
    assert "flow" in tokens and "wing" in tokens             # stemmed


def test_stopping_is_applied_after_normalising_too() -> None:
    """'does' stems to 'doe', which escapes an unstemmed stoplist unless the
    filter runs on both the raw and the normalised form."""
    raw_only = Preprocessor("naive", "naive", "stem", frozenset({"does"}))
    both = Preprocessor("naive", "naive", "stem", frozenset({"does", "doe"}))
    assert "doe" not in raw_only("does the flow separate")   # caught before stemming
    assert both("does the flow separate") == ["the", "flow", "separ"]

    # And the other direction: a word that only *becomes* a stopword once
    # stemmed is caught by the post-normalisation check.
    assert Preprocessor("naive", "naive", "stem", frozenset({"doe"}))("does flow") == ["flow"]


def test_single_characters_are_dropped() -> None:
    assert Preprocessor("naive", "naive", "none", None)("a flow") == ["flow"]


def test_frequency_stopwords_are_the_commonest_terms() -> None:
    docs = [["flow"] * 5 + ["wing"], ["flow"] * 3 + ["rare"]]
    stops = frequency_stopwords(docs, top_n=1)
    assert stops == {"flow"}


def test_configurations_are_hashable_and_named() -> None:
    """Preprocessor is frozen so a configuration can key a cache or a results
    table; the name is what identifies a row in the ablation."""
    a = Preprocessor("punkt", "treebank", "stem", None)
    b = Preprocessor("punkt", "treebank", "stem", None)
    assert a == b and hash(a) == hash(b)
    assert a.name == "punkt+treebank+stem+none"


def test_batch_matches_calling_one_at_a_time() -> None:
    pre = Preprocessor()
    texts = ["flow over a wing", "shock waves in a nozzle"]
    assert pre.batch(texts) == [pre(t) for t in texts]
