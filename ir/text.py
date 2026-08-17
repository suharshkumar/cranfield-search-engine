"""The preprocessing pipeline: raw text in, list of terms out.

Every stage here has a naive implementation and a linguistically informed one,
selectable at run time, because the point of the exercise is to *measure*
whether the sophisticated version actually buys retrieval quality rather than
to assume it does.  (On Cranfield, one of them does not -- see the README.)

    segment   sentences   regex on [.!?]        vs  NLTK Punkt
    tokenise  words       regex on \\w+          vs  Penn Treebank tokeniser
    normalise word forms  Porter stemming       vs  WordNet lemmatisation
    filter    stopwords   off / NLTK list / collection-frequency cutoff
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

# ---------------------------------------------------------------- segmentation

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def segment_naive(text: str) -> list[str]:
    """Split on a top-level punctuation mark followed by whitespace.

    Fails on abbreviations ("fig. 3", "et al.", "no. 42"), which Cranfield's
    aerodynamics prose is full of -- hence the Punkt comparison.
    """
    return [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]


def segment_punkt(text: str) -> list[str]:
    """Punkt: an unsupervised abbreviation/collocation model (Kiss & Strunk 2006)."""
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


# ---------------------------------------------------------------- tokenisation

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
_treebank = TreebankWordTokenizer()


def tokenise_naive(sentence: str) -> list[str]:
    """Maximal runs of letters, digits and internal hyphens."""
    return _WORD.findall(sentence.lower())


def tokenise_treebank(sentence: str) -> list[str]:
    """Penn Treebank rules: splits clitics and keeps punctuation as tokens.

    The punctuation tokens are dropped afterwards -- they carry no retrieval
    signal in a bag-of-words model and only inflate the vocabulary.
    """
    return [t for t in (t.lower() for t in _treebank.tokenize(sentence))
            if _WORD.fullmatch(t)]


# ----------------------------------------------------------------- normalising

_stemmer = PorterStemmer()
_lemmatiser = WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    """Porter's suffix-stripping algorithm.  Aggressive and non-linguistic:
    'boundary' -> 'boundari'.  Conflates more aggressively than lemmatisation,
    which for recall-oriented retrieval is usually the right trade."""
    return _stemmer.stem(token)


@lru_cache(maxsize=200_000)
def lemmatise(token: str) -> str:
    """WordNet lemmatisation, tried as verb then noun.

    Without a POS tag WordNet defaults to noun and leaves most verbs alone, so
    we take the verb reading when it actually changes the surface form --
    a cheap approximation to tagging that recovers most of the conflation.
    """
    verb = _lemmatiser.lemmatize(token, pos="v")
    return verb if verb != token else _lemmatiser.lemmatize(token, pos="n")


def identity(token: str) -> str:
    return token


# ------------------------------------------------------------------ stopwords

# Loaded lazily: nltk.corpus.stopwords touches the filesystem on first access.
@lru_cache(maxsize=1)
def nltk_stopwords() -> frozenset[str]:
    from nltk.corpus import stopwords
    return frozenset(stopwords.words("english"))


def frequency_stopwords(docs_tokens: list[list[str]], top_n: int = 50) -> frozenset[str]:
    """Data-driven stopwords: the ``top_n`` terms by collection frequency.

    Cranfield is a single-domain collection, so its true "function words"
    include domain words -- *flow*, *pressure*, *boundary* appear in a third of
    the abstracts and discriminate almost nothing.  A generic English list
    cannot know that; this list can.
    """
    from collections import Counter
    counts: Counter[str] = Counter()
    for toks in docs_tokens:
        counts.update(toks)
    return frozenset(t for t, _ in counts.most_common(top_n))


# ------------------------------------------------------------------- pipeline


@dataclass(frozen=True)
class Preprocessor:
    """A fully specified preprocessing configuration.

    Kept as data rather than as a subclass hierarchy so that an experiment is
    literally a list of these objects (see ``experiments/preprocessing.py``).
    """

    segmenter: str = "punkt"        # naive | punkt
    tokeniser: str = "treebank"     # naive | treebank
    normaliser: str = "stem"        # none | stem | lemma
    stopwords: frozenset[str] | None = None
    min_length: int = 2             # drop single characters ('a', 'x')

    @property
    def name(self) -> str:
        sw = "none" if not self.stopwords else f"stop{len(self.stopwords)}"
        return f"{self.segmenter}+{self.tokeniser}+{self.normaliser}+{sw}"

    def _segment(self, text: str) -> list[str]:
        return segment_punkt(text) if self.segmenter == "punkt" else segment_naive(text)

    def _tokenise(self, sentence: str) -> list[str]:
        return (tokenise_treebank if self.tokeniser == "treebank" else tokenise_naive)(sentence)

    @property
    def _normalise(self):
        return {"none": identity, "stem": stem, "lemma": lemmatise}[self.normaliser]

    def __call__(self, text: str) -> list[str]:
        """Raw text -> ordered list of index terms."""
        norm = self._normalise
        stops = self.stopwords or frozenset()
        out: list[str] = []
        for sentence in self._segment(text):
            for token in self._tokenise(sentence):
                if len(token) < self.min_length or token in stops:
                    continue
                term = norm(token)
                # Re-check after normalising: 'does' -> 'doe' escapes an
                # unstemmed stoplist, so stopping is applied on both forms.
                if term and term not in stops:
                    out.append(term)
        return out

    def batch(self, texts: list[str]) -> list[list[str]]:
        return [self(t) for t in texts]


#: The configuration the search engine uses unless told otherwise.  It is the
#: winner of the ablation in ``experiments/run_all.py`` measured on the
#: development queries -- chosen by experiment, not by taste.  (Punkt is in
#: here despite making no measurable difference on this collection, because it
#: costs nothing and is the more defensible default on any other corpus.)
DEFAULT = Preprocessor(segmenter="punkt", tokeniser="treebank",
                       normaliser="stem", stopwords=nltk_stopwords())
