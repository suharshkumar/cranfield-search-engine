"""Download the three NLTK resources the pipeline needs (~15 MB)."""
import nltk

for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    print(f"{resource}: {'ok' if nltk.download(resource, quiet=True) else 'FAILED'}")
