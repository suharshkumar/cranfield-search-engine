#!/usr/bin/env bash
# Fetch the Cranfield 1400 test collection (~1.5 MB).
#
# The collection is the founding artefact of IR evaluation: 1400 aerodynamics
# abstracts, 225 information needs written by the authors of those papers, and
# exhaustive relevance judgements made by domain experts.  It is small enough
# to be exhaustively judged, which is precisely what makes it still useful and
# also what limits it -- see the README on statistical power.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
URL="http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz"
mkdir -p "$DIR"

if [ -f "$DIR/cran.all.1400" ] && [ -f "$DIR/cran.qry" ] && [ -f "$DIR/cranqrel" ]; then
  echo "collection already present in $DIR"
  exit 0
fi

echo "downloading Cranfield collection from Glasgow IR..."
curl -fL --retry 3 -o "$DIR/cran.tar.gz" "$URL"
tar -xzf "$DIR/cran.tar.gz" -C "$DIR"

for f in cran.all.1400 cran.qry cranqrel; do
  [ -f "$DIR/$f" ] || { echo "error: $f missing after extraction" >&2; exit 1; }
done

echo "1400 documents, $(grep -c '^\.I' "$DIR/cran.qry") queries, $(wc -l < "$DIR/cranqrel") judgement rows"
