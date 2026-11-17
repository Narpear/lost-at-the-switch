"""Switch statistics for a LinCE-format LID file (sanity check after download).

Usage: python scripts/data_stats.py data/raw/lince/lid_hineng/train.conll
"""
import argparse
import json

from _common import *  # noqa: F401,F403  (sets sys.path)
from switchspec.lid import read_lince_conll, switch_stats
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
for p in ap.parse_args().paths:
    sents = read_lince_conll(p)
    tags = Counter(t for s in sents for t in s.tags)
    print(p)
    print(json.dumps({**switch_stats(sents), "tag_counts": dict(tags)}, indent=2))
