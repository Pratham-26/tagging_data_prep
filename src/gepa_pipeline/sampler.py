from __future__ import annotations

"""Corpus sampling utilities for GEPA optimization.

Provides functions to load and randomly sample texts from corpus files
in either line-delimited or CSV format.
"""

import csv
import random


def sample_corpus(path: str, n: int, delimiter: str | None = None) -> list[str]:
    """Load and randomly sample n texts from corpus file (one per line or CSV)."""
    texts: list[str] = []

    if delimiter is not None:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if row:
                    texts.append(row[0])
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    if len(texts) <= n:
        return texts

    return random.sample(texts, n)
