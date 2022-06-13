"""
Module containing utility functions and constants for quality filtering documents
"""
from typing import List

import pandas as pd

# Decisions about cutoffs were made based on qualitative analysis of an
# N = 200_000 sample from netarkivet
# (obtained with reservoir sampling from the first 1 million records)

# 28 was the 99.5th percentile of word lengths, and words above this length tend
# not to be actual Danish words. (based on me reading them)
# It's worth noting, that the longest Danish word
# (speciallægepraksisplanlægningsstabiliseringsperiode)
# is 51 characters
MAX_WORD_LENGTH = 28

# 95th percentile of sentence lengths
MAX_SENTENCE_LENGTH = 40

# Let's not call it a sentence if it's shorter than 5 words :))
MIN_SENTENCE_LENGTH = 5

# 95th percentile of number of unique words per document
# Documents that have too many unique words could just be random
# mumbojumbo, not real natural text
MAX_UNIQUE_WORDS = 445


def quality_filter(document: List[List[str]]) -> List[List[str]]:
    """
    Does some quality filtering on the supplied document

    Parameters
    ----------
    document: list of list of str
        Document to filter

    Returns
    ----------
    document: list of list of str
        Document with out of place sentences and words filtered.
        If the document doesn't pass the quality filter,
        returns an empty document.
    """
    # Using pandas to access vectorized functionality
    sentences = pd.Series(document)
    # If the number of unique words is to high, we return an empty document
    n_unique_words = sentences.explode().nunique()
    if n_unique_words > MAX_UNIQUE_WORDS:
        return []
    # Filtering out words that are too long
    sentences = sentences.map(
        lambda sentence: [word for word in sentence if len(word) > MAX_WORD_LENGTH]
    )
    # Filtering out sentences that are too short or too long
    sentence_length = sentences.map(len)
    sentences = sentences[
        (sentence_length >= MIN_SENTENCE_LENGTH)
        & (sentence_length <= MAX_SENTENCE_LENGTH)
    ]
    # Converting series back to a list of lists
    return sentences.tolist()
