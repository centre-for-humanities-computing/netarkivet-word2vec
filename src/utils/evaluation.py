""" Utility functions for objectively assessing the performance of language embedding models """
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from danlp.datasets import WordSim353Da, DSD
from deprecated import deprecated
from gensim.models import Word2Vec, Doc2Vec
from scipy.stats import spearmanr

from utils.text import normalize

# Loading DataFrame with odd-one-out tests
odd_df = pd.read_csv("../evaluation/odd_one_out.csv")
odd_one_out = odd_df.to_numpy().tolist()

# Loading similarity Datasets
ws353 = WordSim353Da()
dsd = DSD()


def is_in_vocab(words: Iterable[str], model: Word2Vec) -> np.ndarray:
    """
    Checks which of the given words are in the vocab of the model

    Parameters
    ----------
    words: Iterable of str
        The words which should be checked
    model: Word2Vec
        Word2Vec model conatining the vocab

    Returns
    ----------
    contains: np.ndarray
        A bool array showing for each element whether they are found in the vocab or not.
        (kinda useful for indexing)
    """
    return np.array([word in model.wv.key_to_index for word in words])


def accuracy_odd_one_out(model: Word2Vec) -> float:
    """
    Tests model accuracy on the table provided in evaluation/odd_one_out.csv
    The model is accurate if it correctly guesses that the words in the fourth
    column are the odd ones out.

    Parameters
    ----------
    model: Word2Vec
        Word2Vec model to test

    Returns
    ----------
    accuracy: float
        Accuracy on the odd-one-out test
    """
    accurate = 0
    for words in odd_one_out:
        words = words[1:]
        words = [normalize(word, keep_sentences=False) for word in words]
        if all(word in model.wv.key_to_index for word in words):
            if model.wv.doesnt_match(words) == words[3]:
                accurate += 1
    accuracy = accurate / len(odd_one_out)
    return accuracy


@deprecated(reason="Use KeyedVectors.evaluate_word_pairs instead")
def similarity(
    words1: Iterable[str], words2: Iterable[str], model: Word2Vec
) -> np.ndarray:
    """
    Calculates similarity scores for two series of words based on the model.

    DEPRECATED - Completely unnecessary, evaluate_word_pairs does the same thing better

    Parameters
    ----------
    model: Word2Vec
        Word2Vec model to test

    Returns
    ----------
    similarities: np.ndarray
        Machine-given similarity scores of two series of words
    """
    return np.array(
        [model.wv.similarity(word1, word2) for word1, word2 in zip(words1, words2)]
    )


@deprecated(reason="Use DaNLP evaluation metrics instead")
def accuracy_similarities(model: Word2Vec) -> Tuple[float, float]:
    """
    Tests the model on word pairs with human annotated similarity scores.
    Returns the Spearman's correlation coefficient between cosine
    similarities of two word vectors and human-given similarity scores,
    as well as the extent to which the models vocabulary covers the test vocabulary.

    DEPRECATED - Using DaNLP evaluation measures is recommended

    Parameters
    ----------
    model: Word2Vec
        Word2Vec model to test

    Returns
    ----------
    rho: float
        Spearman's correlation coefficient between human-given and machine-given
        similarity scores.
    vocab_coverage: float
        Indicates how much of the test's vocab is covered by the model.
        Useful as if you evaluate the model on a small vocab it's naturally
        gonna perform better.
        (Could've returned p-value, might do it later idk)
    """
    # Load similarity table to a DataFrame
    similarity_df = pd.read_csv("../evaluation/similarity.csv")
    similarity_df = similarity_df.assign(
        word1=similarity_df["word1"].map(lambda s: normalize(s, keep_sentences=False)),
        word2=similarity_df["word2"].map(lambda s: normalize(s, keep_sentences=False)),
    )

    df = similarity_df
    df = df[is_in_vocab(df["word1"], model) & is_in_vocab(df["word2"], model)]
    human_scores = df["similarity"]
    machine_scores = similarity(df["word1"], df["word2"], model)
    rho, _ = spearmanr(human_scores, machine_scores)
    vocab_coverage = len(df) / len(similarity_df)
    return rho, vocab_coverage


def evaluate_word2vec(model: Word2Vec) -> Dict[str, float]:
    """
    Evaluates the model on different metrics.

    Parameters
    ----------
    model: Word2Vec
        Word2Vec model to test

    Returns
    ----------
    evaluation_metrics: dict
        A dictionary containing score on the odd-one-out test,
        spearman's rho, p-value and vocabulary coverage of the similarity tests.
    """
    odd = accuracy_odd_one_out(model)
    _, spearman_dsd, oov_dsd = model.wv.evaluate_word_pairs(
        dsd.file_path, delimiter="\t"
    )
    _, spearman_ws353, oov_ws353 = model.wv.evaluate_word_pairs(
        ws353.file_path, delimiter=","
    )
    return {
        "Accuracy - Odd one out": odd,
        "DSD similarities Spearman's ρ": spearman_dsd[0],
        "DSD similarities p-value": spearman_dsd[1],
        "DSD similarities out of vocabulary %": oov_dsd,
        "W353 similarities Spearman's ρ": spearman_ws353[0],
        "W353 similarities p-value": spearman_ws353[1],
        "W353 similarities out of vocabulary %": oov_ws353,
    }


def evaluate_doc2vec(
    model: Doc2Vec, texts: List[str], domains: List[str]
) -> Dict[str, float]:
    # TODO: Implement
    # TODO: Write docstring
    return {"sex": 0}
