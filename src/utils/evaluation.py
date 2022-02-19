import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.stats import linregress

odd_df = pd.read_csv("../evaluation/odd_one_out.csv")
odd_one_out = odd_df.to_numpy().tolist()


def is_in_vocab(words, model):
    return np.array([word in model.wv.key_to_index for word in words])


def accuracy_odd_one_out(model: Word2Vec) -> float:
    """
    Tests model accuracy on the table provided in evaluation/odd_one_out.csv
    The model is accurate if it correctly guesses that the words in the fourth
    column are the odd ones out.
    TODO:
        Get more of these and possibly replace them as I stole them from someone.
        Also gotta do some cleanup and lowercase all words.
    """
    N = len(odd_one_out)
    accurate = 0
    for words in odd_one_out:
        words = words[1:]
        if all([word in model.wv.key_to_index for word in words]):
            if model.wv.doesnt_match(words) == words[3]:
                accurate += 1
    # if N == 0:
    #    return 0
    accuracy = accurate / N
    return accuracy


similarity_df = pd.read_csv("../evaluation/similarity.csv")


def similarity(words1, words2, model):
    return np.array(
        [model.wv.similarity(word1, word2) for word1, word2 in zip(words1, words2)]
    )


def accuracy_similarities(model: Word2Vec) -> float:
    """
    Tests the model on word pairs with human annotated similarity scores.
    Returns the absolute coefficient of determination (R^2) between cosine
    similarities between word vectors and human-given scores.
    """
    df = similarity_df
    df = df[is_in_vocab(df["word1"], model) & is_in_vocab(df["word2"], model)]
    human_scores = df["similarity"]
    machine_scores = similarity(df["word1"], df["word2"], model)
    m = linregress(human_scores, machine_scores)
    return m.rvalue**2
