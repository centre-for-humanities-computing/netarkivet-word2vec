import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import r2_score

odd_df = pd.read_csv("../../evaluation/odd_one_out.csv")
odd_one_out = odd_df.to_numpy().tolist()


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
        if all([word in model.wv.key_to_index for word in words]):
            if model.wv.doesnt_match(words) == words[3]:
                accurate += 1
    # if N == 0:
    #    return 0
    accuracy = accurate / N
    return accuracy


similarity_df = pd.read_csv("../../evaluation/similarity.csv")


def accuracy_similarities(model: Word2Vec) -> float:
    """
    Tests the model on word pairs with human annotated similarity scores.
    Returns the absolute coefficient of determination (R^2) between cosine
    similarities between word vectors and human-given scores.
    """
    human_scores = similarity_df["similarity"]
    machine_scores = model.wv.n_similarity(
        similarity_df["word1"], similarity_df["word2"]
    )
    return abs(r2_score(human_scores, machine_scores))
