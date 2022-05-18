from typing import Iterable, List, Tuple

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

from utils.dansk import STOPORD


def create_topic_model(
    document_stream: Iterable[List[str]], n_topics: int
) -> Tuple[LatentDirichletAllocation, spmatrix, TfidfVectorizer]:
    """
    Create topic model of a corpus.

    Parameters
    -----------
    document_stream: iterable of list of str
        Stream of documents in the form of a list of words
    n_topics: int
        Number of topic we want in the model

    Returns
    -----------
    lda: LatentDirichletAllocation
        Fitted scikit-learn LDA model
    matrix: sparse matrix of shape (n_documents, n_features)
        Tf-idf embeddings of the documents in the stream
    vectorizer: TfidfVectorizer
        Tf-idf vectorizer model
    """
    vectorizer = TfidfVectorizer(stop_words=STOPORD, max_features=15_000)
    matrix = vectorizer.fit_transform(document_stream)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
    ).fit(matrix)
    return lda, matrix, vectorizer
