"""
Module with utilities for topic modelling
"""
from typing import Iterable, List, Tuple, Optional, Union

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

from utils.dansk import STOPORD


def create_topic_model(
    document_stream: Iterable[List[str]],
    n_topics: int,
    max_freq: Optional[float] = None,
    model_type: str = "nmf",
) -> Tuple[Union[NMF, LatentDirichletAllocation], spmatrix, TfidfVectorizer]:
    """
    Create topic model of a corpus.

    Parameters
    -----------
    document_stream: iterable of list of str
        Stream of documents in the form of a list of words
    n_topics: int
        Number of topic we want in the model
    max_freq: float or None, default None
        Frequency cutoff, words over this frequency are removed,
        if not specified, Danish stopwords are used
    model_type: {'nmf', 'lda'}, default 'nmf'
        Type of topic model to use

    Returns
    -----------
    model: NMF or LatentDirichletAllocation
        Fitted topic model
    matrix: sparse matrix of shape (n_documents, n_features)
        Tf-idf embeddings of the documents in the stream
    vectorizer: TfidfVectorizer
        Tf-idf vectorizer model

    Raises
    -----------
    ValueError:
        If a non-existant model_type is given, ValueError is raised
    """
    if model_type not in {"nmf", "lda"}:
        raise ValueError(
            f"Given model type({model_type}) does not exist, please use either 'nmf' or 'lda'"
        )
    # Transforming split documents to joint texts
    documents = (" ".join(words) for words in document_stream)
    vectorizer = TfidfVectorizer(
        stop_words=STOPORD,
        max_features=15_000,
        max_df=max_freq or 1.0,
    )
    print("----Fitting Tf-idf vectorizer----")
    matrix = vectorizer.fit_transform(documents)
    print("----Fitting LDA----")
    model_class = NMF if model_type == "nmf" else LatentDirichletAllocation
    model = model_class(
        n_components=n_topics,
    ).fit(matrix)
    return model, matrix, vectorizer
