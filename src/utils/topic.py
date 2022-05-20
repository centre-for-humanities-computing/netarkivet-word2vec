"""
Module with utilities for topic modelling
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from utils.dansk import STOPORD


def create_topic_model(
    document_stream: Iterable[List[str]],
    n_topics: int,
    max_freq: Optional[float] = None,
    max_vocab: int = 15_000,
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
    max_vocab: int, default 15_000
        Maximum number of words to be included in the model.
        If the sample size is too large, the vocabulary can realy get out of
        hand and the tf-idf embeddings will occupy way to much memory.
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
    # Lowercasing it in case some imbecile (me) gives capital letters as input
    model_type = model_type.lower()
    if model_type not in {"nmf", "lda"}:
        raise ValueError(
            f"Given model type({model_type}) does not exist, please use either 'nmf' or 'lda'"
        )
    # Transforming split documents to joint texts
    documents = (" ".join(words) for words in document_stream)
    vectorizer = TfidfVectorizer(
        stop_words=STOPORD,
        max_features=max_vocab,
        max_df=max_freq or 1.0,
    )
    print("----Fitting Tf-idf vectorizer----")
    matrix = vectorizer.fit_transform(documents)
    print("----Fitting Topic model----")
    model_class = NMF if model_type == "nmf" else LatentDirichletAllocation
    model = model_class(
        n_components=n_topics,
    ).fit(matrix)
    return model, matrix, vectorizer


@dataclass
class PornClassifier:
    """
    Class for porn classification based on a topic model previously saved to disk.

    Chooses the highest ranking topic as label for each text and checks whether it is
    the porn topic.

    Attributes
    ----------
    vectorizer: TfidfVectorizer
        Vectorizer to transform texts into tf-idf embeddings
    topic_model: NMF or LatentDirichletAllocation
        The topic model used to determine which topic the text belongs to
    porn_id: int
        Id of the porn topic
    """

    vectorizer: TfidfVectorizer
    topic_model: Union[NMF, LatentDirichletAllocation]
    porn_id: int

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        """
        Predicts whether the given texts are porn or not.

        Parameters
        ----------
        texts: iterable of str
            A stream of strings to give predictions for

        Returns
        ----------
        is_porn: ndarray of bool of shape (n_texts,)
            A numpy array containing whether each text is porn or not
        """
        tf_idf_matrix = self.vectorizer.transform(texts)
        topic_embeddings = self.topic_model.transform(tf_idf_matrix)
        topic_labels = np.argmax(topic_embeddings, axis=1)
        return topic_labels == self.porn_id

    @classmethod
    def load(
        cls, model_name: str, load_path: str = "/work/topic_model/"
    ) -> PornClassifier:
        """
        Loads a given topic model from the given path and given model name.

        Parameters
        ----------
        model_name: str
            name under which the topic model is saved
        load_path: str, default '/work/topic_model'
            Path where all topic models can be found

        Returns
        ----------
        classifier: PornClassifier
            Loaded and initialised classifier

        Notes
        ----------
        BEWARE: This method is highly specific to our project structure.
        Load manually if you have a different one.
        """
        vectorizer = joblib.load(os.path.join(load_path, f"tf-idf_{model_name}.joblib"))
        topic_model = joblib.load(os.path.join(load_path, f"{model_name}.joblib"))
        with open(os.path.join(load_path, "porn_topics.json")) as json_file:
            textual_content = json_file.read()
            porn_id_mapping = json.loads(textual_content)
        porn_id = porn_id_mapping[model_name]
        classifier = cls(vectorizer, topic_model, porn_id)
        return classifier
