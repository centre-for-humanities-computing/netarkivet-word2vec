from typing import Callable, Iterable, List, Optional, TypeVar, Union
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument

# A Corpus is an iterable of sentences for Word2Vec and
# an iterable of TaggedDocuments for Doc2Vec
Corpus = Union[Iterable[List[str]], Iterable[TaggedDocument]]


def train(
    model: Union[Word2Vec, Doc2Vec],
    text_chunks: Iterable[List[str]],
    preprocessing: Callable[[Iterable[str]], Corpus],
    save_path: Optional[str] = None,
) -> Iterable[float]:
    """
    Trains word2vec or doc2vec model sequentially on the supplied chunks.

    Parameters
    ----------
    model: Word2Vec or Doc2Vec
        The model object to train
    text_chunks: Iterable of list of str
        The text chunks to train the model on
    preprocessing: function, iterable of str -> Corpus
        Function that turns a chunk of texts into an iterable
        corpus for the Word2Vec/Doc2Vec model.
    save_path: str or None, default None
        Path to save the model to, if not specified,
        the model will not be saved after each chunk.

    Yields
    ----------
    loss: float
        Loss of the model after each chunk
    """
    # Check how much the model has already gone through
    prev_corpus_count = model.corpus_count
    for texts in text_chunks:
        # If the model hasn't gone through anything yet, it has to be updated
        # This flag indicates that
        update = prev_corpus_count != 0
        # Preprocess corpus for feeding them to the model
        corpus = preprocessing(texts)
        # Collect vocabulary from the chunk before training
        model.build_vocab(corpus, update=update)
        corpus = preprocessing(texts)
        # The actual training :=)
        model.train(
            corpus,
            total_examples=model.corpus_count + prev_corpus_count,  # type: ignore
            epochs=1,
            compute_loss=True,
        )
        loss = model.get_latest_training_loss()
        # Saves model if save_path is specified
        if save_path is not None:
            model.save(f"{save_path}/word2vec.model")
        # Updates corpus count
        prev_corpus_count = model.corpus_count + prev_corpus_count  # type: ignore
        yield loss
