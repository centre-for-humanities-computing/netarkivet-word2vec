from typing import Iterable
from xmlrpc.client import Boolean

import wandb
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from utils.evaluation import accuracy_odd_one_out, accuracy_similarities
from utils.streams import chunk, sentence_stream, stream_texts


def initialise(
    save_path: str,
    init_wandb: bool = True,
    vector_size: int = 100,
    window_size: int = 5,
    min_count: int = 5,
    workers: int = 6,
    load: bool = True,
) -> Word2Vec:
    """
    If a word2vec model has previously been saved to save_path,
    it loads the model.
    Otherwise initialises a new model with the provided hyperparameters.
    If init_wandb is set to True it also initialises a run at Weights and Biases.
    """
    if load:
        try:
            model = Word2Vec.load(f"{save_path}/word2vec.model")
            print("Loading model from save path")
        except FileNotFoundError:
            print(f"Model not found in the directory: {save_path}, creating model")
            model = Word2Vec(
                vector_size=vector_size,
                window=window_size,
                min_count=min_count,
                workers=workers,
                compute_loss=True,
            )
    if init_wandb:
        wandb.init(project="netarkivet-wod-embeddings", entity="kardosdrur")
        wandb.config = {
            "vector_size": vector_size,
            "window": window_size,
            "min_count": min_count,
        }
    return model


def train(
    model: Word2Vec,
    files: Iterable,
    save_path: str = ".",
    text_chunksize: int = 100_000,
    text_sampling_size: int = 150_000,
    window_size: int = 5,
    log: Boolean = True,
    save: Boolean = True,
    sentence_workers=6,
):
    """
    Trains word2vec model on ll texts in the supplied files iterable.
    It progresses in chunks of text_chunksize texts, and takes a random
    uniform sample (with replacement) from the current chunks.
    After training on each chunk the model is saved to save_path is save is True,
    and loss and accuracy metrics are logged to Weights and Biases if
    log is True.
    """
    text_chunks = chunk(
        stream_texts(files), chunk_size=text_chunksize, sample_size=text_sampling_size
    )
    prev_corpus_count = model.corpus_count
    for texts in text_chunks:
        update = prev_corpus_count != 0
        sentences = sentence_stream(
            texts, window_size=window_size, workers=sentence_workers
        )
        model.build_vocab(corpus_iterable=sentences, update=update)
        model.train(
            corpus_iterable=sentences,
            total_examples=model.corpus_count + prev_corpus_count,
            epochs=1,
            compute_loss=True,
            callbacks=[LogCallback()],
        )
        # loss = model.get_latest_training_loss()
        if save:
            model.save(f"{save_path}/word2vec.model")
        odd = accuracy_odd_one_out(model)
        sim = accuracy_similarities(model)
        if log:
            wandb.log({"Accuracy - Odd one out": odd, "Similarities R²": sim})
        print(f"acc_odd: {odd}, sim_r²: {sim}")
        prev_corpus_count = model.corpus_count + prev_corpus_count


class LogCallback(CallbackAny2Vec):
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        wandb.log({"Loss": loss})
        print(f"Loss: {loss}")
