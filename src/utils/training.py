from typing import Iterable
from xmlrpc.client import Boolean

import pandas as pd
import wandb
from gensim.models import Word2Vec

from utils.streams import chunk, sentence_stream, stream_texts

odd_df = pd.read_csv("evaluation/odd_one_out.csv")
odd_one_out = [
    list(words)
    for words in zip(odd_df["word1"], odd_df["word2"], odd_df["word3"], odd_df["word4"])
]


def accuracy_analogies(model: Word2Vec) -> float:
    """
    Tests model accuracy on the analogies found in evaluation/analogies.txt.
    Returns accuracy score.
    TODO: Better analogies and more of them, we gotta get someone to do them.
    """
    score, _ = model.wv.evaluate_word_analogies("evaluation/analogies.txt")
    return score


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
    if N == 0:
        return 0
    accuracy = accurate / N
    return accuracy


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
        except FileNotFoundError:
            print(f"Model not found in the directory: {save_path}, creating model")
            model = Word2Vec(
                vector_size=vector_size,
                window=window_size,
                min_count=min_count,
                workers=workers,
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
        )
        if save:
            model.save(f"{save_path}/word2vec.model")
        loss = model.get_latest_training_loss()
        analogies = accuracy_analogies(model)
        odd = accuracy_odd_one_out(model)
        if log:
            wandb.log(
                {
                    "loss": loss,
                    "accuracy_analogies": analogies,
                    "accuracy_odd_one_out": odd,
                }
            )
        print(f"loss: {loss}, acc_an: {analogies}, acc_odd: {odd}")
        prev_corpus_count = model.corpus_count + prev_corpus_count
