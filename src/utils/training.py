import warnings
from typing import Optional

import wandb
from gensim.models import Word2Vec

from utils.evaluation import accuracy_odd_one_out, accuracy_similarities
from utils.streams import chunk, sentence_stream, stream_cleaned_texts


def initialise(
    save_path: Optional[str] = ".", init_wandb: bool = True, **hyperparameters
) -> Word2Vec:
    """
    If a word2vec model has previously been saved to save_path,
    it loads the model.
    Otherwise initialises a new model with the provided hyperparameters.
    If init_wandb is set to True it also initialises a run at Weights and Biases.

    Parameters
    ----------
    save_path: str or None, default None
        Path where the model has been previously saved.
        If not specified the model won't  be loaded but freshly
        created instead.
    init_wandb: bool, default True
        Specifies whether a Wandb run should be initialised.
    **hyperparameters:
        Set of hyperparameters to initialise the model with if it's not loaded

    Returns
    ----------
    model: Word2Vec
        Initialised model
    """
    model = Word2Vec(compute_loss=True, **hyperparameters)
    if save_path is not None:
        try:
            model = Word2Vec.load(f"{save_path}/word2vec.model")
            print("Loading model from save path")
        except FileNotFoundError:
            warnings.warn(
                f"Model not found in the directory: {save_path}, creating model",
                RuntimeWarning,
            )
    if init_wandb:
        wandb.init(project="netarkivet-wod-embeddings", entity="kardosdrur")
        wandb.config = hyperparameters
    return model


def train(
    model: Word2Vec,
    data_path: str,
    save_path: str,
    non_duplicates_path: str,
    text_chunksize: int = 100_000,
    text_sampling_size: int = 150_000,
    window_size: int = 5,
    log: bool = True,
    save: bool = True,
    sentence_workers: int = 6,
    verbose: bool = False,
) -> None:
    """
    Trains word2vec model on all texts under the supplied data_path

    Parameters
    ----------
    model: Word2Vec
        The model object to train
    data_path: str
        Path to load the data from
    save_path: str
        Path to save the model to after each epoch
    non_duplicates_path: str
        Path to non duplicate id numpy files
    text_chunksize: int, default 100_000
        Amount of texts that should be processed in one epoch
    text_sampling_size: int, default 150_000
        Amount of samples that should be randomly sampled from the chunks
        before they get fed to the model
    window_size: int, default 5
        Window size of the Word2Vec model
    log: bool, defualt True
        Specifies whether the training sequence should be logged to Wandb
    save: bool, default True
        Specifies whether the model should be saved after each epoch
    sentence_workers: int, default 6
        The amount of workers the sentence stream should use
    verbose: bool, default False
        Specifies whether the training sequence should be logged to stdout
    """
    text_chunks = chunk(
        stream_cleaned_texts(data_path, non_duplicates_path, verbose),
        chunk_size=text_chunksize,
        sample_size=text_sampling_size,
    )
    prev_corpus_count = model.corpus_count
    for texts in text_chunks:
        update = prev_corpus_count != 0
        sentences = sentence_stream(
            texts, window_size=window_size, workers=sentence_workers
        )
        model.build_vocab(corpus_iterable=sentences, update=update)
        sentences = sentence_stream(
            texts, window_size=window_size, workers=sentence_workers
        )
        model.train(
            corpus_iterable=sentences,
            total_examples=model.corpus_count + prev_corpus_count,  # type: ignore
            epochs=1,
            compute_loss=True,
        )
        loss = model.get_latest_training_loss()
        if save:
            model.save(f"{save_path}/word2vec.model")
        odd = accuracy_odd_one_out(model)
        rho, vocab_coverage = accuracy_similarities(model)
        if log:
            wandb.log(
                {
                    "Accuracy - Odd one out": odd,
                    "Similarities Sperman's ρ": rho,
                    "Similarities vocabulary coverage": vocab_coverage,
                    "Loss": loss,
                }
            )
        if verbose:
            print(f"acc_odd: {odd}, sim_rho: {odd}, loss: {loss}")
        prev_corpus_count = model.corpus_count + prev_corpus_count  # type: ignore
