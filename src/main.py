"""
Main CLI script for training the Word2Vec model with the given hyperparameters
"""
import argparse
import os
import warnings
from itertools import islice

import numpy as np

# I'm doing this cause Numpy throws a warning to stderr otherwise
# Which is not exactly great when you wanna call -h let's say
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import wandb
from gensim.models import Doc2Vec, Word2Vec

from utils.evaluation import evaluate_doc2vec, evaluate_word2vec
from utils.streams import (
    chunk,
    document_stream,
    filter_porn_records,
    sentence_stream,
    stream_all_records,
    tag_documents,
    to_text_stream,
)
from utils.text import normalized_document
from utils.training import train

DATA_PATH = "/work/netarkivet-cleaned/"
TEXT_CHUNKSIZE = 100_000
TEXT_SAMPLESIZE = 150_000


def create_parser() -> argparse.ArgumentParser:
    """
    Generates parser for the main function CLI.

    Returns
    ----------
    parser: argparse.ArgumentParser
        The parser boiii
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default="word2vec",
        type=str,
        help="""
        Specifies whether doc2vec or word2vec should be trained on the corpus
        (optional,default=word2vec)
        """,
    )
    parser.add_argument(
        "-d",
        "--data_path",
        dest="data_path",
        required=False,
        type=str,
        default=DATA_PATH,
        help=f"Path to the root directory of the data files (optional, default={DATA_PATH})",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        dest="save_path",
        required=False,
        type=str,
        default=None,
        help="""
        Path, where the model is going to be saved and where the model is initialised from.
        (optional, default=None)
        """,
    )
    parser.add_argument(
        "-p",
        "--preprocessing_workers",
        dest="preprocessing_workers",
        required=False,
        default=2,
        type=int,
        help="""
        Number of processes assigned to preprocess strings for training
        (optional, default=2)
        """,
    )
    parser.add_argument(
        "-t",
        "--training_workers",
        dest="training_workers",
        required=False,
        default=6,
        type=int,
        help="""
        Number of processes assigned to train the model
        (optional, default=6)
        """,
    )
    parser.add_argument(
        "-w",
        "--window",
        dest="window",
        required=False,
        default=5,
        type=int,
        help="""
        Maximum distance between the current and predicted word within a sentence.
        (optional, default=5)
        """,
    )
    parser.add_argument(
        "-v",
        "--vector_size",
        dest="vector_size",
        required=False,
        default=100,
        type=int,
        help="""
        Dimensionality of the desired word vectors.
        (optional, default=100)
        """,
    )
    parser.add_argument(
        "--sg",
        dest="sg",
        required=False,
        default=0,
        type=int,
        help="""
        Training algorithm: 1 for skip-gram; otherwise CBOW.
        (optional, default=0)
        """,
    )
    parser.add_argument(
        "--negative",
        dest="negative",
        required=False,
        default=5,
        type=int,
        help="""
        If > 0, negative sampling will be used,
        the int for negative specifies how many “noise words” should be drawn
        (usually between 5-20).
        If set to 0, no negative sampling is used.
        (optional, default=5)
        """,
    )
    parser.add_argument(
        "--hs",
        dest="hs",
        required=False,
        default=0,
        type=int,
        help="""
        If 1, hierarchical softmax will be used for model training.
        If 0, and negative is non-zero, negative sampling will be used.
        (optional, default=0)
        """,
    )
    parser.add_argument(
        "--ns_exponent",
        dest="ns_exponent",
        required=False,
        default=0.75,
        type=float,
        help="""
        The exponent used to shape the negative sampling distribution.
        A value of 1.0 samples exactly in proportion to the frequencies,
        0.0 samples all words equally,
        while a negative value samples low-frequency words more than
        high-frequency words.
        (optional, default=0.75) 
        """,
    )
    parser.add_argument(
        "--cbow_mean",
        dest="cbow_mean",
        required=False,
        default=1,
        type=int,
        help="""
        If 0, use the sum of the context word vectors.
        If 1, use the mean, only applies when cbow is used.
        (optional, default=1) 
        """,
    )
    parser.add_argument(
        "--n_chunks",
        dest="n_chunks",
        required=False,
        default=None,
        type=int,
        help="""
        Specifies how many chunks should be used to train the model.
        If not specified the model will be trained on the entire corpus.
        (optional, default=None)
        """,
    )
    # RARELY IF EVER USED, and it clutters both the code and the CLI, so I decided to remove it
    # --text_chunksize, --text_samplesize
    # Setting defaults instead
    parser.set_defaults(text_chunksize=TEXT_CHUNKSIZE, text_samplesize=TEXT_SAMPLESIZE)
    parser.add_argument(
        "-n",
        "--no_porn_filtering",
        dest="filter_porn",
        action="store_false",
        help="Flag to turn of porn filtering",
    )
    parser.add_argument(
        "--min_count",
        dest="min_count",
        required=False,
        default=100,
        type=int,
        help="""
        Ignores all words with total frequency lower than this.
        (optional, default=100)
        """,
    )
    parser.set_defaults(filter_porn=True)
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Creating a dict for the hyperparameters, so that they are easier to log to Wandb
    # and it's easier to instantiate a model
    hyperparameters = {
        "vector_size": args.vector_size,
        "window": args.window,
        "workers": args.training_workers,
        "sg": args.sg,
        "negative": args.negative,
        "hs": args.hs,
        "ns_exponent": args.ns_exponent,
        "cbow_mean": args.cbow_mean,
        "min_count": args.min_count,
    }
    if args.model == "doc2vec":
        # If we use doc2vec we have to remove the following parameters
        # Unfortunatelly gensim's API does not tolerate them :((
        hyperparameters.pop("sg")
        hyperparameters.pop("cbow_mean")

    print("Initializing logging")
    try:
        # We try to log to wandb
        wandb.init(
            project=f"netarkivet-{args.model}", entity="chcaa", config=hyperparameters
        )
        log = "wandb"
    except Exception:
        # If we don't succeed, we log to stdout instead and warn the user
        warnings.warn(
            "Weights and biases could not be initialised, logging to stdout",
            RuntimeWarning,
        )
        log = "stdout"

    print("Initialising streaming")
    # Streaming all records
    records = stream_all_records(args.data_path)
    # If the model is Doc2Vec, we have to separate a testing set for evaluation
    if args.model == "doc2vec":
        # Turning the records into an iterator, so that elements can be consumed
        records = iter(records)
        # Taking the first 20_000 elements
        test_records = list(islice(records, 20_000))
        test_texts = to_text_stream(test_records)
        test_documents = [normalized_document(text) for text in test_texts]
        test_domains = [record["domain_key"] for record in test_records]

    # Filtering porn based on topic when asked to
    if args.filter_porn:
        records = filter_porn_records(records)

    # Turning records stream to text stream
    texts = to_text_stream(records)
    # Creating text chunk stream
    text_chunks = chunk(
        texts,
        chunk_size=args.text_chunksize,
        sample_size=args.text_samplesize,
    )
    if args.n_chunks is not None:
        text_chunks = islice(text_chunks, args.n_chunks)

    print("Initialising model")
    # Check whether we want to train a word2vec or a doc2vec
    if args.model == "word2vec":
        evaluate = evaluate_word2vec
        model_class = Word2Vec
        # If we want to train word2vec, we turn the stream of texts into a sentence stream
        preprocess = lambda texts: sentence_stream(
            texts, workers=args.preprocessing_workers
        )
        wandb.config = hyperparameters
    else:
        evaluate = lambda model: evaluate_doc2vec(model, test_documents, test_domains)
        model_class = Doc2Vec
        # If we want to train doc2vec, we turn the stream of texts into a stream of TaggedDocuments
        preprocess = lambda texts: tag_documents(
            document_stream(texts, workers=args.preprocessing_workers)
        )
    # ----Initializing the model----
    try:
        # We try to load the specified model
        model = model_class.load(os.path.join(args.save_path, f"{args.model}.model"))
        print("Loading model from save path")
    except (FileNotFoundError, TypeError):
        # If loading fails, we throw a warning to the user, and initiate the model
        # with the given hyperparameters
        warnings.warn(
            f"Model not found in the directory: {args.save_path}, creating model",
            RuntimeWarning,
        )
        model = model_class(compute_loss=True, **hyperparameters)
    if args.save_path is not None:
        save_path = os.path.join(args.save_path, f"{args.model}.model")
    else:
        save_path = None
    print("Training sequence started")
    # ----Start training----
    for loss in train(
        model=model,
        text_chunks=text_chunks,
        preprocessing=preprocess,
        save_path=save_path,
    ):
        # For each chunk we log the evaluation results
        logging_info = {**evaluate(model), "Loss": loss}
        if log == "wandb":
            wandb.log(logging_info)
        else:
            print(logging_info)
    print("Training terminated")
    if log == "wandb":
        wandb.alert(title="Training terminated", text="")


if __name__ == "__main__":
    main()
