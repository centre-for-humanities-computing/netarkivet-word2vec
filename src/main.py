"""
Main CLI script for training the Word2Vec model with the given hyperparameters
"""
from itertools import islice
import os
import sys

from utils.text import normalized_document

# Silence imports
sys.stdout = os.devnull
import argparse
import warnings
import wandb
from gensim.models import Doc2Vec, Word2Vec

from utils.evaluation import evaluate_doc2vec, evaluate_word2vec
from utils.streams import (
    chunk,
    document_stream,
    filter_porn_topic,
    sentence_stream,
    stream_all_records,
    stream_cleaned_texts,
    tag_documents,
    to_text_stream,
)
from utils.training import train

DATA_PATH = "/work/netarkivet-cleaned/"
TEXT_CHUNKSIZE = 100_000
TEXT_SAMPLESIZE = 150_000

# Unsilence stdout
sys.stdout = sys.__stdout__


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
        help="Specifies which model should be trained on the corpus"
        + "(options: {'word2vec', 'doc2vec'} optional,default=word2vec)",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        dest="data_path",
        nargs="?",
        required=False,
        type=str,
        default=DATA_PATH,
        help=f"Path to the root directory of the data files (optional, default={DATA_PATH})",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        dest="save_path",
        required=True,
        type=str,
        help="Path, where the model is going to be saved and where the model is "
        + "initialised from",
    )
    parser.add_argument(
        "-p",
        "--preprocessing_workers",
        dest="preprocessing_workers",
        nargs="?",
        required=False,
        default=6,
        type=int,
        help="Number of processes assigned to preprocess strings for training (optional,"
        + "default=6)",
    )
    parser.add_argument(
        "-t",
        "--training_workers",
        dest="training_workers",
        nargs="?",
        required=False,
        default=6,
        type=int,
        help="Number of processes assigned to train the model (optional,default=6)",
    )
    parser.add_argument(
        "-w",
        "--window_size",
        dest="window_size",
        nargs="?",
        required=False,
        default=5,
        type=int,
        help="Window size of the work2vec model (optional,default=5)",
    )
    parser.add_argument(
        "-v",
        "--vector_size",
        dest="vector_size",
        nargs="?",
        required=False,
        default=100,
        type=int,
        help="Dimensionality of the desired word vectors (optional,default=100)",
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
    parser.set_defaults(filter_porn=True)
    parser.add_argument(
        "-g",
        "--skip_gram",
        action="store_true",
        help="Flag to force Word2Vec to use skip-gram instead of CBOW",
    )
    parser.set_defaults(skip_gram=False)
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Creating a dict for the hyperparameters, so that they are easier to log to Wandb
    # and it's easier to instantiate a model
    hyperparameters = {
        "vector_size": args.vector_size,
        "window": args.window_size,
        "workers": args.training_workers,
    }

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
    # If the model id Doc2Vec, we have to separate a testing set for evaluation
    if args.model == "doc2vec":
        # Turning the records into an iterator, so that elements can be consumed
        records = iter(records)
        # Taking the first 20_000 elements
        test_records = list(islice(records, 20_000))
        test_texts = to_text_stream(test_records)
        test_documents = [normalized_document(text) for text in test_texts]
        test_domains = [record["domain_key"] for record in records]

    # Turning records stream to text stream
    texts = to_text_stream(records)
    # Filtering porn based on topic when asked to
    if args.filter_porn:
        texts = filter_porn_topic(texts)
    # Creating text chunk stream
    text_chunks = chunk(
        texts,
        chunk_size=args.text_chunksize,
        sample_size=args.text_samplesize,
    )

    print("Initialising model")

    # We add skip_gram as a hyperparameter if it is set to True in args
    if args.skip_gram:
        hyperparameters["sg"] = 1
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
    except FileNotFoundError:
        # If loading fails, we throw a warning to the user, and initiate the model
        # with the given hyperparameters
        warnings.warn(
            f"Model not found in the directory: {args.save_path}, creating model",
            RuntimeWarning,
        )
        model = model_class(compute_loss=True, **hyperparameters)

    print("Training sequence started")
    # ----Start training----
    for loss in train(
        model=model,
        text_chunks=text_chunks,
        preprocessing=preprocess,
        save_path=os.path.join(args.save_path, f"{args.model}.model"),
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
