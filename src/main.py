"""
Main CLI script for training the Word2Vec model with the given hyperparameters
"""
import os
import sys

# Silence imports
sys.stdout = os.devnull
import argparse
import warnings
import wandb
from gensim.models import Doc2Vec, Word2Vec

from utils.evaluation import evaluate_word2vec
from utils.streams import chunk, document_stream, sentence_stream, stream_cleaned_texts
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
    # parser.add_argument(
    #     "--text_chunksize",
    #     dest="text_chunksize",
    #     nargs="?",
    #     required=False,
    #     default=100000,
    #     type=int,
    #     help="Size of chunks of text the model has to work on and shuffle (optional, "
    #     + "default=100_000)",
    # )
    # parser.add_argument(
    #     "--text_samplesize",
    #     dest="text_samplesize",
    #     nargs="?",
    #     required=False,
    #     default=150000,
    #     type=int,
    #     help="Size sample that has to be drawn randomly from each chunk (optional, "
    #     + "default=150_000)",
    # )

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
    print("Initialising model")
    # Creating a dict for the hyperparameters, so that they are easier to log to Wandb
    # and it's easier to instantiate a model
    hyperparameters = {
        "vector_size": args.vector_size,
        "window": args.window_size,
        "workers": args.training_workers,
    }
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
        wandb_project = "netarkivet-word2vec"
    else:
        # Since we don't have any other evaluation metrics for doc2vec than
        # loss, it gives back an empty dict which can be then expanded
        evaluate = lambda _: {}
        model_class = Doc2Vec
        # If we want to train doc2vec, we turn the stream of texts into a stream of TaggedDocuments
        preprocess = lambda texts: document_stream(
            texts, workers=args.preprocessing_workers
        )
        wandb_project = "netarkivet-doc2vec"
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

    # ----Initializing logging----
    try:
        # We try to log to wandb
        wandb.init(project=wandb_project, entity="chcaa", config=hyperparameters)
        log = "wandb"
    except Exception:
        # If we don't succeed, we log to stdout instead and warn the user
        warnings.warn(
            "Weights and biases could not be initialised, logging to stdout",
            RuntimeWarning,
        )
        log = "stdout"

    # ----Creating text chunk stream----
    text_chunks = chunk(
        stream_cleaned_texts(args.data_path, args.filter_porn),
        chunk_size=args.text_chunksize,
        sample_size=args.text_samplesize,
    )

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
