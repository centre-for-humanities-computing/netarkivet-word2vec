import sys, getopt

from utils.streams import snappy_files
from utils.training import initialise, train
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        ["data_path", "-d", "--data_path"],
        dest="data_path",
        required=True,
        type=str,
        help="Path to the root directory of the data files",
    )
    parser.add_argument(
        ["save_path", "-s", "--save_path"],
        dest="save_path",
        required=True,
        type=str,
        help="Path, where the model is going to be saved and where the model is initialised from (if load is True)",
    )
    parser.add_argument(
        ["-sw", "--sentence_workers"],
        dest="sentence_workers",
        required=False,
        default=6,
        type=int,
        help="Number of processes assigned to yield sentences from the files (optional,default=6)",
    )
    parser.add_argument(
        ["-tw", "--training_workers"],
        dest="training_workers",
        required=False,
        default=6,
        type=int,
        help="Number of processes assigned to train the model (optional,default=6)",
    )
    parser.add_argument(
        ["-ws", "--window_size"],
        dest="window_size",
        required=False,
        default=5,
        type=int,
        help="Window size of the work2vec model (optional,default=5)",
    )
    parser.add_argument(
        ["-vs", "--vector_size"],
        dest="vector_size",
        required=False,
        default=100,
        type=int,
        help="Dimensionality of the desired word vectors (optional,default=100)",
    )
    parser.add_argument(
        ["-l", "--load"],
        dest="load",
        required=False,
        default=True,
        type=bool,
        help="Specifies whether to load the already existent model from save_path as an initialization step or not (optional, default=True)",
    )
    parser.add_argument(
        ["-tc", "--text_chunksize"],
        dest="text_chunksize",
        required=False,
        default=100_000,
        type=int,
        help="Size of chunks of text the model has to work on and shuffle (optional, default=100_000)",
    )
    parser.add_argument(
        ["-ts", "--text_samplesize"],
        dest="text_samplesize",
        required=False,
        default=150_000,
        type=int,
        help="Size sample that has to be drawn randomly from each chunk (optional, default=150_000)",
    )
    args = parser.parse_args()
    files = snappy_files(data_path=args.data_path)
    print("Initialising model")
    model = initialise(
        args.save_path,
        init_wandb=True,
        load=args.load,
        vector_size=args.vector_size,
        window_size=args.window_size,
        workers=args.training_workers,
    )
    print("Training sequence started")
    train(
        model,
        files,
        save_path=args.save_path,
        log=True,
        save=True,
        text_chunksize=args.text_chunksize,
        text_sampling_size=args.text_samplesize,
        window_size=args.window_size,
        sentence_workers=args.sentence_workers,
    )
    print("Training terminated")


if __name__ == "main":
    main()
