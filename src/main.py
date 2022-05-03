import argparse

from utils.training import initialise, train


def create_parser() -> argparse.ArgumentParser:
    """
    Generates parser for the main functiuon CLI.

    Returns
    ----------
    parser: argparse.ArgumentParser
        The parser boiii
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        dest="data_path",
        required=True,
        type=str,
        help="Path to the root directory of the data files",
    )
    parser.add_argument(
        "--save_path",
        dest="save_path",
        required=True,
        type=str,
        help="Path, where the model is going to be saved and where the model is "
        + "initialised from (if load is True)",
    )
    parser.add_argument(
        "--non_duplicates_path",
        dest="non_duplicates_path",
        required=True,
        type=str,
        help="Path, where non_duplicate id numpy files are stored",
    )
    parser.add_argument(
        "--sentence_workers",
        dest="sentence_workers",
        required=False,
        default=6,
        type=int,
        help="Number of processes assigned to yield sentences from the files (optional,"
        + "default=6)",
    )
    parser.add_argument(
        "--training_workers",
        dest="training_workers",
        required=False,
        default=6,
        type=int,
        help="Number of processes assigned to train the model (optional,default=6)",
    )
    parser.add_argument(
        "--window_size",
        dest="window_size",
        required=False,
        default=5,
        type=int,
        help="Window size of the work2vec model (optional,default=5)",
    )
    parser.add_argument(
        "--vector_size",
        dest="vector_size",
        required=False,
        default=100,
        type=int,
        help="Dimensionality of the desired word vectors (optional,default=100)",
    )
    parser.add_argument(
        "--load",
        dest="load",
        required=False,
        default=True,
        type=bool,
        help="Specifies whether to load the already existent model from save_path as an"
        + " initialization step or not (optional, default=True)",
    )
    parser.add_argument(
        "--text_chunksize",
        dest="text_chunksize",
        required=False,
        default=100000,
        type=int,
        help="Size of chunks of text the model has to work on and shuffle (optional, "
        + "default=100_000)",
    )
    parser.add_argument(
        "--text_samplesize",
        dest="text_samplesize",
        required=False,
        default=150000,
        type=int,
        help="Size sample that has to be drawn randomly from each chunk (optional, "
        + "default=150_000)",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    print("Initialising model")
    model = initialise(
        save_path=args.save_path if args.load else None,
        init_wandb=True,
        vector_size=args.vector_size,
        window_size=args.window_size,
        workers=args.training_workers,
    )
    print("Training sequence started")
    train(
        model,
        data_path=args.data_path,
        save_path=args.save_path,
        non_duplicates_path=args.non_duplicates_path,
        filter_porn=True,
        log=True,
        save=True,
        text_chunksize=args.text_chunksize,
        text_sampling_size=args.text_samplesize,
        window_size=args.window_size,
        sentence_workers=args.sentence_workers,
    )
    print("Training terminated")


if __name__ == "__main__":
    main()
