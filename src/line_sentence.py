"""
Script to extract a given number of chunks of data into a line sentence file.
Training supposedly speeds up, as the Word2Vec C code does not have to interact with
the Python streaming.
"""

import argparse
from itertools import islice
from utils.streams import chunk, flatten, sentence_stream, stream_cleaned_texts

DATA_PATH = "/work/netarkivet-cleaned/"


def create_parser() -> argparse.ArgumentParser:
    """
    Generates parser for the main function CLI.

    Returns
    ----------
    parser: argparse.ArgumentParser
        The parser boiii
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_path", nargs="1", type=str, help="Save path to the line_sentence file"
    )
    parser.add_argument(
        "--n_chunks",
        dest="n_chunks",
        required=False,
        default=40,
        type=str,
        help="Number of chunks to extract to the line_sentence file, (optional, default=40)",
    )
    parser.add_argument(
        "--data_path",
        dest="data_path",
        required=False,
        type=str,
        default=DATA_PATH,
        help=f"Path to the root directory of the data files (optional, default={DATA_PATH})",
    )
    return parser


def main() -> None:
    """
    Main function to run when the script is run
    """
    parser = create_parser()
    args = parser.parse_args()
    # Take the first n elementsa of all the text chunks
    text_chunks = islice(
        chunk(
            stream_cleaned_texts(args.data_path, filter_porn=True),
            chunk_size=100_000,
            sample_size=150_100,
        ),
        args.n_chunks,
    )
    with open(args.save_path, "w") as out_file:
        # Flatten the chunks so they turn into a stream of texts
        # Then feed them into sentence_stream, which transforms them to
        # a stream of normalized tokenized sentences
        sentences = sentence_stream(flatten(text_chunks), workers=4)
        for sentence in sentences:
            out_file.write(f"{sentence}\n")


if __name__ == "__main__":
    main()
