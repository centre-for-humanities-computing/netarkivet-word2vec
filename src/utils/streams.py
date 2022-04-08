import multiprocessing
import os
import random
import re
from itertools import islice
from typing import Callable, Iterable, List, Optional

import pandas as pd

from utils.text import sentences


def reusable(gen_func: Callable) -> Callable:
    """
    Function decorator that turns your generator function into an
    iterator, thereby making it reusable.

    Parameters
    ----------
    gen_func: Callable
        Generator function, that you want to be reusable

    Returns
    ----------
    _multigen: Callable
        Sneakily created iterator class wrapping the generator function
    """

    class _multigen:
        def __init__(self, *args, limit=None, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
            self.limit = limit

        def __iter__(self):
            if self.limit is not None:
                return islice(gen_func(*self.__args, **self.__kwargs), self.limit)
            return gen_func(*self.__args, **self.__kwargs)

    return _multigen


@reusable
def chunk(
    iterable: Iterable, chunk_size: int, sample_size: Optional[int] = None
) -> Iterable[List]:
    """
    Generator function that chunks an iterable for you.

    Parameters
    ----------
    iterable: Iterable[T]
        The iterable you'd like to chunk.
    chunk_size: int
        The size of chunks you would like to get back
    sample_size: int or None, default None
        If specified the yielded lists will be randomly sampled with the buffer
        with replacement. Sample size determines how big you want those lists to be.

    Yields
    ----------
    buffer: List[T]
        sample_size or chunk_size sized lists chunked from the original iterable
    """
    buffer = []
    for index, elem in enumerate(iterable):
        buffer.append(elem)
        if (index % chunk_size == (chunk_size - 1)) and (index != 0):
            if sample_size is None:
                yield buffer
            else:
                yield random.choices(buffer, k=sample_size)
            buffer = []


def chunked(chunk_size: int, sample_size: Optional[int] = None) -> Callable:
    """
    Decorator that chunks a generator function.

    Parameters
    ----------
    chunk_size: int
        The size of chunks you would like to get back
    sample_size: int or None, default None
        If specified the yielded lists will be randomly sampled with the buffer
        with replacement. Sample size determines how big you want those lists to be.

    Returns
    ----------
    _chunked: Callable
        Wrapper for the generator function.
    """

    def _chunked(gen_func: Callable):
        def _iterable(*args, **kwargs):
            return chunk(gen_func(*args, **kwargs), chunk_size, sample_size=sample_size)

        return _iterable

    return _chunked


# DEPRECATED


@reusable
def __snappy_files(data_path: str = "."):
    """
    Generator function yielding all snappy parquet file paths
    from the directory - data_path
    """
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".snappy.parquet") and os.path.isfile(file_path):
                yield file_path


# DEPRECATED
def __stream_texts(files: Iterable):
    """
    Generator function that streams all texts from all file paths.
    """
    for file in files:
        df = pd.read_parquet(file)
        df = df[df["language"] == "da"]
        for text in df["content"]:
            yield text


def get_years(data_path: str = ".") -> List[str]:
    """
    Gets the names of the year folders.

    Parameters
    ----------
    data_path: str, default '.'
        Specifies where our data lives

    Returns
    ----------
    years: List[str]
        List of all years processed
    """
    years = []
    for root, dirs, files in os.walk(data_path):
        for directory in dirs:
            if re.match(r"\d\d\d\d", directory):
                years.append(dir)
    return years


@reusable
def stream_cleaned_texts(data_path: str = ".") -> Iterable[str]:
    """
    Generator yielding all cleaned texts, that are not duplicates :)

    Parameters
    ----------
    data_path: str, default '.'
        Specifies where our data lives, where to get file contents from.

    Yields
    ----------
    text: str
        Cleaned text
    """
    years = get_years(data_path=data_path)
    for year in years:
        duplicates_df = pd.read_json(
            f"{data_path}/deduplicated/{year}/mask.jsonl",
            orient="records",  # the data is stored in the form of records
            lines=True,  # in a jsonl file, meaning that all records are at a new line
            dtype=False,  # We don't let the type to be inferred cause it fucks up the id
        )
        # since the id column is of format: "{year}_{file_id}_{text_id},"
        # I expand it to three new columns
        duplicates_df = duplicates_df.join(
            duplicates_df["id"]
            .str.split("_", expand=True)
            .rename(columns={0: "year", 1: "file_id", 2: "text_id"})  # type: ignore
            .astype({"year": "int32", "file_id": "int32", "text_id": "int32"})
        )
        for file_id in duplicates_df["file_id"]:
            # Loading in the current file to process
            file_df = pd.read_json(
                f"{data_path}/{year}/{file_id}.jsonl", orient="records", lines=True
            )
            # I add a column to the DataFrame based on duplicates_df to see if the text is
            # a duplicate or not
            file_df = file_df.join(
                duplicates_df[duplicates_df["file_id"] == file_id].set_index("text_id")[
                    "duplicate"
                ]
            )
            # Filtering out duplicates
            file_df = file_df[~file_df["duplicate"]]
            # Yield all remaining texts
            for text in file_df["text"]:
                yield text


def sentence_stream(
    texts: Iterable[str], window_size: int = 5, chunksize: int = 2000, workers: int = 6
) -> Iterable:
    """
    Streams sentences from the given text stream.

    Parameters
    ----------
    texts: Iterable[str]
        Text stream to sentencize
    window_size: int, default 5
        Windows size of the word2vec model.
        The stream will only yield sentences that are as least
        as long as window_size
    chunksize: int, default 2000
        Size of text chunks the stream should process in parallel
    workers: int, default 6
        Number of workers the stream should use to sentencize
        the texts coming from the stream

    Yields
    ----------
    sentence: List[str]
        List of words in a sentence
    """
    with multiprocessing.Pool(processes=workers) as pool:
        docs = pool.imap_unordered(sentences, texts, chunksize=chunksize)
        for doc in docs:
            for sentence in doc:
                if len(sentence) >= window_size:
                    yield sentence
