from itertools import islice
import multiprocessing
import os
import random
from typing import Callable, Iterable
import pandas as pd

from utils.text import sentences


def reusable(gen_func: Callable) -> Callable:
    """
    Function decorator that turns your generator function into an
    iterator, thereby making it reusable.
    """

    class _multigen:
        def __init__(self, *args, limit=None, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
            self.limit = limit

        def __iter__(self):
            if self.limit is not None:
                return islice(gen_func(*self.__args, **self.__kwargs), self.limit)
            else:
                return gen_func(*self.__args, **self.__kwargs)

    return _multigen


@reusable
def chunk(iterable: Iterable, chunk_size: int, sample_size: int = None):
    """
    Generator function that chunks an iterable for you returning chunk_size
    length lists.
    If sample_size is provided, the yielded list will be randomly sampled
    from the chunk with replacement.
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


def chunked(chunk_size: int, sample: int = None):
    """
    Decorator that chunks a generator function.
    """

    def _chunked(gen_func: Callable):
        def _iterable(*args, **kwargs):
            return chunk(gen_func(*args, **kwargs), chunk_size, sample_size=sample)

        return _iterable

    return _chunked


@reusable
def snappy_files(data_path: str = "."):
    """
    Generator function yielding all snappy parquet file paths
    from the directory - data_path
    """
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".snappy.parquet") and os.path.isfile(file_path):
                yield file_path


def stream_texts(files: Iterable):
    """
    Generator function that streams all texts from all file paths.
    """
    for file in files:
        df = pd.read_parquet(file)
        df = df[df["language"] == "da"]
        for text in df["content"]:
            yield text


def sentence_stream(
    texts: Iterable, window_size: int = 5, chunksize: int = 2000, workers: int = 6
):
    """
    Processes the supplied texts in parallel with the supplied number of
    workers and chunksize.
    Yields all sentences as lists of tokens that are over length window_size.
    """
    with multiprocessing.Pool(processes=workers) as pool:
        docs = pool.imap_unordered(sentences, texts, chunksize=chunksize)
        for doc in docs:
            for sentence in doc:
                if len(sentence) >= window_size:
                    yield sentence
