import json
import multiprocessing
import os
import random
import re
from itertools import islice
from typing import Callable, Iterable, List, Optional, Set, TypeVar

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


BAR_LENGTH = 100
N_DECIMALS = 1
FILL_CHARACTER = "█"

U = TypeVar("U")


@reusable
def progress_bar_stream(items: List[U]) -> Iterable[U]:
    """
    Wraps list in an iterable that shows a progress bar and the current element.

    Parameters
    ----------
    items: list of U
        Items to iterate over (of type U)

    Yields
    ----------
    item: U
        Current item under processing
    """
    total = len(items)
    for iteration, item in enumerate(items):
        percent = ("{0:." + str(N_DECIMALS) + "f}").format(
            100 * (iteration / float(total))
        )
        filledLength = int(BAR_LENGTH * iteration // total)
        bar = FILL_CHARACTER * filledLength + "-" * (BAR_LENGTH - filledLength)
        os.system("clear")
        print(f"Progress: |{bar}| {percent}% \n Current item processed: {item}\n")
        yield item


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
                years.append(directory)
    return years


def not_duplicates(mask_filename: str) -> pd.DataFrame:
    """
    Gives ya all ids in a mask file that are not duplicates

    Parameters
    ----------
    mask_filename: str
        Path to the mask file

    Returns
    ----------
    df: pd.DataFrame of structure {'file_id': int, 'text_id': int}
        DataFrame containing file ids and text IDs
    """
    # Lists to store the ids
    file_ids = []
    text_ids = []
    with open(mask_filename) as mask_file:
        for line in mask_file:
            # Parse all records in the jsonl
            record = json.loads(line)
            if not record["duplicate"]:
                # Parse file id and text id from id field
                # The first is ommited, as it stores the year
                _, file_id, text_id = record["id"].split("_")
                # Convert them to int so they don't take up as much space in memory
                # and are easier to work with
                file_id, text_id = int(file_id), int(text_id)
                file_ids.append(file_id)
                text_ids.append(text_id)
    df = pd.DataFrame({"file_id": file_ids, "text_id": text_ids})
    return df


def stream_by_text_id(file_path: str, text_ids: Set) -> Iterable[str]:
    """
    Streams all texts from a jsonl file that are in the set of text_ids supplied.

    Parameters
    ----------
    file_path: str
        Path to the jsonl file in question

    Yields
    ----------
    text: str
        Text content of the given fields
    """
    with open(file_path) as input_file:
        # Since id is not one of the fields, I have to enumerate all records
        for text_id, line in enumerate(input_file):
            if text_id in text_ids:
                # parsing the record
                record = json.loads(line)
                # If passes quality filters, it yields the content of the record
                if record["passed_quality_filter"] and record["language"] == "da":
                    yield record["text"]


@reusable
def stream_cleaned_texts(data_path: str = ".", verbose=True) -> Iterable[str]:
    """
    Generator yielding all cleaned texts, that are not duplicates :)

    Parameters
    ----------
    data_path: str, default '.'
        Specifies where our data lives, where to get file contents from.

    Yields
    ----------
    text: str
        Cleaned, normalized text
    """
    # List of all years
    years = get_years(data_path=data_path)
    for year in years:
        if verbose:
            print("Processing year: ", year)
        mask = os.path.join(data_path, f"deduplicated/{year}/mask.jsonl")
        mask_year = not_duplicates(mask_filename=mask)
        # All unique file ids, so that we can iterate over them
        file_ids = mask_year["file_id"].unique()
        for file_id in file_ids:
            if verbose:
                print("   Processing file: ", file_id)
            # All text ids that are in the file
            text_ids = mask_year["text_id"][mask_year["file_id"] == file_id]
            # A set of unique text ids that are not duplicates
            # it's a set cause it checks for membership in O(1) time :))
            # I run pandas' unique method firs tho, cause it runs in C or sth
            # So it's about 50 times as fast
            text_ids = set(text_ids.unique())
            for text in stream_by_text_id(
                os.path.join(data_path, f"{year}/{file_id}.jsonl"), text_ids=text_ids
            ):
                yield text


T = TypeVar("T")


@reusable
def reservoir_sample(stream: Iterable[T], sample_size: int) -> List[T]:
    """
    Samples a given number of items randomly from a stream of unknown length.
    An implementation of Algorithm R by Alan Waterman.

    Parameters
    ----------
    stream: Iterable[T]
        The stream to sample from.
    sample_size: int
        Number of items to sample.

    Returns
    ----------
    reservoir: List[T]
        Random sample from the stream.
    """
    reservoir = []
    for index, item in enumerate(stream):
        if index < sample_size:
            reservoir.append(item)
        else:
            j = random.randint(0, index)
            if j <= sample_size:
                reservoir[j] = item
    return reservoir


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
