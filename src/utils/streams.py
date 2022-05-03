import json
import multiprocessing
import os
import pickle
import random
import re
from itertools import islice, zip_longest
from typing import Callable, Iterable, List, Optional, Set, TypeVar

import numpy as np
import pandas as pd

import utils.text


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
        filled_length = int(BAR_LENGTH * iteration // total)
        progress_bar = FILL_CHARACTER * filled_length + "-" * (
            BAR_LENGTH - filled_length
        )
        os.system("clear")
        print(
            f"Progress: |{progress_bar}| {percent}% \n Current item processed: {item}\n"
        )
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


def stream_by_text_id(
    file_path: str, text_ids: Set[int], porn_domains: Optional[Set[str]] = None
) -> Iterable[str]:
    """
    Streams all texts from a jsonl file that are in the set of text_ids supplied.

    Parameters
    ----------
    file_path: str
        Path to the jsonl file in question
    text_ids: set of integer
        The ids to stream from the file.
    porn_domains: set of str or None, default None
        If provided, these domains will not be streamed.

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
                record_okay = (
                    record["passed_quality_filter"] and record["language"] == "da"
                )
                if porn_domains is not None:
                    record_okay = (
                        record_okay and record["domain_key"] not in porn_domains
                    )
                if record_okay:
                    yield record["text"]


@reusable
def stream_year(
    data_path: str,
    non_duplicates_path: str,
    year: str,
    porn_domains: Optional[Set[str]] = None,
) -> Iterable[str]:
    """
    Streams all texts from a given year.

    Parameters
    ----------
    data_path: str
        Path to the dataset
    non_duplicates_path: str
        Path, where non duplicate id numpy files are
    year: str
        The year from which texts should be streamed.
    porn_domains: set of str or None, default None
        If provided, these domains will not be streamed.

    Yields
    ----------
    text: str
        Text content from the given year.
    """
    for root, _, files in os.walk(os.path.join(non_duplicates_path, year)):
        for mask_file in files:
            if mask_file.endswith(".npy"):
                file_id = mask_file.split(".")[0]
                text_ids = set(np.load(os.path.join(root, mask_file)))
                for text in stream_by_text_id(
                    os.path.join(data_path, f"{year}/{file_id}.jsonl"),
                    text_ids=text_ids,
                    porn_domains=porn_domains,
                ):
                    yield text


def get_porn_domains(data_path: str) -> Set[str]:
    """
    Gives all domains that are suspected to be porn in the dataset.

    Parameters
    ----------
    data_path: str
        Path to the dataset

    Returns
    ----------
    unsafe_sites: set of str
        Set of unsafe sites (double checked)
    """
    file_path = os.path.join(data_path, "safe_search_domains_safe.pkl")
    with open(file_path, "rb") as f:
        safe_domains_dict = pickle.load(f)
    unsafe_double_checked = safe_domains_dict["unsafe_sites_double_checked"]
    return set(unsafe_double_checked)


@reusable
def stream_cleaned_texts(
    data_path: str, non_duplicates_path: str, filter_porn=True
) -> Iterable[str]:
    """
    Generator yielding all cleaned texts, that are not duplicates :)

    Parameters
    ----------
    data_path: str
        Specifies where our data lives, where to get file contents from.
    non_duplicates_path: str
        Path, where non duplicate id numpy files are
    filter_porn: bool, default True
        Specifies whether suspected porn domains should be left
        out of the stream.

    Yields
    ----------
    text: str
        Cleaned text
    """
    # In case we want to filter out porn, we load the set of unsafe sites from disk
    if filter_porn:
        porn_domains = get_porn_domains(data_path=data_path)
    else:
        porn_domains = None
    # List of all years
    years = get_years(data_path=data_path)
    # Collects streams of all years into a list
    year_streams = [
        stream_year(data_path, non_duplicates_path, year, porn_domains=porn_domains)
        for year in years
    ]
    # Streams texts from all years at the same time, so that the data is more shuffled
    # We use the zip_longest function from itertools, so that we iterate as
    # long as the longest iterable is not exhausted
    # Once a shorter iterable is exhausted, we will get None values.
    for texts in zip_longest(*year_streams, fillvalue=None):
        for text in texts:
            # If the text is not from an exhausted stream, we yield it
            if text is not None:
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


@reusable
def document_stream(
    texts: Iterable[str], chunksize: int = 2000, workers: int = 6
) -> Iterable[List[str]]:
    """
    Streams documents from the given text stream.

    Parameters
    ----------
    texts: Iterable of str
        Text stream to normalize and tokenize
    chunksize: int, default 2000
        Size of text chunks the stream should process in parallel
    workers: int, default 6
        Number of workers the stream should use to sentencize
        the texts coming from the stream
    """
    with multiprocessing.Pool(processes=workers) as pool:
        # We use imap_unordered, as the order of the documents does not matter for training
        # Doc2Vec, in fact it's better if they are shuffled
        # This stream is going to be chunked and sampled anyway
        docs = pool.imap_unordered(
            utils.text.normalized_document, texts, chunksize=chunksize
        )
        for doc in docs:
            yield doc


@reusable
def sentence_stream(
    texts: Iterable[str], window_size: int = 5, chunksize: int = 2000, workers: int = 6
) -> Iterable[List[str]]:
    """
    Streams sentences from the given text stream.

    Parameters
    ----------
    texts: Iterable of str
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
    sentence: list of str
        List of words in a sentence
    """
    with multiprocessing.Pool(processes=workers) as pool:
        docs = pool.imap_unordered(utils.text.sentences, texts, chunksize=chunksize)
        # We use imap_unordered, as the order of the sentences does not matter for training
        # Word2Vec, in fact it's better if they are shuffled
        # This stream is going to be chunked and sampled anyway
        for doc in docs:
            for sentence in doc:
                if len(sentence) >= window_size:
                    yield sentence
