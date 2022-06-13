""" Module containing several useful streaming tools, both general and project related. """
import functools
import json
import multiprocessing
import os
import pickle
import random
import re
from itertools import islice, zip_longest
from typing import Callable, Iterable, List, Optional, Set, TypeVar

from deprecated import deprecated
from gensim.models.doc2vec import TaggedDocument

import utils.text
from utils.notebook import is_notebook
from utils.topic import PornClassifier

if is_notebook():
    print("Code running in a notebook, loading display tools")
    from IPython.display import clear_output

T = TypeVar("T")


def flatten(nested: Iterable[Iterable[T]]) -> Iterable[T]:
    """
    Function that turns a nested stream into a flat stream.

    Parameters
    ----------
    nested: iterable of iterable of T
        Nested iterable that you want to flatten

    Yields
    ----------
    element: T
        Individual elements of the nested iterable
    """
    for sub in nested:
        for elem in sub:
            yield elem


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

    @functools.wraps(gen_func, updated=())
    class _multigen:
        def __init__(self, *args, limit=None, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
            self.limit = limit
            # functools.update_wrapper(self, gen_func)

        def __iter__(self):
            if self.limit is not None:
                return islice(gen_func(*self.__args, **self.__kwargs), self.limit)
            return gen_func(*self.__args, **self.__kwargs)

    return _multigen


U = TypeVar("U")


@reusable
def chunk(
    iterable: Iterable[U], chunk_size: int, sample_size: Optional[int] = None
) -> Iterable[List[U]]:
    """
    Generator function that chunks an iterable for you.

    Parameters
    ----------
    iterable: Iterable of T
        The iterable you'd like to chunk.
    chunk_size: int
        The size of chunks you would like to get back
    sample_size: int or None, default None
        If specified the yielded lists will be randomly sampled with the buffer
        with replacement. Sample size determines how big you want those lists to be.

    Yields
    ----------
    buffer: list of T
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

I = TypeVar("I")


@reusable
def progress_bar_stream(items: List[I]) -> Iterable[I]:
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
        if is_notebook():
            clear_output(wait=True)
        else:
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
    years: list of str
        List of all years processed
    """
    years = []
    for root, dirs, files in os.walk(data_path):
        for directory in dirs:
            if re.match(r"\d\d\d\d", directory):
                years.append(directory)
    return years


@deprecated(
    """
    Phased out in favor of streaming records, as it provides better separation
    of concerns and we can extract valuable information at a higher level.
    """
)
def stream_file(
    file_path: str, porn_domains: Optional[Set[str]] = None
) -> Iterable[str]:
    """
    Streams all texts from a jsonl file.

    Parameters
    ----------
    file_path: str
        Path to the jsonl file in question
    porn_domains: set of str or None, default None
        If provided, these domains will not be streamed.

    Yields
    ----------
    text: str
        Text content of the given fields
    """
    with open(file_path) as input_file:
        # Since id is not one of the fields, I have to enumerate all records
        for line in input_file:
            # parsing the record
            record = json.loads(line)
            # If passes quality filters, it yields the content of the record
            record_okay = (
                record["passed_quality_filter"]
                and record["language"] == "da"
                and not record["is_duplicate"]
            )
            if porn_domains is not None:
                record_okay = record_okay and (record["domain_key"] not in porn_domains)
            if record_okay:
                yield record["text"]


def stream_records_from_file(file_path: str) -> Iterable[dict]:
    """
    Streams all records from the file at the given path that have passed
    the quality filters, are in Danish and aren't duplicates

    Parameters
    ----------
    file_path: str
        Path to the file you'd like to stream

    Yields
    ----------
    record: dict
        Each record from the file
    """
    with open(file_path) as input_file:
        # Since id is not one of the fields, I have to enumerate all records
        for line in input_file:
            # parsing the record
            record = json.loads(line)
            # If passes quality filters, it yields the content of the record
            record_okay = (
                record["passed_quality_filter"]
                and record["language"] == "da"
                and not record["is_duplicate"]
            )
            if record_okay:
                yield record


@reusable
def stream_year(
    data_path: str,
    year: str,
) -> Iterable[dict]:
    """
    Streams all records from a given year.

    Parameters
    ----------
    data_path: str
        Path to the dataset
    year: str
        The year from which records should be streamed.

    Yields
    ----------
    record: dict
        Records from the given year.
    """
    for root, _, files in os.walk(os.path.join(data_path, f"{year}")):
        # Go through all files in the year directory
        for file in files:
            # If it's a jsonl file, stream all records from it
            if file.endswith(".jsonl"):
                records = stream_records_from_file(os.path.join(root, file))
                for record in records:
                    yield record


@reusable
def to_text_stream(records: Iterable[dict]) -> Iterable[str]:
    """
    Turns a stream of records to a stream of texts

    Parameters
    ----------
    records: iterable of dict
        Stream of records you want to turn into texts

    Yields
    ----------
    text: str
        Texts extracted from the records
    """
    for record in records:
        yield record["text"]


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
    # Soo we got this pickle file containing a dict of some random stuff
    file_path = os.path.join(data_path, "safe_search_domains_safe.pkl")
    with open(file_path, "rb") as f:
        # And then we load it
        safe_domains_dict = pickle.load(f)
    # This is it's rough structure:
    # {
    #    'cleanweb': dict[str, bool]
    #       contains whether a certain website is clean or not
    #    'google public DNS': dict[str, bool]
    #    'unsafe_sites': list[str]
    #       Compares cleanweb to public DNS, but lot of false positives
    #    'unsafe_sites_double_checked': list[str]
    #       Same thing with higher tolerance
    # }
    unsafe_double_checked = safe_domains_dict["unsafe_sites_double_checked"]
    # We turn this to a set, cause I love them unreasonably much,
    # but then yeah O(1) access time and stuff
    return set(unsafe_double_checked)


def filter_porn_domains(records: Iterable[dict], domains: Set[str]) -> Iterable[dict]:
    """
    Filters away all the porn domains from a given stream of records

    Parameters
    ----------
    records: iterable of dict
        Stream of records to be filtered
    domains: set of str
        Set of porn domains you want to filter from the records

    Yields
    ----------
    record: dict
        A record that does not belong to one of the porn domains
    """
    for record in records:
        if record["domain_key"] not in domains:
            yield record


DEFAULT_TOPIC_MODEL = "nmf_100"
TOPIC_MODEL_PATH = "/work/topic_model/"
PORN_THRESHOLD = 0.0002


def filter_porn_records(
    records: Iterable[dict], chunk_size: int = 150_000
) -> Iterable[dict]:
    """
    Filters porn based on a precomputed NMF topic model.

    Parameters
    ----------
    records: iterable of dict
        Stream of records to be filtered
    chunk_size: int, default 150_000
        Chunk size of records that the model operates on.

    Yields
    ----------
    record: dict
        A record that doesn't contain porn texts

    Notes
    ----------
    Since optimized linear algebra is involved,
    you should set the chunk size to be reasonably high.

    Operating on one text at a time would be really slow.
    """
    record_chunks = chunk(records, chunk_size)
    # Loading topic model and vectorizer
    cls = PornClassifier.load(DEFAULT_TOPIC_MODEL, TOPIC_MODEL_PATH, PORN_THRESHOLD)
    for record_chunk in record_chunks:
        texts = to_text_stream(record_chunk)
        # Running porn detection for a chunk of records
        predictions = cls.predict(texts)
        for is_porn, record in zip(predictions, record_chunk):
            if not is_porn:
                yield record


@reusable
def stream_all_records(data_path: str) -> Iterable[dict]:
    """
    Generator yielding all records from the dataset.

    Parameters
    ----------
    data_path: str
        Specifies where our data lives, where to get file contents from.
    Yields
    ----------
    record: dict
        All records
    """
    # List of all years
    years = get_years(data_path=data_path)
    # Collects streams of all years into a list
    year_streams = [stream_year(data_path, year) for year in years]
    # Streams records from all years at the same time, so that the data is more shuffled
    # We use the zip_longest function from itertools, so that we iterate as
    # long as the longest iterable is not exhausted
    # Once a shorter iterable is exhausted, we will get None values.
    for records in zip_longest(*year_streams, fillvalue=None):
        for record in records:
            # If the record is not from an exhausted stream, we yield it
            if record is not None:
                yield record


V = TypeVar("V")


def reservoir_sample(stream: Iterable[V], sample_size: int) -> List[V]:
    """
    Samples a given number of items randomly from a stream of unknown length.
    An implementation of Algorithm R by Alan Waterman.

    Parameters
    ----------
    stream: Iterable of T
        The stream to sample from.
    sample_size: int
        Number of items to sample.

    Returns
    ----------
    reservoir: list of T
        Random sample from the stream.
    """
    reservoir = []
    for index, item in enumerate(stream):
        if index < sample_size:
            reservoir.append(item)
        else:
            j = random.randint(0, index)
            if j < sample_size:
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
        Number of workers the stream should use to normalize
        the texts coming from the stream

    Yields
    ----------
    doc: List of str
        List of words in a given document
    """
    with multiprocessing.Pool(processes=workers) as pool:
        # We use imap_unordered, as the order of the documents does not matter for training
        # Doc2Vec, in fact it's better if they are shuffled
        # This stream is going to be chunked and sampled anyway
        docs = pool.imap_unordered(
            utils.text.normalized_document, texts, chunksize=chunksize
        )
        for doc in docs:
            if doc:
                yield doc


@reusable
def tag_documents(
    doc_stream: Iterable[List[str]], save_vectors: bool = False
) -> Iterable[TaggedDocument]:
    """
    Turns a stream of documents to a stream of TaggedDocuments.

    Parameters
    ----------
    doc_stream: iterable of list of str
        Stream of documents in the form of list of words.
    save_vectors: bool, default False
        Specifies whether the document vectors should be persisted to disk.
        If False, all documents have the same tag.

    Yields
    ----------
    doc: TaggedDocument
        Document with tag added

    Notes
    ----------
    For large corpora like Netarkivet, it does make sense to only save
    model parameters but not the document vectors.
    Document vectors can be inferred at will once the model is trained, thus
    there is no need to store them. They also take an increasingly large amount of
    space on the disk, which increases model loading times and takes up valuable disk space.
    """
    for tag, doc in enumerate(doc_stream):
        yield TaggedDocument(words=doc, tags=[tag] if save_vectors else [0])


@reusable
def sentence_stream(
    texts: Iterable[str], chunksize: int = 2000, workers: int = 6
) -> Iterable[List[str]]:
    """
    Streams sentences from the given text stream.

    Parameters
    ----------
    texts: Iterable of str
        Text stream to sentencize
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
        docs = pool.imap_unordered(utils.text.sentencize, texts, chunksize=chunksize)
        # We use imap_unordered, as the order of the sentences does not matter for training
        # Word2Vec, in fact it's better if they are shuffled
        # This stream is going to be chunked and sampled anyway
        for doc in docs:
            if doc:
                for sentence in doc:
                    yield sentence
