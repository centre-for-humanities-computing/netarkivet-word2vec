import string
from typing import List


def only_dots(s: str) -> str:
    """
    Exchanges all question marks and exclamation marks in the text for dots.
    Returns a new string.
    """
    return s.translate(str.maketrans({"?": ".", "!": ".", ";": "."}))


def sentencize(s: str) -> List[str]:
    """
    Sentencizes text. Returns a list of all sentences in the text.
    """
    s = only_dots(s)
    return s.split(".")


def remove_digits(s: str) -> str:
    """
    Removes all digits from the text.
    Returns a new string.
    """
    return s.translate(str.maketrans("", "", string.digits))


punct = "\"#$%&'()*+,-/:<=>@[\\]^_`{|}~"


def remove_punctuation(s: str) -> str:
    """
    Replaces all punctuation from the text with spaces
    except for dots, exclamation marks and question marks.
    Returns a new string.
    """
    return s.translate(str.maketrans(punct, " " * len(punct)))


def normalize(s: str) -> str:
    """
    Removes digits and punctuation from the text supplied.
    Returns new string.
    """
    s = remove_digits(s)
    s = remove_punctuation(s)
    return s


def tokenize(s: str) -> List[str]:
    """
    Tokenizes cleaned text.
    Returns a list of tokens.
    """
    return s.lower().split()


def sentences(text: str) -> List[List[str]]:
    """
    Cleans up the text, sentencizes and tokenizes it.
    Returns a list of sentences in the form of a list of tokens.
    """
    return [tokenize(sentence) for sentence in sentencize(normalize(text))]
