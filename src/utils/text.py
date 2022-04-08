import string
from typing import List


def only_dots(text: str) -> str:
    """
    Exchanges all question marks and exclamation marks in the text for dots.

    Parameters
    ----------
    text: str
        Text to alter

    Returns
    ----------
    text: str
        New string containing only dots
    """
    return text.translate(str.maketrans({"?": ".", "!": ".", ";": "."}))


def sentencize(text: str) -> List[str]:
    """
    Sentencizes text

    Parameters
    ----------
    text: str
        Text to sentencize

    Returns
    ----------
    sentences: List[str]
        List of sentence strings
    """
    text = only_dots(text)
    return text.split(".")


def remove_digits(text: str) -> str:
    """
    Removes all digits from the text.

    Parameters
    ----------
    text: str
        Text to alter

    Returns
    ----------
    text: str
        New string without digits
    """
    return text.translate(str.maketrans("", "", string.digits))


PUNCT = "\"#$%&'()*+,-/:<=>@[\\]^_`{|}~"


def remove_punctuation(text: str) -> str:
    """
    Replaces all punctuation from the text with spaces
    except for dots, exclamation marks and question marks.

    Parameters
    ----------
    text: str
        Text to alter

    Returns
    ----------
    text: str
        New string without punctuation
    """
    return text.translate(str.maketrans(PUNCT, " " * len(PUNCT)))


def normalize(text: str) -> str:
    """
    Removes digits and punctuation from the text supplied.

    Parameters
    ----------
    text: str
        Text to alter

    Returns
    ----------
    text: str
        New normalized string
    """
    text = remove_digits(text)
    text = remove_punctuation(text)
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenizes cleaned text.

    Parameters
    ----------
    text: str
        Text to tokenize

    Returns
    ----------
    tokens: List[str]
        List of tokens
    """
    return text.lower().split()


def sentences(text: str) -> List[List[str]]:
    """
    Cleans up the text, sentencizes and tokenizes it.

    Parameters
    ----------
    text: str
        Text to sentencize

    Returns
    ----------
    sentences: List[List[str]]
        List of sentences in the form of list of tokens.
    """
    return [tokenize(sentence) for sentence in sentencize(normalize(text))]
