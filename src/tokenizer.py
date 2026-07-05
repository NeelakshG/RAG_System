import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list:
    """Split text into a list of word/punctuation tokens."""
    return _TOKEN_RE.findall(text)
    

def count_tokens(text: str) -> int:
    """Return the number of approximate tokens in text."""
    return len(tokenize(text))