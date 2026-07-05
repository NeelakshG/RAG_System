import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list:
    """Split text into a list of word/punctuation tokens."""
    return _TOKEN_RE.findall(text)
    

def count_tokens(text: str) -> int:
    """Return the number of approximate tokens in text."""
    return len(tokenize(text))


def find_token_spans(text: str) -> list:
    """Return (start_char, end_char) for each token, in order."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]