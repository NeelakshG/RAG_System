"""Tests for src/tokenizer.py."""
from src.tokenizer import tokenize, count_tokens


def test_tokenize_basic():
    assert tokenize("ERR_2043, retry?") == ["ERR_2043", ",", "retry", "?"]


def test_tokenize_empty():
    assert tokenize("") == []


def test_count_tokens_matches_tokenize_length():
    text = "config_key=max_retries (default: 3)"
    assert count_tokens(text) == len(tokenize(text))


def test_punctuation_is_split_out():
    assert tokenize("e.g.") == ["e", ".", "g", "."]


def test_whitespace_is_ignored():
    assert tokenize("hello   world\n\tfoo") == ["hello", "world", "foo"]


def test_count_tokens_whitespace_only():
    assert count_tokens("   \n\t  ") == 0


def test_underscored_identifier_is_one_token():
    assert tokenize("northwind_sdk") == ["northwind_sdk"]
