from src.chunkers import _fixed_windows, _window_to_char_span, FixedChunker, get_chunker
from src.models import Document
from src.tokenizer import find_token_spans


def _make_doc(text: str) -> Document:
    return Document(
        doc_id="doc1",
        source_path="corpus/doc1.md",
        source_name="doc1.md",
        fmt="md",
        raw_text=text,
        clean_text=text,
        metadata={},
    )


def test_fixed_windows_basic():
    windows = _fixed_windows(n_tokens=25, size=10, overlap=2)
    assert windows == [(0, 10), (8, 18), (16, 25)]


def test_fixed_windows_exact_fit():
    windows = _fixed_windows(n_tokens=20, size=10, overlap=0)
    assert windows == [(0, 10), (10, 20)]


def test_fixed_windows_shorter_than_one_window():
    windows = _fixed_windows(n_tokens=5, size=10, overlap=2)
    assert windows == [(0, 5)]


def test_window_to_char_span_matches_original_text():
    text = "The ERR_2043 error means the retry limit was exceeded, please check config."
    spans = find_token_spans(text)
    char_start, char_end = _window_to_char_span(spans, 4, 10)
    assert text[char_start:char_end] == "the retry limit was exceeded,"


def test_window_to_char_span_single_token():
    text = "hello world"
    spans = find_token_spans(text)
    char_start, char_end = _window_to_char_span(spans, 0, 1)
    assert text[char_start:char_end] == "hello"


def test_fixed_chunker_produces_overlapping_chunks_with_original_spacing():
    text = "The ERR_2043 error means the retry limit was exceeded, please check config."
    doc = _make_doc(text)
    chunker = FixedChunker(size=6, overlap=2)
    chunks = chunker.chunk(doc)

    assert [c.text for c in chunks] == [
        "The ERR_2043 error means the retry",
        "the retry limit was exceeded,",
        "exceeded, please check config.",
    ]
    assert [c.chunking_strategy for c in chunks] == ["fixed"] * 3
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert [c.chunk_id for c in chunks] == ["doc1::0", "doc1::1", "doc1::2"]
    assert all(c.section_heading is None for c in chunks)
    assert all(c.char_count == len(c.text) for c in chunks)


def test_fixed_chunker_empty_document_produces_no_chunks():
    doc = _make_doc("")
    chunks = FixedChunker(size=6, overlap=2).chunk(doc)
    assert chunks == []


def test_get_chunker_returns_fixed_chunker():
    chunker = get_chunker("fixed")
    assert isinstance(chunker, FixedChunker)


def test_get_chunker_unknown_strategy_raises():
    try:
        get_chunker("nonsense")
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
