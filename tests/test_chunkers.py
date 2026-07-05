from src.chunkers import (
    _fixed_windows,
    _window_to_char_span,
    _split_by_headings,
    _split_paragraphs,
    _split_lines,
    _split_sentences,
    _split_words,
    _recursive_split,
    _merge_small_pieces,
    FixedChunker,
    RecursiveChunker,
    get_chunker,
)
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


def test_split_by_headings_basic():
    text = "# Title\nIntro text.\n\n## Sub\nMore text here."
    result = _split_by_headings(text)
    assert result == [
        ("Title", "# Title\nIntro text.\n\n"),
        ("Sub", "## Sub\nMore text here."),
    ]


def test_split_by_headings_with_preamble():
    text = "Some intro with no heading.\n\n# First Heading\nBody."
    result = _split_by_headings(text)
    assert result[0] == (None, "Some intro with no heading.\n\n")
    assert result[1] == ("First Heading", "# First Heading\nBody.")


def test_split_by_headings_no_headings_at_all():
    text = "Just plain text, no markdown headings."
    result = _split_by_headings(text)
    assert result == [(None, "Just plain text, no markdown headings.")]


def test_split_by_headings_deeper_levels():
    text = "# Title\nintro\n### Deep Sub\nbody"
    result = _split_by_headings(text)
    assert result == [
        ("Title", "# Title\nintro\n"),
        ("Deep Sub", "### Deep Sub\nbody"),
    ]


def test_split_paragraphs_reconstructs_exactly():
    text = "Para one.\n\nPara two.\n\nPara three."
    pieces = _split_paragraphs(text)
    assert "".join(pieces) == text
    assert pieces == ["Para one.\n\n", "Para two.\n\n", "Para three."]


def test_split_lines_reconstructs_exactly():
    text = "line one\nline two\nline three"
    pieces = _split_lines(text)
    assert "".join(pieces) == text
    assert pieces == ["line one\n", "line two\n", "line three"]


def test_split_sentences_reconstructs_exactly():
    text = "First sentence. Second sentence! Third?"
    pieces = _split_sentences(text)
    assert "".join(pieces) == text
    assert pieces == ["First sentence. ", "Second sentence! ", "Third?"]


def test_split_words_reconstructs_exactly():
    text = "one two  three"
    pieces = _split_words(text)
    assert "".join(pieces) == text
    assert pieces == ["one ", "two  ", "three"]


def test_recursive_split_returns_whole_text_if_already_small():
    text = "Short text."
    assert _recursive_split(text, max_tokens=100) == [text]


def test_recursive_split_splits_at_paragraph_level_when_needed():
    text = "One two three.\n\nFour five six."
    result = _recursive_split(text, max_tokens=4)
    assert result == ["One two three.\n\n", "Four five six."]


def test_recursive_split_falls_through_to_sentence_level():
    text = "Sentence one is here. Sentence two is here too. Sentence three ends it."
    result = _recursive_split(text, max_tokens=6)
    assert result == _split_sentences(text)


def test_recursive_split_gives_up_gracefully_on_unsplittable_text():
    text = "onewordonly"
    result = _recursive_split(text, max_tokens=0)
    assert result == [text]


def test_merge_small_pieces_combines_up_to_max_tokens():
    pieces = ["One. ", "Two. ", "Three. ", "Four. "]
    result = _merge_small_pieces(pieces, max_tokens=4)
    assert result == ["One. Two. ", "Three. Four. "]


def test_merge_small_pieces_preserves_oversized_leaf_standalone():
    pieces = ["short. ", "a,b,c,d,e,f,g,h", "short2."]
    result = _merge_small_pieces(pieces, max_tokens=5)
    assert result == ["short. ", "a,b,c,d,e,f,g,h", "short2."]


def test_merge_small_pieces_empty_list():
    assert _merge_small_pieces([], max_tokens=10) == []


def test_merge_small_pieces_single_piece():
    assert _merge_small_pieces(["only piece"], max_tokens=10) == ["only piece"]


def test_merge_small_pieces_preserves_reconstruction():
    pieces = ["One. ", "Two. ", "Three. ", "Four. "]
    result = _merge_small_pieces(pieces, max_tokens=4)
    assert "".join(result) == "".join(pieces)


def test_recursive_chunker_single_small_section():
    text = "# Title\nShort body."
    doc = _make_doc(text)
    chunks = RecursiveChunker(max_tokens=100).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].section_heading == "Title"
    assert chunks[0].chunking_strategy == "recursive"
    assert chunks[0].chunk_id == "doc1::0"


def test_recursive_chunker_tracks_correct_char_offsets():
    text = "# One\nBody one.\n\n# Two\nBody two."
    doc = _make_doc(text)
    chunks = RecursiveChunker(max_tokens=100).chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].section_heading == "One"
    assert chunks[1].section_heading == "Two"
    assert [c.chunk_index for c in chunks] == [0, 1]
    for c in chunks:
        assert doc.clean_text[c.start_char:c.end_char] == c.text


def test_recursive_chunker_splits_oversized_section_and_labels_all_pieces():
    text = (
        "# Big\n"
        "Sentence one is here. Sentence two is here too. Sentence three ends it."
    )
    doc = _make_doc(text)
    chunks = RecursiveChunker(max_tokens=6).chunk(doc)
    assert len(chunks) > 1
    assert all(c.section_heading == "Big" for c in chunks)
    assert all(c.chunking_strategy == "recursive" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert "".join(c.text for c in chunks) == doc.clean_text


def test_recursive_chunker_empty_document_produces_no_chunks():
    doc = _make_doc("")
    chunks = RecursiveChunker(max_tokens=100).chunk(doc)
    assert chunks == []


def test_get_chunker_returns_recursive_chunker():
    chunker = get_chunker("recursive")
    assert isinstance(chunker, RecursiveChunker)
