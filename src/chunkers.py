import re

from src.models import Document, Chunk
from src.tokenizer import find_token_spans, count_tokens

_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


def _fixed_windows(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) token-index pairs for a sliding window over n_tokens."""
    step = size - overlap  # how far the window advances each iteration; the `overlap`
                            # tokens at the end of one window reappear at the start of the next
    windows = []            # collects the (start, end) pairs we produce
    start = 0               # token index where the current window begins

    while start < n_tokens:                 # keep going until we've started a window that reaches the tail
        end = min(start + size, n_tokens)    # window is `size` tokens wide, but never runs past n_tokens
        windows.append((start, end))         # record this window

        if end == n_tokens:                  # this window already reached the last token
            break                            # stop -- advancing further would only produce a smaller,
                                              # redundant window fully contained in this one's tail

        start += step                        # move the window forward by the step size for the next pass

    return windows


def _window_to_char_span(spans: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    """Convert a token-index window (start, end) into a (char_start, char_end) span."""
    char_start = spans[start][0]
    char_end = spans[end - 1][1]
    return char_start, char_end


class Chunker:
    """Shared interface every chunking strategy implements."""

    def chunk(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError


class FixedChunker(Chunker):
    """Fixed-size sliding token window with overlap."""

    def __init__(self, size: int = 256, overlap: int = 32):
        self.size = size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.clean_text
        spans = find_token_spans(text)          # per-token (start_char, end_char)
        windows = _fixed_windows(len(spans), self.size, self.overlap)

        chunks = []
        for i, (start, end) in enumerate(windows):
            char_start, char_end = _window_to_char_span(spans, start, end)
            chunk_text = text[char_start:char_end]   # sliced from the ORIGINAL string
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{i}",
                    doc_id=doc.doc_id,
                    source_name=doc.source_name,
                    chunk_index=i,
                    text=chunk_text,
                    section_heading=None,   # fixed-size chunking doesn't track structure
                    chunking_strategy="fixed",
                    char_count=len(chunk_text),
                    token_count=end - start,
                    start_char=char_start,
                    end_char=char_end,
                )
            )
        return chunks


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split text into (heading_text_or_None, section_text) at each markdown heading line."""
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        # no headings anywhere -- the whole text is one unlabeled section
        return [(None, text)]

    sections = []

    if matches[0].start() > 0:
        # text before the first heading belongs to no section
        preamble = text[: matches[0].start()]
        sections.append((None, preamble))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end]                       # includes the heading line itself
        heading_text = re.sub(r"^#{1,6}\s+", "", match.group()).strip()
        sections.append((heading_text, section_text))

    return sections


def _split_keep_delimiter(text: str, pattern: str) -> list[str]:
    """Split text on pattern, reattaching each delimiter to the preceding piece.

    Guarantees "".join(result) == text.
    """
    parts = re.split(f"({pattern})", text)  # capturing group keeps delimiters in the output

    pieces = []
    for i in range(0, len(parts), 2):        # walk pairwise: piece, delimiter, piece, delimiter, ...
        piece = parts[i]
        if i + 1 < len(parts):                # a delimiter followed this piece -- glue it on
            piece += parts[i + 1]
        pieces.append(piece)

    return [p for p in pieces if p]           # drop empty strings (e.g. trailing delimiter at end of text)


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs at blank lines; join(pieces) == text exactly."""
    return _split_keep_delimiter(text, r"\n\s*\n")


def _split_lines(text: str) -> list[str]:
    """Split text into lines; join(pieces) == text exactly."""
    return text.splitlines(keepends=True)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? boundaries; join(pieces) == text exactly."""
    return _split_keep_delimiter(text, r"(?<=[.!?])\s+")


def _split_words(text: str) -> list[str]:
    """Split text on whitespace; join(pieces) == text exactly."""
    return _split_keep_delimiter(text, r"\s+")


_RECURSIVE_LEVELS = [_split_paragraphs, _split_lines, _split_sentences, _split_words]


def _recursive_split(text: str, max_tokens: int, levels: list = None) -> list[str]:
    """Recursively split text via the separator hierarchy until every piece fits max_tokens."""
    if levels is None:
        levels = _RECURSIVE_LEVELS

    if count_tokens(text) <= max_tokens:
        # already small enough -- nothing to split
        return [text]

    if not levels:
        # ran out of separators to try; this piece can't be split any further
        return [text]

    pieces = levels[0](text)

    if len(pieces) <= 1:
        # this separator didn't actually break the text apart -- try the next level down
        return _recursive_split(text, max_tokens, levels[1:])

    result = []
    for piece in pieces:
        result.extend(_recursive_split(piece, max_tokens, levels[1:]))
    return result


def _merge_small_pieces(pieces: list[str], max_tokens: int) -> list[str]:
    """Greedily merge adjacent pieces up to max_tokens; never split a piece further."""
    if not pieces:
        return []

    result = []
    current = pieces[0]              # accumulator for the chunk being built up

    for piece in pieces[1:]:
        merged = current + piece
        if count_tokens(merged) <= max_tokens:
            current = merged          # still fits -- keep growing this chunk
        else:
            result.append(current)    # accumulator is full -- finalize it
            current = piece            # start a new accumulator with this piece

    result.append(current)             # flush whatever's left over
    return result


class RecursiveChunker(Chunker):
    """Structure-aware chunker: splits on headings, then recurses through
    paragraphs -> lines -> sentences -> words only where a section is too big,
    then merges small adjacent pieces back up toward max_tokens.
    """

    def __init__(self, max_tokens: int = 256):
        self.max_tokens = max_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.clean_text
        sections = _split_by_headings(text)

        chunks = []
        offset = 0    # running char position -- sections/pieces are contiguous and gapless
        index = 0     # running chunk index across the WHOLE document, not per-section

        for heading, section_text in sections:
            leaves = _recursive_split(section_text, self.max_tokens)
            merged = _merge_small_pieces(leaves, self.max_tokens)

            for piece in merged:
                if not piece:
                    continue  # skip empty pieces (e.g. an entirely empty document)

                char_start = offset
                char_end = offset + len(piece)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::{index}",
                        doc_id=doc.doc_id,
                        source_name=doc.source_name,
                        chunk_index=index,
                        text=piece,
                        section_heading=heading,
                        chunking_strategy="recursive",
                        char_count=len(piece),
                        token_count=count_tokens(piece),
                        start_char=char_start,
                        end_char=char_end,
                    )
                )
                offset = char_end
                index += 1

        return chunks


def get_chunker(strategy: str, config=None, embedder=None) -> Chunker:
    """Factory: returns the right Chunker subclass for the given strategy name."""
    if strategy == "fixed":
        size = getattr(config, "chunk_size", 256)
        overlap = getattr(config, "chunk_overlap", 32)
        return FixedChunker(size=size, overlap=overlap)
    if strategy == "recursive":
        max_tokens = getattr(config, "max_tokens", 256)
        return RecursiveChunker(max_tokens=max_tokens)
    raise NotImplementedError(f"chunking strategy not yet implemented: {strategy}")
