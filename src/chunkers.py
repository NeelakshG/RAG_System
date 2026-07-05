from src.models import Document, Chunk
from src.tokenizer import find_token_spans


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


def get_chunker(strategy: str, config=None, embedder=None) -> Chunker:
    """Factory: returns the right Chunker subclass for the given strategy name."""
    if strategy == "fixed":
        size = getattr(config, "chunk_size", 256)
        overlap = getattr(config, "chunk_overlap", 32)
        return FixedChunker(size=size, overlap=overlap)
    raise NotImplementedError(f"chunking strategy not yet implemented: {strategy}")
