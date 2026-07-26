import argparse
from pathlib import Path

from src.chunkers import get_chunker
from src.indexer import build_indexes
from src.loaders import load_file
from src.models import Chunk, Document

_SUPPORTED_EXTENSIONS = {"md", "txt", "html", "docx", "pdf"}


def discover_files(corpus_dir: Path) -> list[Path]:
    """Return every file under corpus_dir (recursive) whose extension
    src.loaders.load_file actually supports (md, txt, html, docx, pdf).
    """
    return sorted(
        path
        for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lstrip(".") in _SUPPORTED_EXTENSIONS
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every supported file under corpus_dir into Documents."""
    return [load_file(path, corpus_dir) for path in discover_files(corpus_dir)]


def chunk_corpus(
    docs: list[Document], strategy: str, config=None, embedder=None
) -> list[Chunk]:
    """Chunk every document with the given strategy, flatten into one list."""
    chunker = get_chunker(strategy, config, embedder)
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunker.chunk(doc))
    return chunks


class SentenceTransformerEmbedder:
    """Real embedder: wraps sentence-transformers BAAI/bge-small-en-v1.5.

    Exposes .embed(texts) -> list[list[float]], the same interface
    StubEmbedder fakes in tests. This is the only place in the codebase
    allowed to import sentence-transformers directly.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()


def ingest(corpus_dir: Path, strategy: str = "recursive", config=None, embedder=None) -> dict:
    """End-to-end: load_corpus -> chunk_corpus -> build_indexes.

    If embedder is None, a real SentenceTransformerEmbedder is constructed
    (import happens lazily inside its __init__, so passing a StubEmbedder
    here never touches sentence-transformers at all).
    """
    if embedder is None:
        embedder = SentenceTransformerEmbedder()

    docs = load_corpus(corpus_dir)
    chunks = chunk_corpus(docs, strategy, config, embedder)
    return build_indexes(chunks, embedder, config)


def main() -> None:
    """CLI entry point: argparse --corpus-dir (default data/corpus),
    --strategy (default 'recursive'), calls ingest(), prints the stats dict.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument(
        "--strategy", default="recursive", choices=["fixed", "recursive", "semantic"]
    )
    args = parser.parse_args()

    stats = ingest(args.corpus_dir, strategy=args.strategy)
    print(stats)


if __name__ == "__main__":
    main()
