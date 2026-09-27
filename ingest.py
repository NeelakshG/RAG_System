import argparse
from pathlib import Path

from src.chunkers import get_chunker
from src.indexer import build_index_version, build_indexes, publish_index_version
from src.loaders import load_file
from src.models import Chunk, Document

SUPPORTED_EXTENSIONS = {"md", "txt", "html", "docx", "pdf"}


def discover_files(corpus_dir: Path) -> list[Path]:
    """Return every file under corpus_dir (recursive) whose extension
    src.loaders.load_file actually supports (md, txt, html, docx, pdf).
    """
    return sorted(
        path
        for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lstrip(".") in SUPPORTED_EXTENSIONS
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


def ingest_and_publish(corpus_dir: Path, config, embedder) -> dict:
    """Production ingest: load -> chunk -> build both indexes into a NEW
    version directory -> atomically make it the live one. Unlike ingest(),
    this never rebuilds the live index in place, so it's safe to run while
    the API is serving queries.
    """
    docs = load_corpus(corpus_dir)
    chunks = chunk_corpus(docs, config.strategy, config, embedder)
    stats, version_dir = build_index_version(chunks, embedder, config)
    if version_dir is not None:
        publish_index_version(config, version_dir)
    return stats


def main() -> None:
    """CLI entry point: argparse --corpus-dir (default: config.corpus_dir),
    --strategy (default: config.strategy), builds + publishes a new index
    version, prints the stats dict. A running API keeps serving its old
    index until it's restarted or POST /v1/ingest is called.
    """
    from config import Config

    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path(config.corpus_dir))
    parser.add_argument(
        "--strategy", default=config.strategy, choices=["fixed", "recursive", "semantic"]
    )
    args = parser.parse_args()

    config = Config(strategy=args.strategy)
    stats = ingest_and_publish(args.corpus_dir, config, SentenceTransformerEmbedder(config.embedding_model))
    print(stats)


if __name__ == "__main__":
    main()
