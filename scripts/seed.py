"""One-shot seed: generate the synthetic corpus and ingest it, so a fresh
clone (or a fresh docker-compose up) has something to ask about. Idempotent --
safe to run repeatedly; skips generation/ingestion once each has already run.

    python scripts/seed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from scripts.make_corpus import make_corpus


def main() -> None:
    config = Config()
    corpus_dir = Path("data/corpus")

    if corpus_dir.exists() and any(corpus_dir.iterdir()):
        print(f"Corpus already exists at {corpus_dir}, skipping generation.")
    else:
        print(f"Generating synthetic corpus at {corpus_dir}...")
        make_corpus(corpus_dir)

    if Path(config.bm25_path).exists():
        print(f"Index already exists at {config.bm25_path} (strategy={config.strategy}), skipping ingest.")
        return

    print(f"Ingesting corpus with strategy={config.strategy}...")
    from api.service import RAGService  # deferred: this is what loads the embedder

    service = RAGService(config)
    stats = service.ingest(str(corpus_dir))
    print("Done:", stats)


if __name__ == "__main__":
    main()
