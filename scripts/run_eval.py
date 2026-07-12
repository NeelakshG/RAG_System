import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from ingest import SentenceTransformerEmbedder, chunk_corpus, load_corpus
from src.eval import aggregate, load_golden_questions, run_eval
from src.indexer import DenseIndex, SparseIndex, build_indexes
from src.llm import OllamaClient
from src.retriever import CrossEncoderReranker

STRATEGIES = ["fixed", "recursive", "semantic"]


def _mean_tokens(chunks) -> float:
    return sum(c.token_count for c in chunks) / len(chunks) if chunks else 0.0


def run_strategy(strategy: str, corpus_dir: Path, embedder, reranker, client) -> dict:
    """Ingest the corpus with `strategy`, run the golden set against it, and
    return the aggregate scores plus chunking-shape stats that explain them.
    """
    config = Config(strategy=strategy)

    docs = load_corpus(corpus_dir)
    chunks = chunk_corpus(docs, strategy, config, embedder)
    index_stats = build_indexes(chunks, embedder, config)

    dense_index = DenseIndex(persist_dir=config.chroma_persist_dir, collection_name=config.collection_name)
    sparse_index = SparseIndex.load(config.bm25_path)

    questions = load_golden_questions(str(Path(__file__).parent.parent / "golden_qa.json"))
    results = run_eval(questions, strategy, dense_index, sparse_index, embedder, reranker, client, config)

    return {
        "strategy": strategy,
        "chunks_produced": len(chunks),
        "chunks_indexed": index_stats["indexed"],
        "chunks_deduped": index_stats["deduped"],
        "mean_chunk_tokens": round(_mean_tokens(chunks), 1),
        "scores": aggregate(results),
    }


def print_comparison_table(summaries: list[dict]) -> None:
    metrics = ["correctness", "faithfulness", "retrieval_relevance", "citation_accuracy", "fallback_rate"]
    header = ["strategy", "chunks", "deduped", "mean_tokens"] + metrics
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for s in summaries:
        overall = s["scores"]["overall"]
        row = [
            s["strategy"],
            str(s["chunks_indexed"]),
            str(s["chunks_deduped"]),
            str(s["mean_chunk_tokens"]),
        ] + [
            f"{overall[m]:.2f}" if overall[m] is not None else "n/a" for m in metrics
        ]
        print(" | ".join(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument("--output", type=Path, default=Path("data/eval/comparison.json"))
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder()
    reranker = CrossEncoderReranker()
    client = OllamaClient()

    summaries = [
        run_strategy(strategy, args.corpus_dir, embedder, reranker, client)
        for strategy in args.strategies
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2))

    print_comparison_table(summaries)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
