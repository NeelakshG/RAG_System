"""Record real answers to the demo's example questions, so the public demo
can show them instantly and keep showing them even when the free LLM quota
is used up. Run once locally (with Ollama, or with Groq configured), then
commit the output:

    python scripts/record_demo_answers.py
    git add dashboard/demo_answers.json dashboard/eval_comparison.json

Also copies data/eval/comparison.json (from scripts/run_eval.py), if it
exists, to dashboard/eval_comparison.json for the demo's Eval tab.
"""

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dashboard.backend import DEMO_ANSWERS_PATH, EXAMPLE_QUESTIONS, build_embedded_service


def main() -> None:
    service = build_embedded_service(REPO_ROOT)
    records = []
    for question in EXAMPLE_QUESTIONS:
        print(f"Asking: {question}")
        result = service.ask(question, use_hybrid=True)
        print(f"  -> {result['answer'][:100]}")
        records.append({"question": question, "use_hybrid": True, "result": result})

    DEMO_ANSWERS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} answers to {DEMO_ANSWERS_PATH}")

    comparison = REPO_ROOT / "data" / "eval" / "comparison.json"
    if comparison.exists():
        target = REPO_ROOT / "dashboard" / "eval_comparison.json"
        shutil.copyfile(comparison, target)
        print(f"Copied eval results to {target}")
    else:
        print("No data/eval/comparison.json yet (run scripts/run_eval.py to add eval results to the demo).")


if __name__ == "__main__":
    main()
