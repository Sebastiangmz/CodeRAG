#!/usr/bin/env python3
"""Evaluation script for CodeRAG Q&A system."""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coderag.config import get_settings
from coderag.generation.generator import ResponseGenerator
from coderag.models.query import Query


def load_eval_dataset(dataset_path: Path) -> List[Dict]:
    """Load evaluation questions from JSONL file."""
    questions = []
    with open(dataset_path, "r") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def evaluate_responses(
    generator: ResponseGenerator,
    questions: List[Dict],
    repo_id: str,
) -> Dict:
    """Evaluate responses for a set of questions."""
    results = {
        "total": len(questions),
        "grounded": 0,
        "with_citations": 0,
        "abstentions": 0,
        "avg_response_time": 0,
        "avg_citations_per_response": 0,
        "responses": [],
    }

    total_time = 0
    total_citations = 0

    for i, question_data in enumerate(questions, 1):
        question = question_data["question"]
        print(f"\n[{i}/{len(questions)}] Q: {question}")

        query = Query(
            question=question,
            repo_id=repo_id,
            top_k=5,
        )

        start_time = time.time()
        response = generator.generate(query)
        elapsed = time.time() - start_time

        total_time += elapsed
        num_citations = len(response.citations)
        total_citations += num_citations

        if response.grounded:
            results["grounded"] += 1
        if num_citations > 0:
            results["with_citations"] += 1
        if not response.has_evidence or "could not find" in response.answer.lower():
            results["abstentions"] += 1

        result = {
            "question": question,
            "answer": response.answer,
            "citations": [str(c) for c in response.citations],
            "grounded": response.grounded,
            "num_chunks": len(response.retrieved_chunks),
            "response_time": round(elapsed, 2),
        }
        results["responses"].append(result)

        print(f"  A: {response.answer[:100]}...")
        print(f"  Citations: {num_citations} | Grounded: {response.grounded} | Time: {elapsed:.2f}s")

    results["avg_response_time"] = total_time / len(questions)
    results["avg_citations_per_response"] = total_citations / len(questions)
    results["grounded_rate"] = results["grounded"] / results["total"]
    results["citation_rate"] = results["with_citations"] / results["total"]
    results["abstention_rate"] = results["abstentions"] / results["total"]

    return results


def print_summary(results: Dict):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total questions:          {results['total']}")
    print(f"Grounded responses:       {results['grounded']} ({results['grounded_rate']:.1%})")
    print(f"With citations:           {results['with_citations']} ({results['citation_rate']:.1%})")
    print(f"Abstentions:              {results['abstentions']} ({results['abstention_rate']:.1%})")
    print(f"Avg response time:        {results['avg_response_time']:.2f}s")
    print(f"Avg citations/response:   {results['avg_citations_per_response']:.1f}")
    print("=" * 60)


def main():
    """Run evaluation on the CodeRAG system itself."""
    print("CodeRAG Evaluation")
    print("=" * 60)

    # Check if models are available
    try:
        print("\nInitializing generator (this may take a while to load models)...")
        generator = ResponseGenerator()
        print("✓ Generator initialized")
    except Exception as e:
        print(f"\n❌ Failed to initialize generator: {e}")
        print("\nℹ️  Make sure models are downloaded first:")
        print("   python scripts/download_models.py")
        return 1

    # For this demo, we'll index the CodeRAG repo itself
    repo_id = "coderag-self"  # You need to index this first
    print(f"\nUsing repository: {repo_id}")
    print("⚠️  Make sure you've indexed a repository first via the UI or API")

    # Load evaluation datasets
    eval_dir = Path("eval_datasets")
    datasets = {
        "closed": eval_dir / "closed_questions.jsonl",
        "open": eval_dir / "open_questions.jsonl",
    }

    all_results = {}

    for name, dataset_path in datasets.items():
        if not dataset_path.exists():
            print(f"\n⚠️  Skipping {name}: {dataset_path} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name.upper()} Questions")
        print(f"{'=' * 60}")

        questions = load_eval_dataset(dataset_path)
        results = evaluate_responses(generator, questions, repo_id)
        all_results[name] = results

        print_summary(results)

        # Save results
        output_path = Path(f"eval_results_{name}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
