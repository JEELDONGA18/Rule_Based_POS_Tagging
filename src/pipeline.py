"""Run evaluation of the rule-based POS tagger on annotated JSONL data."""
import json
from pathlib import Path
from collections import Counter
from preprocessing import tokenize
from models import tag_tokens

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "annotated" / "gold_standard.jsonl"

def evaluate(data_path: Path = DATA_PATH) -> dict:
    examples = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = correct = 0
    tag_counts = Counter()
    tag_correct = Counter()
    sentence_results = []

    for item in examples:
        tokens = item["tokens"]
        gold = item["tags"]
        predicted_pairs = tag_tokens(tokens)
        predicted = [tag for _, tag in predicted_pairs]
        if len(gold) != len(predicted):
            raise ValueError(f"Token/tag length mismatch for: {item['text']}")
        for g, p in zip(gold, predicted):
            total += 1
            tag_counts[g] += 1
            if g == p:
                correct += 1
                tag_correct[g] += 1
        sentence_results.append({
            "text": item["text"],
            "gold": gold,
            "predicted": predicted,
            "correct": gold == predicted,
        })

    per_tag = {
        tag: round(tag_correct[tag] / count, 3)
        for tag, count in sorted(tag_counts.items())
    }
    return {
        "sentences": len(examples),
        "tokens": total,
        "correct_tokens": correct,
        "accuracy": correct / total if total else 0.0,
        "per_tag_accuracy": per_tag,
        "results": sentence_results,
    }

if __name__ == "__main__":
    report = evaluate()
    print(f"Sentences evaluated: {report['sentences']}")
    print(f"Correct tokens: {report['correct_tokens']}/{report['tokens']}")
    print(f"Token accuracy: {report['accuracy']:.2%}")
    print("Per-tag accuracy:", report["per_tag_accuracy"])
    print("\\nSentence-level results:")
    for row in report["results"]:
        status = "PASS" if row["correct"] else "CHECK"
        print(f"[{status}] {row['text']}")
        if not row["correct"]:
            print("  Gold:     ", row["gold"])
            print("  Predicted:", row["predicted"])
