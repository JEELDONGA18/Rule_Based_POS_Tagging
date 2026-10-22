"""Extended test evaluation and rule-order sensitivity experiment."""
import json
from collections import Counter
from pathlib import Path
from preprocessing import tokenize
from models import tag_tokens

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "annotated" / "extended_test.jsonl"
ORDERS = ["lexical_first", "morphology_first", "reverse_lexical", "reverse_morphological"]


def load_examples():
    return [json.loads(line) for line in TEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_order(examples, order):
    total = correct = 0
    tag_total, tag_correct = Counter(), Counter()
    errors = []
    for item in examples:
        tokens = item["tokens"]
        gold = item["tags"]
        predicted = [tag for _, tag in tag_tokens(tokens, rule_order=order)]
        if len(gold) != len(predicted):
            raise ValueError(f"Token/tag length mismatch in: {item['text']}")
        for i, (g, p) in enumerate(zip(gold, predicted)):
            total += 1
            tag_total[g] += 1
            if g == p:
                correct += 1
                tag_correct[g] += 1
            else:
                errors.append({"sentence": item["text"], "token": tokens[i], "gold": g, "predicted": p})
    return {
        "rule_order": order,
        "correct_tokens": correct,
        "total_tokens": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "per_tag_accuracy": {tag: round(tag_correct[tag] / count, 3) for tag, count in sorted(tag_total.items())},
        "errors": errors,
    }


def main():
    extended = load_examples()
    original_path = ROOT / "data" / "annotated" / "gold_standard.jsonl"
    original = [json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    combined = original + extended

    baseline_extended = evaluate_order(extended, "lexical_first")
    print(f"Extended challenge set: {len(extended)} sentences, {baseline_extended['total_tokens']} tokens")
    print(f"Production rule order accuracy: {baseline_extended['correct_tokens']}/{baseline_extended['total_tokens']} = {baseline_extended['accuracy']:.2%}")
    print("Extended-set errors:")
    for err in baseline_extended["errors"]:
        print(f" - {err['token']!r} in {err['sentence']!r}: gold={err['gold']}, predicted={err['predicted']}")
    print("\nNOTE: this is a small manually annotated challenge set, not a representative real-world benchmark.")

    print(f"\nRule-order sensitivity across original + extended sets ({len(combined)} sentences):")
    sensitivity_results = [evaluate_order(combined, order) for order in ORDERS]
    for result in sensitivity_results:
        print(f"{result['rule_order']}: {result['correct_tokens']}/{result['total_tokens']} = {result['accuracy']:.2%}")
    print("\nTag changes compared with lexical_first:")
    for order in ORDERS[1:]:
        diffs = []
        for item in combined:
            tokens = item["tokens"]
            base = [tag for _, tag in tag_tokens(tokens, rule_order="lexical_first")]
            other = [tag for _, tag in tag_tokens(tokens, rule_order=order)]
            for token, b, o in zip(tokens, base, other):
                if b != o:
                    diffs.append(f"{token} ({item['text']}): {b}->{o}")
        print(f"  {order}: {len(diffs)} token tag change(s)" + (f"; {'; '.join(diffs)}" if diffs else ""))

    output = ROOT / "data" / "evaluation_analysis.json"
    output.write_text(json.dumps({
        "extended_set": baseline_extended,
        "sensitivity_dataset_sentences": len(combined),
        "sensitivity_results": sensitivity_results,
    }, indent=2), encoding="utf-8")
    print(f"\nDetailed results saved to: {output.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
