from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from preprocessing import tokenize
from models import tag_tokens_with_trace

app = Flask(__name__)

def analyze_sentence(sentence: str):
    """Run the same rule implementation used by the evaluation pipeline."""
    clean_text = " ".join(sentence.strip().split())
    tokens = tokenize(clean_text)
    trace = tag_tokens_with_trace(tokens)

    return {
        "input": sentence,
        "preprocessed": clean_text,
        "tokens": tokens,
        "trace": trace,
        "results": [
            {"token": row["token"], "tag": row["final_tag"], "rule": row["source"]}
            for row in trace
        ],
        "stages": [
            {"id": "preprocess", "title": "Text Preprocessing", "description": "Clean extra spaces and prepare the sentence."},
            {"id": "tokenize", "title": "Tokenization", "description": "Split the sentence into words and punctuation."},
            {"id": "regex", "title": "Regex-Based Tagging", "description": "Apply explicit lexical regular-expression rules."},
            {"id": "morphology", "title": "Morphological Rules", "description": "Apply suffix regex rules after lexical rules."},
            {"id": "default", "title": "Default Tagger", "description": "Use DefaultTagger only when no regex rule matches."},
            {"id": "context", "title": "Context-Based Rules", "description": "Apply neighboring-word corrections to selected ambiguous words."},
            {"id": "results", "title": "Final POS Results", "description": "Display each token, its tag, and the decision that produced it."}
        ]
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/tag", methods=["POST"])
def api_tag():
    payload = request.get_json(silent=True) or {}
    sentence = str(payload.get("sentence", "")).strip()
    if not sentence:
        return jsonify({"error": "Please enter a technical repair sentence."}), 400
    if len(sentence) > 1000:
        return jsonify({"error": "Please keep the sentence under 1000 characters."}), 400
    return jsonify(analyze_sentence(sentence))

if __name__ == "__main__":
    app.run(debug=True)
