# Rule-Based POS Tagger for Technical Repair Transcripts

## Project overview
This mini-project builds a rule-based Part-of-Speech (POS) tagger for short technical repair transcripts. It combines:
1. An ordered `nltk.tag.RegexpTagger` pattern cascade.
2. A `DefaultTagger("NOUN")` fallback.
3. A contextual correction pass for ambiguous technical words such as `pump`, `wire`, `gauge`, and `torque`.
4. A manually annotated gold-standard set for evaluation.

The project does not train a machine-learning model. The annotated set is a small demonstration benchmark, not a representative industry dataset.

## Repository structure
```text
project-folder/
├── docs/
│   ├── Architecture_and_Methodology.pdf
│   └── Evaluation_Report.pdf
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── pipeline.py
│   └── evaluation_analysis.py
├── data/
│   ├── raw_transcripts/
│   │   └── sample_transcripts.txt
│   ├── annotated/
│   │   ├── gold_standard.jsonl
│   │   └── extended_test.jsonl
│   └── evaluation_analysis.json
├── requirements.txt
└── README.md
```

## Setup
Use Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the evaluation
From the project root:

```bash
python src/pipeline.py
```

Expected output includes the number of sentences and tokens evaluated, token accuracy, per-tag accuracy, and sentence-level errors.

### Run extended evaluation and rule-order sensitivity

```bash
python src/evaluation_analysis.py
```

This evaluates the additional annotated challenge set and compares four configurations: lexical-first (production), morphology-first, reversed lexical rule order, and reversed morphological rule order. Detailed results are written to `data/evaluation_analysis.json`. The additional challenge set is intentionally small and should not be described as a representative real-world benchmark.

## POS tag inventory
- `NOUN`: technical objects and things
- `VERB`: actions or commands
- `ADJ`: descriptive words
- `ADV`: manner or degree words
- `DET`: determiners/articles
- `PRON`: pronouns
- `ADP`: prepositions
- `AUX`: auxiliary/modal verbs
- `CONJ`: conjunctions
- `NUM`: numbers
- `PUNCT`: punctuation

## Evaluation definition
Token accuracy = correctly predicted tags / total gold-standard tokens.

## Important limitations
- Context rules are hand-written and can fail on unusual sentence structures.
- Suffix patterns are clues, not guarantees of grammatical role.
- The fallback tag `NOUN` can be wrong for unknown words.
- Rule order affects predictions.
- The dataset is small and manually annotated. For a stronger evaluation, annotate more real transcripts and report results on a held-out set.

## Deliverables
The `docs/` folder contains the architecture/methodology PDF and evaluation report PDF. Source code, sample transcripts, and gold annotations are included in their respective folders.


## Run the animated web application

The frontend is built with HTML, CSS, and JavaScript. Flask connects it to the existing Python/NLTK tagger.

```bash
pip install -r requirements.txt
python app.py
```

Open the local address printed by Flask (normally `http://127.0.0.1:5000`). Enter a repair sentence and select **Analyze sentence**. The interface animates each pipeline stage and then displays a result table containing the token, initial tag, rule source, final tag, and explanation.

The animation is a visual explanation of the implemented pipeline. The regex cascade contains lexical and suffix patterns, and the DefaultTagger is the cascade fallback; the UI separates these concepts into stages for learning purposes.


## How the implementation handles each stage

- **Regex-based tagging:** `src/models.py` contains an ordered `LEXICAL_PATTERNS` list and uses NLTK `RegexpTagger`.
- **Morphological rules:** `MORPHOLOGICAL_PATTERNS` contains suffix regex rules for `-ly`, `-ing`, `-ed`, and adjective endings. These run after lexical patterns in the actual cascade.
- **Default tagging:** `DefaultTagger("NOUN")` is the final fallback only if neither lexical nor morphological regex rules match.
- **Context rules:** `tag_tokens_with_trace()` makes a second pass and may revise selected ambiguous words using neighboring tokens.
- **Frontend trace:** The table reports the matched rule source, regex pattern (if any), initial tag, final tag, and contextual explanation. The pipeline cards visualize these actual per-token decisions.
