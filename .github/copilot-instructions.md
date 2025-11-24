<!-- .github/copilot-instructions.md - Guidance for AI coding agents working on this repo -->
# Copilot instructions — chatbot_with_rules_ml

This file contains focused, actionable guidance for AI coding assistants (Copilot-style agents) to be immediately productive in this repository.

**Big Picture**
- **Pattern**: This project combines a lightweight rule engine with an on-start ML intent classifier. Rules are evaluated first; ML is used as a fallback or to produce intents when no rule triggers. See `src/data/chatbot_pipeline.py` for the control flow.
- **Major components**:
  - `src/data/rules_engine.py`: YAML-driven rules. Actions include `reply` (immediate text) and `set_intent` (force an intent for ML-response layer).
  - `src/data/ml_intent_classifier.py`: lightweight sklearn pipeline (TF-IDF + LogisticRegression). Trains from `src/data/training_data.csv` at startup.
  - `src/data/main.py`: small CLI runner demonstrating how the pieces are wired together.
  - `config.py`: `ChatbotConfig` with `MLConfig` (contains `model_path` and `min_confidence`). Note: the `model_path` value currently exists but model persistence is not implemented by `IntentClassifier`.

**How the pipeline works (important)**
- Startup: `Chatbot` constructs a `RuleEngine` from `src/data/rules.yml` and an `IntentClassifier` which calls `train_from_csv(training_data_path)` (training happens in memory every run).
- Message handling order: `RuleEngine.evaluate()` → if returns `reply` or `set_intent` act accordingly → otherwise `IntentClassifier.predict()` → if confidence < `ChatbotConfig.ml.min_confidence` return fallback message.
- See `ChatbotResponse.source` which is one of `rule`, `ml`, or `fallback` — use this to route logs or to add instrumentation.

**Project-specific conventions & notes**
- Rules format: `src/data/rules.yml` uses `contains_any` string-substring checks. Add new condition types by extending `RuleEngine.evaluate()` (example: regex, entity checks, or user session state).
- Training data: `src/data/training_data.csv` must contain `text,intent` columns. New training rows should be concise utterances mapped to one intent label.
- ML behavior: `IntentClassifier` trains on-the-fly (no persistence). If you add model saving/loading, prefer to use `config.MLConfig.model_path` and `joblib` for serialization.
- Relative imports: modules under `src/data` use package-relative imports (e.g., `from .ml_intent_classifier import IntentClassifier`). When running locally, ensure `src` is on `PYTHONPATH` (see Run section).

**Run / dev workflows (how-to)**
- Install dependencies (use a virtualenv):
  - `python -m venv .venv`
  - PowerShell: `.\.venv\Scripts\Activate.ps1`
  - `pip install -r requirements.txt`
- Run the CLI (from repository root in PowerShell):
  - `$env:PYTHONPATH = 'src'; python -m data.main`
  - This sets `src` on `PYTHONPATH` so package-relative imports inside `src/data` succeed.
- Quick alternative (explicit script): `python src/data/main.py` may fail because of package-relative imports. Prefer the `PYTHONPATH` + `-m data.main` approach.

**Code patterns to follow when editing**
- Keep rule actions simple and explicit: `reply` returns a text immediately; `set_intent` should only set an intent string used by the response layer.
- When extending intent handling, update `_respond_by_intent()` in `src/data/chatbot_pipeline.py` to keep intent→response mapping centralized.
- If adding new data files or models, place them under `src/data/` and reference via repo-relative paths in code (e.g., `src/data/training_data.csv`).

**Integration points and potential risks**
- External deps: scikit-learn, pandas, PyYAML. The classifier relies on scikit-learn's `predict_proba` interface.
- Thread-safety: current classifier and rule engine are in-memory and not thread-safe. If turning this into a long-running multi-threaded service, add locks or move training to an initialization step and ensure inference-only code is used in request threads.
- Model persistence is not implemented: `config.MLConfig.model_path` is present but unused — avoid relying on model files unless you implement save/load.

**Examples (copy-pasteable snippets)**
- Add a new rule that forces an intent:
```yaml
- name: refund_route
  conditions:
    contains_any: ["refund", "money back"]
  action:
    type: set_intent
    intent: "billing_issue"
```
- Example to run locally (PowerShell):
```powershell
$env:PYTHONPATH = 'src'
python -m data.main
```

**What to ask the repo owner if unclear**
- Should the ML model be persisted to `config.MLConfig.model_path`? (currently not implemented)
- Do you plan to expand rule condition types (regex, context, user state)? If yes, give examples of required operators.

---
If you want, I can implement model persistence (save/load), add `__init__.py` files to make `src` a package, or add unit tests for the rule engine and the classifier. Which should I do next?
