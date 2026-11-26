# Mental Health LLM Error Analysis (105-Sample Benchmark)

## 1. Overview / Objective
This repository investigates failure modes when classifying mental health-related text using locally hosted open-weight Large Language Models (LLMs) via Ollama. Across a stratified sample of 105 texts (15 samples per class across 7 mental health categories), we evaluate exact-label accuracy, refusal/safety-guardrail behaviour, output format adherence, and class confusion patterns.

## 2. Dataset & Model Setup
- **Dataset**: Stratified sample of **105 statements** (15 per class x 7 classes: `Normal`, `Depression`, `Suicidal`, `Anxiety`, `Stress`, `Bipolar`, `Personality disorder`) drawn (`random_state=42`) from `Combined Data.csv` after removing missing values, empty strings, and duplicate statements. Raw dataset files are **not** stored in this repository.
- **Models**: `llama3.2:3b` and `qwen2.5:7b` served locally via Ollama (`http://localhost:11434/api/generate`).
- **Prompt & Inference Condition**: Zero-shot single-label classification prompt (`temperature=0`, `stream=False`, truncating input text to the first 1,000 characters) instructing the model to reply with *only* the exact target label name.

## 3. Error Taxonomy & Quantitative Summary
Errors (N=105 total per model) are categorised into four mutually exclusive types:
1. **Refusal / Over-cautious**: Model declines classification or outputs safety/helpline messages instead of a target label.
2. **Invalid / Non-standard Label**: Non-refusal output that does not match any of the 7 predefined label strings.
3. **Class Confusion (Normal vs Clinical)**: Valid label output that confuses `Normal` with a clinical/distress label (or vice versa).
4. **Class Confusion (Inter-disorder)**: Valid label output that confuses one clinical/distress category with another.

| Model | Exact Match | Refusal / Over-cautious | Normal vs Clinical | Inter-disorder Confusion | Invalid Label |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `llama3.2:3b` | 21.9% (23/105) | 30.5% (32/105) | 23.8% (25/105) | 21.9% (23/105) | 1.9% (2/105) |
| `qwen2.5:7b` | 54.3% (57/105) | 0.0% (0/105) | 10.5% (11/105) | 35.2% (37/105) | 0.0% (0/105) |

*Key qualitative observations & illustrative examples (anonymised/abbreviated from `results/compare_105_results.csv`):*
- **Refusal / Over-cautious** (`llama3.2:3b`, 32/105): Safety guardrails trigger on sensitive or crisis-adjacent phrasing, contributing to 0/15 exact matches on `Suicidal` and 0/15 on `Personality disorder`.
- **Normal vs Clinical** (`llama3.2:3b`: 25/105; `qwen2.5:7b`: 11/105): Everyday anecdotes or informal remarks can be over-interpreted as distress (e.g., an awkward/funny past anecdote labelled `Anxiety` by `llama3.2:3b` while labelled `Normal` by `qwen2.5:7b`).
- **Inter-disorder Confusion** (`qwen2.5:7b`: 37/105; `llama3.2:3b`: 23/105): Even when `qwen2.5:7b` adheres to valid labels 100% of the time, overlapping affective/cognitive symptoms cause frequent boundary confusion across clinical categories.

## 4. Limitations & Next Steps
- **Limitations**: The evaluation uses a modest stratified subset (n=105) from a single public text compilation (`Combined Data.csv`) under single-label zero-shot prompting without clinical adjudication or calibration metrics.
- **Next Steps**: To address over-cautious refusals, ambiguous symptom boundaries, and lack of clinical grounding observed in zero-shot classification, the next phase extends this work toward a Retrieval-Augmented Generation (RAG) pipeline grounded in **MIMIC-IV** and **NICE clinical guidelines (CG103 / CG178)** for preprint validation.

## 5. Reproducibility
```bash
pip install -r requirements.txt
ollama pull llama3.2:3b && ollama pull qwen2.5:7b
python3 scripts/compare_105.py
python3 scripts/analyze_errors.py
