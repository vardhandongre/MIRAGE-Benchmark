# MIRAGE‑MMST Benchmark · Evaluation Guide

## 0  Overview

**MIRAGE‑MMST** probes **single‑turn multimodal reasoning** in real‑world agricultural consultations.  Each instance combines:

* **Question (`q`)** – a farmer’s natural‑language query.
* **Image set (`I`)** – one or more user‑supplied photos (crop, pest, symptom, etc.).
* **Metadata (`meta`)** – timestamp, geo‑location and other contextual hints.

Formally, an instance is $(q,\, I,\, \text{meta}) \in \mathcal{Q}\times\mathcal{I}^m\times\mathcal{M}$.  A model must output a structured reply $r=(e,\,c\lor m)$ where

* **`e`** – identified entity/entities (crop, pest, disease),
* **`c`** – causal explanation of the observed symptoms, and
* **`m`** – evidence‑grounded management recommendation (*included only when the user explicitly requests advice*).

The benchmark measures a model’s ability to (1) detect visual cues, (2) identify agronomic entities, (3) reason causally from multimodal evidence, and (4) deliver precise, context‑appropriate guidance.

### Subsets

| Subset         | Description                                                                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Standard**   | Self‑contained questions answerable directly from $q$ and $I$.  The model must identify entities, explain causes, and—when asked—suggest management.                                       |
| **Contextual** | Questions with implicit background assumptions (e.g. growth stage inferred from date/location).  Correct answers require reconstructing missing context from priors or external knowledge. |

---

## 1  Dataset access

The full benchmark is on **[Hugging Face Datasets](https://huggingface.co/datasets/MIRAGE-Benchmark/MIRAGE)** :

```python
from datasets import load_dataset

# Standard split
ds_standard = load_dataset("MIRAGE-Benchmark/MIRAGE", "MMST_Standard")

# Contextual split
ds_contextual = load_dataset("MIRAGE-Benchmark/MIRAGE", "MMST_Contextual")
```

| Split             | # Dialogues |
| ----------------- | ----------- |
| `MMST_Standard`   | 8 184       |
| `MMST_Contextual` | 3 934       |

For a quick test, two miniature files with identical schema are provided in **`Datasets/sample_bench/`**:

* `sample_standard_benchmark.json`
* `sample_contextual_benchmark.json`

---

## 2  Repository layout

```text
MMST/
├── chat_models
├── Datasets
├── Evaluation
├── Inference
├── README.md
└── requirements.txt
```

---

## 3  End‑to‑end workflow

### 3.0 Environment setup

```bash
# create & activate a clean env (example)
conda create -n mirage python=3.12 -y
conda activate mirage

pip install -r requirements.txt
```

We uses three open‑source reasoning models for evaluation:

* `microsoft/Phi-4-reasoning`
* `Qwen/Qwen3-32B`
* `DeepSeek-R1-Distill-Llama-70B`

We recommend serving them with **[vLLM](https://github.com/vllm-project/vllm)** (or any OpenAI‑compatible endpoint).

### 3.1 Generate predictions

```bash
cd Inference
bash bash_generate.sh   # edit BENCH_TYPE / MODEL_NAME inside the script
```

`bash_generate.sh` calls `generate.py`, writes raw completions to `Inference/results/`, and then invokes `split.py` to create judge‑ready shards.

### 3.2 Score with LLM‑as‑Judge

```bash
cd ../Evaluation
bash bash_LLMsAsJudges.sh   # edit SUBJECT_NAME / JUDGE_NAME inside the script
```

The script launches parallel evaluators (`LLMsAsJudges_ID.py` or `LLMsAsJudges_MG.py`).  Scores are saved in `Evaluation/results/` with logs in `Evaluation/logs/`.

### 3.3 Print aggregate scores

```bash
bash bash_print_scores.sh
```

`print_scores.py` merges instance‑level judgments and prints a tidy table (LaTeX‑ready if desired).  For Management‑Guidance tasks it also reports a **Weighted Sum** metric:

$$
\text{Weighted Sum}=\frac{2\,\text{Accuracy}+\text{Relevance}+\text{Completeness}+\text{Parsimony}}{6}\;\in[0,1]
$$

---

## 4  Evaluation criteria

### 4.1 Identification (ID)

| Metric                      | Range | Definition                                                                                                                     |
| --------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Identification Accuracy** | 0 / 1 | 1 if the predicted entity matches any gold `entity_name`, `scientific_name` or `common_names` (case‑insensitive); otherwise 0. |
| **Reasoning Accuracy**      | 0–4   | Agreement with expert rationale based on key visual clues, descriptive precision, and logical causal links.                    |

**Reasoning Accuracy rubric**

| Score | Guideline                                                     |
| ----- | ------------------------------------------------------------- |
| **4** | ≥2 key clues, precise description **and** clear causal chain. |
| **3** | ≥2 clues, mostly precise, partial causal linkage.             |
| **2** | 1–2 clues with incomplete reasoning.                          |
| **1** | ≤1 vague clue, no causal link.                                |
| **0** | Off‑topic or no usable observation.                           |

### 4.2 Management Guidance (MG)

Each facet is scored 0–4.

| Facet            | Focus                                                                                     |
| ---------------- | ----------------------------------------------------------------------------------------- |
| **Accuracy**     | Correctness of agricultural facts, species names, diagnoses, causal logic.                |
| **Relevance**    | Alignment with visible evidence and the user’s stated needs; excludes irrelevant content. |
| **Completeness** | Coverage of all key points in the expert answer (diagnosis, treatment, prevention …).     |
| **Parsimony**    | Clarity & conciseness—actionable guidance without unnecessary complexity (Occam’s Razor). |

**MG rubric (excerpt)**

| Score | Accuracy           | Relevance          | Completeness          | Parsimony              |
| ----- | ------------------ | ------------------ | --------------------- | ---------------------- |
| **4** | Fully correct      | Entirely on‑topic  | All key points        | Succinct & unambiguous |
| **3** | Minor errors       | Mostly relevant    | Misses minor detail   | Slight extra content   |
| **2** | Significant errors | Partially relevant | Omits major component | Verbose/technical      |
| **1** | Major inaccuracies | Off‑topic          | Superficial           | Indirect/unclear       |
| **0** | Incorrect          | Unrelated          | No key elements       | No actionable advice   |

---

## 5  License

Dataset — **CC‑BY‑NC‑SA‑4.0**

Code     — **Apache 2.0**
