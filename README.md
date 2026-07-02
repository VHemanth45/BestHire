# Redrob Intelligent Candidate Discovery

This repository contains our team's submission for the Redrob Hackathon: Intelligent Candidate Discovery.

## 🧠 System Architecture

Our ranking engine utilizes a 3-stage hybrid architecture designed to aggressively filter out keyword stuffers and favor deep production experience over superficial keyword matching:

1. **Hard Gates & Initial Retrieval (35%)**: We first filter out non-ML titles and honeypots. We then compute semantic scores via weighted JD-section embeddings (using `BAAI/bge-small-en-v1.5`), explicitly penalizing anti-pattern matches.
2. **Cross-Encoder Reranking (45%)**: We rerank the top candidates using a Cross-Encoder (`BAAI/bge-reranker-base`). It scores a synthesized candidate profile (top skills, title, summary, work history) against our nuanced JD text to catch subtle, deep contextual matches.
3. **Structured Signals (20%)**: We integrate years of experience, notice period, recruiter engagement rates, recency, GitHub activity, and assessment credibility into a base structured score.

Before blending, each of the three component scores (semantic, cross-encoder, structured) is min-max normalized over the retrieved candidate set, so the 45/35/20 weights reflect true relative influence rather than being skewed by the components' differing natural scales.

Finally, a robust multiplier actively penalizes heavily consulting-based careers, exclusively framework-driven engineers lacking deep retrieval knowledge, and suspicious CV-primary claimers.

> **Reproducibility:** Ranking is fully deterministic — scoring uses a pinned `REFERENCE_DATE` (not `date.today()`) so recency/notice signals don't drift between runs, reasoning selection uses a stable hash, and ties are broken by `candidate_id` ascending to match the submission validator.

## ⚙️ Setup Instructions

**Prerequisites:**
- Python 3.11+
- Ensure the `candidates.jsonl` dataset is located in the root of the repository.

**Installation:**
This project relies on standard data science and ML libraries. If you are using `uv`, simply run:
```bash
uv sync
```
*Alternatively, you can install the dependencies via pip:*
```bash
pip install pandas numpy sentence-transformers
```

## 🚀 How to Reproduce the Submission CSV

The execution pipeline is explicitly split into two steps to comply with the Hackathon's offline compute constraints during the final ranking phase (Stage 3).

### Step 1: Pre-computation & Model Caching (Requires Network)
Run the baseline builder to process the candidate dataset, generate embeddings, and pre-cache the HuggingFace models locally.

```bash
python build_baseline.py
```
> **Note:** This process requires an internet connection and takes approximately **8-12 minutes on a standard CPU**. It will create an `artifacts/` directory containing all pre-computed NumPy embeddings, JSON data, and cached model weights.

### Step 2: Ranking & CSV Generation (Strictly Offline)
Once the artifacts are built and models are cached, you can run the ranking script. This step is designed to run entirely offline, utilizing `local_files_only=True` for the HuggingFace models.

```bash
python rank.py
```
> **Performance:** This step is highly optimized and runs in **under 5 seconds** on a standard CPU.

**Outputs Generated:**
- `submission_new.csv`: The final top 100 ranked candidates formatted for submission, including detailed, deterministic, and human-like reasoning.
- `diagnostics.csv`: A comprehensive breakdown of individual component scores (semantic, cross-encoder, structured, and multipliers) for auditing and evaluation.

## 📁 Repository Structure
- `build_baseline.py`: Handles offline embeddings generation, candidate parsing, and triggers HuggingFace caching.
- `rank.py`: Core offline ranking logic, score blending, hard gates, honeypot detection, and reasoning generation.
- `submission_metadata_template.yaml`: Detailed methodology, AI usage, and compute environment declarations.
- `artifacts/`: (Generated) Directory storing pre-computed `.npy` and `.json` files, alongside cached models.
