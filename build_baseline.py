# build_baseline.py — Section-wise JD Embeddings with Importance Weights

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

# ── 1. Parse JD into sections ──────────────────────────────────────
# Each section gets its own embedding and an importance weight.
# Weights reflect how much each section should influence candidate ranking.
# Higher weight = more important for matching.

JD_SECTIONS = {
    # ─── HIGHEST IMPORTANCE: Core technical requirements ───────────
    "must_have_skills": {
        "weight": 0.30,
        "text": (
            "Things you absolutely need: "
            "Production experience with embeddings-based retrieval systems "
            "(sentence-transformers, OpenAI embeddings, BGE, E5, or similar) "
            "deployed to real users. Handled embedding drift, index refresh, "
            "retrieval-quality regression in production. "
            "Production experience with vector databases or hybrid search "
            "infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, "
            "Elasticsearch, FAISS, or something similar. Operational experience matters. "
            "Strong Python with high code quality. "
            "Hands-on experience designing evaluation frameworks for ranking systems "
            "— NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation."
        ),
    },

    # ─── HIGH IMPORTANCE: Role mandate & first 90 days ─────────────
    "role_mandate": {
        "weight": 0.20,
        "text": (
            "Own the intelligence layer of a talent platform. "
            "Ranking, retrieval, and matching systems that decide what recruiters "
            "see when they search for candidates. "
            "Ship a v2 ranking system with embeddings, hybrid retrieval, and "
            "LLM-based re-ranking. Set up evaluation infrastructure — offline "
            "benchmarks, online A/B testing, recruiter-feedback loops. "
            "Drive long-term architecture of candidate-JD matching at scale."
        ),
    },

    # ─── HIGH IMPORTANCE: Technical depth + product mindset ────────
    "core_identity": {
        "weight": 0.15,
        "text": (
            "Deep technical depth in modern ML systems — embeddings, retrieval, "
            "ranking, LLMs, fine-tuning. "
            "Scrappy product-engineering attitude — willing to ship a working "
            "ranker in a week even if suboptimal, learning from real users. "
            "Shipped end-to-end ranking, search, or recommendation system to "
            "real users at meaningful scale. Strong opinions about retrieval "
            "(hybrid vs dense), evaluation (offline vs online), and LLM "
            "integration (when to fine-tune vs prompt)."
        ),
    },

    # ─── MEDIUM IMPORTANCE: Nice-to-haves ─────────────────────────
    "nice_to_have": {
        "weight": 0.10,
        "text": (
            "LLM fine-tuning experience (LoRA, QLoRA, PEFT). "
            "Experience with learning-to-rank models (XGBoost-based or neural). "
            "Prior exposure to HR-tech, recruiting tech, or marketplace products. "
            "Background in distributed systems or large-scale inference optimization. "
            "Open-source contributions in the AI/ML space."
        ),
    },

    # ─── MEDIUM IMPORTANCE: Experience band & disqualifiers ────────
    "experience_profile": {
        "weight": 0.10,
        "text": (
            "5-9 years experience, ideally 6-8 years total with 4-5 in applied "
            "ML/AI roles at product companies, not pure services. "
            "Must have production deployment experience, not purely research. "
            "Must have written production code in the last 18 months. "
            "Pre-LLM-era ML production experience valued over recent LangChain-only projects."
        ),
    },

    # ─── LOWER IMPORTANCE: Location & logistics ────────────────────
    "logistics": {
        "weight": 0.04,
        "text": (
            "Location: Pune or Noida preferred. Offices used Tuesday/Thursday. "
            "Candidates in Hyderabad, Mumbai, Delhi NCR welcome. "
            "Quarterly travel for offsites. "
            "Sub-30-day notice period preferred, can buy out up to 30 days. "
            "Outside India case-by-case, no work visa sponsorship."
        ),
    },

    # ─── LOWER IMPORTANCE: Culture & work style ────────────────────
    "culture_fit": {
        "weight": 0.03,
        "text": (
            "Async-first communication, writes extensively. "
            "Disagrees openly and decides quickly. "
            "Moves fast, comfortable with unstable codebase. "
            "Plans to stay 3+ years, not optimizing for title progression."
        ),
    },
}

# ── 2. Load model ─────────────────────────────────────────────────
print("Loading model...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# ── 3. Embed JD sections ──────────────────────────────────────────
print(f"Embedding {len(JD_SECTIONS)} JD sections...")

section_names = list(JD_SECTIONS.keys())
section_texts = [JD_SECTIONS[name]["text"] for name in section_names]
section_weights = np.array(
    [JD_SECTIONS[name]["weight"] for name in section_names],
    dtype="float32"
)

# Normalize weights to sum to 1.0 (safety check)
section_weights /= section_weights.sum()

print("Section weights:")
for name, w in zip(section_names, section_weights):
    print(f"  {name:25s} → {w:.3f}")

# Embed all sections
jd_section_embeddings = model.encode(
    [BGE_PREFIX + text for text in section_texts],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

print(f"JD section embeddings shape: {jd_section_embeddings.shape}")
# Should be (7, 384) — one embedding per positive section

# ── 4. Embed anti-pattern text SEPARATELY (negative signal) ───────
# This is NOT part of the positive weighted sum — it's used as a
# penalty multiplier in rank.py to down-weight candidates who
# semantically match these red flags.
ANTI_PATTERN_TEXT = (
    "Title-chasers switching companies every 1.5 years. "
    "Framework enthusiasts whose work is LangChain tutorials and demos "
    "rather than systems thinking. "
    "Entire career at consulting firms like TCS, Infosys, Wipro, Accenture, "
    "Cognizant, Capgemini without product-company experience. "
    "Primary expertise in computer vision, speech, or robotics without "
    "significant NLP/IR exposure. "
    "Closed-source proprietary systems for 5+ years without external validation."
)

jd_anti_emb = model.encode(
    [BGE_PREFIX + ANTI_PATTERN_TEXT],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

print(f"Anti-pattern embedding shape: {jd_anti_emb.shape}")  # (1, 384)

# ── 5. Save JD artifacts ──────────────────────────────────────────
np.save(OUT / "jd_section_embs.npy", jd_section_embeddings)
np.save(OUT / "jd_section_weights.npy", section_weights)
np.save(OUT / "jd_anti_pattern_emb.npy", jd_anti_emb)

with open(OUT / "jd_section_meta.json", "w") as f:
    json.dump({
        "section_names": section_names,
        "section_weights": section_weights.tolist(),
        "section_texts": section_texts,
    }, f, indent=2)

print("JD artifacts saved.")

# ── 6. Load and flatten candidates ───────────────────────────────
print("\nLoading candidates...")
candidate_ids = []
candidate_texts = []

with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        cand = json.loads(line)
        p    = cand["profile"]
        sig  = cand["redrob_signals"]

        # Skills — advanced skills repeated for upweighting
        skills_weighted = []
        for s in cand.get("skills", []):
            skills_weighted.append(s["name"])
            if s["proficiency"] == "advanced":
                skills_weighted.append(s["name"])
        skills_text = ", ".join(skills_weighted)

        # Career history — title + company + description
        history_parts = []
        for job in cand.get("career_history", []):
            desc = job.get("description", "")
            history_parts.append(
                f"{job['title']} at {job['company']} "
                f"({job.get('industry','')}) — {desc}"
            )
        history_text = " | ".join(history_parts)

        # Flattened text block
        flattened = (
            f"Headline: {p.get('headline', '')}. "
            f"Summary: {p.get('summary', '')}. "
            f"Current: {p.get('current_title', '')} "
            f"at {p.get('current_company', '')}. "
            f"Skills: {skills_text}. "
            f"Career: {history_text}."
        )

        candidate_ids.append(cand["candidate_id"])
        candidate_texts.append(flattened)

print(f"Loaded {len(candidate_texts)} candidates")

# ── 7. Embed candidates ───────────────────────────────────────────
print("Embedding candidates (this takes ~8-12 min on CPU)...")
candidate_embeddings = model.encode(
    candidate_texts,
    batch_size=256,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

print(f"Embedding matrix shape: {candidate_embeddings.shape}")  # (100000, 384)

# ── 8. Save candidate artifacts ───────────────────────────────────
np.save(OUT / "candidate_baseline_embs.npy", candidate_embeddings)

with open(OUT / "candidate_ids.json", "w") as f:
    json.dump(candidate_ids, f)

print("\nAll artifacts saved:")
print(f"  candidate_baseline_embs.npy  {candidate_embeddings.nbytes / 1e6:.1f} MB")
print(f"  jd_section_embs.npy         ({jd_section_embeddings.shape})")
print(f"  jd_section_weights.npy      ({section_weights.shape})")
print(f"  jd_anti_pattern_emb.npy     ({jd_anti_emb.shape})")
print(f"  jd_section_meta.json")
print(f"  candidate_ids.json")
print("Done.")