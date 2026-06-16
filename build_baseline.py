# precompute_baseline.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

JD_SIGNAL_TEXT = """
We need someone who is simultaneously comfortable with two things that sound contradictory:
    1. Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
    2. Scrappy product-engineering attitude — willing to ship a working ranker in a week even if the underlying ML is "obviously suboptimal," because we need to learn from real users before we know what to actually optimize for.
Things you absolutely need
    • Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model — we care that you've handled embedding drift, index refresh, retrieval-quality regression in production.
    • Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar. Again, the specific tech doesn't matter; the operational experience does.
    • Strong Python. Yes really, we care about code quality.
    • Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful.
nice-to-haves
    • LLM fine-tuning experience (LoRA, QLoRA, PEFT)
    • Experience with learning-to-rank models (XGBoost-based or neural)
    • Prior exposure to HR-tech, recruiting tech, or marketplace products
    • Background in distributed systems or large-scale inference optimization
    • Open-source contributions in the AI/ML space
things we don't care about
    • Title-chasers. If your career trajectory shows you optimizing for "Senior" → "Staff" → "Principal" titles by switching companies every 1.5 years, we're not a fit. We need someone who plans to be here for 3+ years.
    • Framework enthusiasts. If your GitHub is full of LangChain tutorials and your blog posts are "How I used [hot framework] to build [demo]" — that's fine but it's not what we need. We need people who think about systems, not frameworks.
    • People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career. We've had bad fit experiences in both directions. If you're currently at one of these companies but have prior product-company experience, that's fine.
    • People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure. We respect your work but you'd be re-learning fundamentals here.
    • People whose work has been entirely on closed-source proprietary systems for 5+ years without external validation (papers, talks, open-source). We need to see how you think, not just trust that you can think.
location    • Location: Pune/Noida-preferred but flexible. We have offices in Noida and Pune(mostly used Tue/Thu). We don't require any specific number of in-office days but we expect quarterly travel for offsites. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply. Outside India: case-by-case, but we don't sponsor work visas.
    • Notice period: We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher.
    """

# ── 1. Load model ─────────────────────────────────────────────────
print("Loading model...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# ── 2. Embed JD ───────────────────────────────────────────────────
print("Embedding JD...")
jd_embedding = model.encode(
    [BGE_PREFIX + JD_SIGNAL_TEXT],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

np.save(OUT / "jd_baseline_emb.npy", jd_embedding)
print(f"JD embedding shape: {jd_embedding.shape}")  # should be (1, 384)

# ── 3. Load and flatten candidates ───────────────────────────────
print("Loading candidates...")
candidate_ids = []
candidate_texts = []
raw_candidates_list = []

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
        raw_candidates_list.append(cand)

print(f"Loaded {len(candidate_texts)} candidates")

# ── 4. Embed candidates ───────────────────────────────────────────
print("Embedding candidates (this takes ~8-12 min on CPU)...")
candidate_embeddings = model.encode(
    candidate_texts,
    batch_size=256,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

print(f"Embedding matrix shape: {candidate_embeddings.shape}")  # (100000, 384)

# ── 5. Save artifacts ─────────────────────────────────────────────
np.save(OUT / "candidate_baseline_embs.npy", candidate_embeddings)

with open(OUT / "candidate_ids.json", "w") as f:
    json.dump(candidate_ids, f)

with open(OUT / "candidates_raw.json", "w") as f:
    json.dump(raw_candidates_list, f)

print("\nAll artifacts saved:")
print(f"  candidate_baseline_embs.npy  {candidate_embeddings.nbytes / 1e6:.1f} MB")
print(f"  jd_baseline_emb.npy")
print(f"  candidate_ids.json")
print(f"  candidates_raw.json")
print("Done.")