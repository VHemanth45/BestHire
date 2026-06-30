# rank.py

import numpy as np
import json
import pandas as pd
import time
from datetime import datetime, date
from pathlib import Path
from sentence_transformers import CrossEncoder

OUT = Path("artifacts")

# ── Configuration ──────────────────────────────────────────────────
TOP_K_RETRIEVAL = 200                        # candidates sent to cross-encoder
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
CROSS_ENCODER_BATCH_SIZE = 8

# Score blending weights (must sum to 1.0)
W_CROSS      = 0.45
W_SEMANTIC   = 0.35
W_STRUCTURED = 0.20

# ── Load artifacts ─────────────────────────────────────────────────
print("Loading artifacts...")
embs  = np.load(OUT / "candidate_baseline_embs.npy")
ids   = json.load(open(OUT / "candidate_ids.json"))

# Load candidates directly from JSONL (avoids 500MB+ intermediate JSON)
cands = []
with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            cands.append(json.loads(line))
print(f"Loaded {len(cands)} candidates from candidates.jsonl")

# Load section-wise JD embeddings and weights
jd_sections = np.load(OUT / "jd_section_embs.npy")    # (N_sections, 384)
jd_weights  = np.load(OUT / "jd_section_weights.npy")  # (N_sections,)

# Load anti-pattern embedding (negative signal)
jd_anti_emb = np.load(OUT / "jd_anti_pattern_emb.npy")  # (1, 384)

with open(OUT / "jd_section_meta.json") as f:
    jd_meta = json.load(f)

print(f"JD sections loaded: {len(jd_meta['section_names'])}")
for name, w in zip(jd_meta["section_names"], jd_weights):
    print(f"  {name:25s} → {w:.3f}")
print(f"Anti-pattern embedding loaded.")

# ── Load cross-encoder model ──────────────────────────────────────
print(f"\nLoading cross-encoder: {CROSS_ENCODER_MODEL}")
t0_model = time.perf_counter()
cross_encoder = CrossEncoder(
    CROSS_ENCODER_MODEL,
    local_files_only=True
)
print(f"Cross-encoder loaded in {time.perf_counter() - t0_model:.1f}s")

# ── Section-weighted semantic scores ───────────────────────────────
# For each candidate, compute cosine similarity against EACH JD section,
# then combine with importance weights.
print("Computing section-weighted semantic scores...")

# Positive signal: weighted sum of per-section similarities
# (100000, N_sections) = (100000, 384) @ (384, N_sections)
per_section_sims = embs @ jd_sections.T
positive_sem_scores = per_section_sims @ jd_weights  # (100000,)

# Negative signal: anti-pattern similarity as penalty multiplier
# Higher anti_sim → candidate profile matches red-flag text → penalty
anti_sims = (embs @ jd_anti_emb.T).squeeze()  # (100000,)
anti_penalty = 1.0 - (anti_sims * 0.4)  # tune 0.4 later
anti_penalty = np.clip(anti_penalty, 0.3, 1.0)  # floor at 0.3 to avoid zeroing out

sem_scores = positive_sem_scores * anti_penalty

# ── Constants ──────────────────────────────────────────────────────

CONSULTING = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "hcl", "tech mahindra",
    "mphasis", "hexaware", "ltimindtree", "persistent",
    "genpact", "dxc technology", "unisys", "kyndryl"
}
# Only 47 unique titles in dataset — whitelist is exhaustive and safer
ML_RELEVANT_TITLES = {
    # Clearly ML
    "ml engineer", "senior ml engineer", "junior ml engineer",
    "staff machine learning engineer", "machine learning engineer",
    "senior machine learning engineer", "lead ml engineer",
    "ai engineer", "senior ai engineer", "lead ai engineer",
    "ai specialist", "ai research engineer",
    "data scientist", "senior data scientist",
    "applied ml engineer", "senior applied scientist",
    "nlp engineer", "senior nlp engineer",
    "search engineer", "recommendation systems engineer",
    "computer vision engineer",
    "senior software engineer (ml)", "software engineer (ml)",
    "staff ml engineer",
}

CV_PRIMARY_SKILLS = {
    "cnn", "opencv", "yolo", "object detection", "image classification",
    "image segmentation", "computer vision",
    "Image Segmentation", "Object Tracking", "Pose Estimation",
    "Face Recognition", "OCR", "Depth Estimation",
}

NLP_IR_SKILLS = {
    "nlp", "information retrieval", "semantic search", "embeddings",
    "sentence transformers", "faiss", "vector search", "qdrant",
    "pinecone", "weaviate", "milvus", "elasticsearch", "opensearch",
    "bm25", "retrieval", "ranking", "recommendation systems",
    "learning to rank", "haystack", "rag"
}

FRAMEWORK_ONLY_SKILLS = {"langchain", "llamaindex", "autogen", "crewai"}

LANGCHAIN_PRIMARY_SIGNALS = {"LangChain", "LangGraph", "LlamaIndex"}

DEEP_RETRIEVAL_SKILLS = {
    "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus",
    "OpenSearch", "Elasticsearch", "Sentence Transformers",
    "BM25", "Haystack", "Vector Search", "Semantic Search",
    "Learning to Rank", "Information Retrieval", "pgvector"
}

TODAY = date.today()

# ── JD query text for cross-encoder ────────────────────────────────
JD_QUERY_TEXT = (
    "Senior AI Engineer for founding team at Series A AI-native talent intelligence platform. "
    "Own the intelligence layer: ranking, retrieval, and candidate-JD matching systems at scale. "
    "Ship v2 ranking system with embeddings-based retrieval, hybrid search (dense + BM25), "
    "and LLM-based re-ranking that demonstrably improves recruiter-engagement metrics. "
    "Production experience with sentence-transformers, BGE, E5, or OpenAI embeddings "
    "deployed to real users, handling embedding drift, index refresh, retrieval-quality regression. "
    "Vector databases and hybrid search infrastructure: Pinecone, Weaviate, Qdrant, Milvus, "
    "OpenSearch, Elasticsearch, FAISS, pgvector — operational experience matters. "
    "Design evaluation frameworks for ranking: NDCG, MRR, MAP, offline-to-online correlation, "
    "A/B test interpretation, recruiter-feedback loops. "
    "Strong Python, code quality, async-first engineering culture. "
    "5-9 years experience, 4-5 in applied ML/AI at product companies (not consulting/services). "
    "Shipped end-to-end ranking, search, or recommendation system to real users at meaningful scale. "
    "Strong opinions on hybrid vs dense retrieval, offline vs online evaluation, "
    "LLM integration (fine-tune vs prompt). "
    "Nice-to-have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost, neural), "
    "HR-tech or marketplace product exposure, distributed systems, open-source contributions. "
    "Not a fit: pure research without production deployment, framework-only experience "
    "(LangChain/LlamaIndex without deep retrieval), career entirely at consulting firms, "
    "primary expertise in computer vision/speech/robotics without NLP/IR depth."
)


# ── Honeypot detection ─────────────────────────────────────────────
def is_honeypot(c):
    skills  = c["skills"]
    sig     = c["redrob_signals"]
    p       = c["profile"]
    history = c.get("career_history", [])
    flags   = 0

    assessed = sig.get("skill_assessment_scores", {})
    advanced = [s for s in skills if s["proficiency"] == "advanced"]

    # Flag 1: Many advanced claims, most fail assessments
    if len(advanced) >= 6:
        low = sum(
            1 for s in advanced
            if s["name"] in assessed and assessed[s["name"]] < 55
        )
        if low >= 3:
            flags += 1

    # Flag 2: Claimed YOE impossible vs career start date
    if history:
        try:
            earliest = min(
                datetime.strptime(job["start_date"], "%Y-%m-%d").year
                for job in history
            )
            max_possible = 2026 - earliest
            if p.get("years_of_experience", 0) > max_possible + 1:
                flags += 1
        except:
            pass

    # Flag 3: Total duration months vs claimed YOE wildly off
    total_months  = sum(job.get("duration_months", 0) for job in history)
    claimed_months = p.get("years_of_experience", 0) * 12
    if total_months > claimed_months * 1.5 and total_months > 24:
        flags += 1

    # Flag 4: All assessments suspiciously perfect
    if len(assessed) >= 5:
        perfect = sum(1 for v in assessed.values() if v >= 90)
        if perfect / len(assessed) > 0.8:
            flags += 1

    # Flag 5: Skill count explosion
    if len(skills) >= 20:
        adv_count = sum(1 for s in skills if s["proficiency"] == "advanced")
        if adv_count >= 15:
            flags += 1

    return flags >= 1


JD_PRIORITY_SKILLS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "bm25", "vector search",
    "semantic search", "sentence transformers", "information retrieval",
    "learning to rank", "ranking systems", "search & discovery",
    "hybrid search", "retrieval", "embeddings", "pgvector",
    "haystack", "rag", "content matching", "vector representations",
    "reranking", "dense retrieval", "sparse retrieval",
    "recommendation systems"
}

JD_NICE_TO_HAVE = {
    "lora", "qlora", "peft", "fine-tuning llms", "llms",
    "mlops", "kubeflow", "mlflow", "a/b testing",
    "learning to rank", "ndcg", "evaluation frameworks"
}

def build_rerank_text(c):
    p = c["profile"]
    parts = []

    title = p.get("current_title", "")
    company = p.get("current_company", "")
    yoe = p.get("years_of_experience", 0)
    if title or company:
        parts.append(f"{title} at {company}, {yoe} years experience.")

    skills_all = c.get("skills", [])
    
    # Split into JD-priority vs rest
    priority_skills = [
        s for s in skills_all
        if s["name"].lower() in JD_PRIORITY_SKILLS
    ]
    other_skills = [
        s for s in skills_all
        if s["name"].lower() not in JD_PRIORITY_SKILLS
    ]

    # Sort each group by proficiency then duration
    def sort_key(s):
        return (
            {"expert": 0, "advanced": 1, "intermediate": 2, "beginner": 3}
            .get(s.get("proficiency", "beginner"), 4),
            -s.get("duration_months", 0)
        )

    priority_skills.sort(key=sort_key)
    other_skills.sort(key=sort_key)

    # Priority skills first, then fill up to 12 total
    ordered = priority_skills + other_skills
    top_skills = ordered[:12]

    skill_parts = [
        f"{s['name']} ({s['proficiency']}, {s.get('duration_months',0)}mo)"
        for s in top_skills
    ]
    if skill_parts:
        parts.append(f"Skills: {', '.join(skill_parts)}.")

    summary = p.get("summary", "")
    if summary:
        parts.append(f"Summary: {summary[:400]}.")

    history = c.get("career_history", [])
    for job in history[:3]:
        desc = job.get("description", "")[:200]
        parts.append(
            f"{job.get('title','')} at {job.get('company','')} "
            f"({job.get('duration_months',0)}mo): {desc}."
        )

    return " ".join(parts)


def compute_multiplier(c):
    """Compute the multiplier for a candidate (extracted for reuse)."""
    p   = c["profile"]
    sig = c["redrob_signals"]
    country = p.get("country", "India")
    multiplier = 1.0

    # Consulting fraction
    def consulting_fraction(history):
        total, consulting = 0, 0
        for job in history:
            m  = job.get("duration_months", 0)
            co = job.get("company", "").lower()
            total += m
            if any(kw in co for kw in CONSULTING):
                consulting += m
        return consulting / total if total > 0 else 0.0

    cf = consulting_fraction(c.get("career_history", []))
    if   cf >= 0.80: multiplier *= 0.45
    elif cf >= 0.50: multiplier *= 0.70
    elif cf >= 0.20: multiplier *= 0.85  # was 0.30 — 20% consulting career is meaningful

    # Recency
    last_active   = datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
    days_inactive = (TODAY - last_active).days
    if   days_inactive > 180: multiplier *= 0.55
    elif days_inactive > 90:  multiplier *= 0.80

    # Outside India but willing to relocate
    if country != "India" and sig.get("willing_to_relocate", False):
        multiplier *= 0.75

    # CV-primary penalty (duration >= 36mo to count as CV-primary)
    cv_skills = [
        s for s in c.get("skills", [])
        if s["name"].lower() in CV_PRIMARY_SKILLS
        and s["proficiency"] in ("advanced", "expert")
        and s.get("duration_months", 0) >= 36
    ]
    cv_count = len(cv_skills)
    cand_advanced = {s["name"].lower() for s in c["skills"]
                     if s["proficiency"] == "advanced"}
    nlp_count = sum(1 for sk in cand_advanced if sk in NLP_IR_SKILLS)

    if   cv_count >= 2 and nlp_count == 0:
        multiplier *= 0.40
    elif cv_count >= 2 and cv_count > nlp_count + 1:
        multiplier *= 0.60
    elif cv_count > nlp_count:
        multiplier *= 0.70

    # Framework enthusiast penalty
    cand_all_skills  = {s["name"].lower() for s in c["skills"]}
    framework_skills = cand_all_skills & FRAMEWORK_ONLY_SKILLS
    real_ir_skills   = cand_all_skills & NLP_IR_SKILLS

    if len(framework_skills) >= 2 and len(real_ir_skills) <= 1:
        multiplier *= 0.65

    first_advanced = next(
        (s["name"].lower() for s in c["skills"]
         if s["proficiency"] == "advanced"), ""
    )
    if first_advanced in FRAMEWORK_ONLY_SKILLS:
        multiplier *= 0.75

    # Context-aware LangChain penalty — only fires if retrieval depth is shallow
    langchain_count = sum(
        1 for s in c.get("skills", [])
        if s["name"] in LANGCHAIN_PRIMARY_SIGNALS
        and s["proficiency"] in ("advanced", "expert")
    )
    retrieval_count = sum(
        1 for s in c.get("skills", [])
        if s["name"] in DEEP_RETRIEVAL_SKILLS
        and s["proficiency"] in ("advanced", "expert")
    )

    # Only penalize if LangChain dominates and retrieval is shallow
    if langchain_count >= 1 and retrieval_count == 0:
        multiplier *= 0.45   # pure framework person, no retrieval depth
    elif langchain_count >= 1 and retrieval_count <= 1:
        multiplier *= 0.70   # some retrieval but framework-heavy
    # else: has LangChain BUT also has real retrieval depth → no penalty

    return multiplier


def compute_structured_scores(idx):
    """Compute the structured component scores for a candidate.

    Returns a dict of individual scores and the combined structured score.
    """
    c   = cands[idx]
    p   = c["profile"]
    sig = c["redrob_signals"]

    yoe = p.get("years_of_experience", 0)
    if   5 <= yoe <= 9:   exp = 1.00
    elif 9 < yoe <= 12:   exp = 0.85
    elif 3 <= yoe < 5:    exp = 0.70
    elif yoe > 12:        exp = 0.65
    else:                 exp = 0.35

    notice = sig.get("notice_period_days", 90)
    if   notice <= 30:  notice_s = 1.00
    elif notice <= 60:  notice_s = 0.75
    elif notice <= 90:  notice_s = 0.45
    else:               notice_s = 0.15

    rr  = sig.get("recruiter_response_rate", 0)
    icr = sig.get("interview_completion_rate", 0)
    oar = sig.get("offer_acceptance_rate", -1)
    if oar < 0: oar = 0.5
    engagement = 0.45 * rr + 0.35 * icr + 0.20 * oar

    last_active   = datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
    days_inactive = (TODAY - last_active).days
    recency  = float(np.exp(-days_inactive / 60))

    gh       = sig.get("github_activity_score", -1)
    github_s = 0.30 if gh < 0 else min(1.0, gh / 60)

    assessed  = sig.get("skill_assessment_scores", {})
    penalties = sum(
        1 for s in c["skills"]
        if s["proficiency"] == "advanced"
        and s["name"] in assessed
        and assessed[s["name"]] < 55
    )
    credibility = max(0.0, 1.0 - penalties * 0.20)

    # Weighted combination matching original relative weights
    # Original: exp=0.15, notice=0.08, engagement=0.08, recency=0.09,
    #           github=0.05, credibility=0.05  (sum=0.50)
    # Normalize to sum to 1.0 for the structured component
    structured = (
        0.30 * exp +
        0.16 * notice_s +
        0.16 * engagement +
        0.18 * recency +
        0.10 * github_s +
        0.10 * credibility
    )

    return {
        "experience_fit": exp,
        "notice_period_score": notice_s,
        "engagement_score": engagement,
        "recency_score": recency,
        "github_score": github_s,
        "assessment_credibility": credibility,
        "structured_combined": structured,
    }


def passes_hard_gates(idx):
    """Check if candidate passes all hard gates. Returns True if passes."""
    c   = cands[idx]
    p   = c["profile"]
    sig = c["redrob_signals"]
    title   = p.get("current_title", "").lower().strip()
    country = p.get("country", "India")

    if is_honeypot(c):
        return False
    if not sig["open_to_work_flag"]:
        return False
    if country != "India" and not sig.get("willing_to_relocate", False):
        return False
    if title not in ML_RELEVANT_TITLES:
        return False
    return True


# ── Phase 1: Initial retrieval + hard gate filtering ───────────────
t_total_start = time.perf_counter()
t_retrieval_start = time.perf_counter()

print("\n── Phase 1: Retrieval + Hard Gate Filtering ──")

# Get initial ranking by semantic score
initial_ranking = np.argsort(sem_scores)[::-1]

# Filter through hard gates and collect Top-K
topk_indices = []
for idx in initial_ranking:
    if passes_hard_gates(idx):
        topk_indices.append(idx)
    if len(topk_indices) >= TOP_K_RETRIEVAL:
        break

topk_indices = np.array(topk_indices)
t_retrieval = time.perf_counter() - t_retrieval_start
print(f"Retrieved {len(topk_indices)} candidates passing hard gates "
      f"(from {len(cands):,} total)")
print(f"Retrieval time: {t_retrieval:.2f}s")

# ── Phase 2: Cross-encoder reranking ──────────────────────────────
t_rerank_start = time.perf_counter()
print(f"\n── Phase 2: Cross-Encoder Reranking ({len(topk_indices)} candidates) ──")

# Build candidate texts for reranking
rerank_texts = [build_rerank_text(cands[idx]) for idx in topk_indices]

# Build query-document pairs for cross-encoder
pairs = [[JD_QUERY_TEXT, text] for text in rerank_texts]

# Batch inference
print(f"Running cross-encoder inference (batch_size={CROSS_ENCODER_BATCH_SIZE})...")
raw_cross_scores = cross_encoder.predict(
    pairs,
    batch_size=CROSS_ENCODER_BATCH_SIZE,
    show_progress_bar=True,
    max_length=512,
)
raw_cross_scores = np.array(raw_cross_scores, dtype=np.float64)

# Normalize cross-encoder scores to [0, 1] via sigmoid
# Sigmoid maps logits naturally to [0,1], preserves relative differences,
# and doesn't let one outlier steal the 1.0 slot (unlike min-max).
norm_cross_scores = 1.0 / (1.0 + np.exp(-raw_cross_scores))

t_rerank = time.perf_counter() - t_rerank_start
print(f"Reranking time: {t_rerank:.2f}s  "
      f"({len(topk_indices) / t_rerank:.0f} candidates/sec)")
print(f"Cross-encoder raw score range: [{raw_cross_scores.min():.4f}, {raw_cross_scores.max():.4f}]")
print(f"Sigmoid-normalized score range: [{norm_cross_scores.min():.4f}, {norm_cross_scores.max():.4f}]")

# ── Phase 3: Score blending ────────────────────────────────────────
print(f"\n── Phase 3: Score Blending ──")
print(f"Weights: cross={W_CROSS}, semantic={W_SEMANTIC}, structured={W_STRUCTURED}")

final_scores_topk = np.zeros(len(topk_indices))
diagnostics = []

for i, idx in enumerate(topk_indices):
    c = cands[idx]

    # Semantic component (already computed)
    semantic = float(sem_scores[idx])

    # Structured component
    struct_info = compute_structured_scores(idx)
    structured = struct_info["structured_combined"]

    # Cross-encoder component (already normalized)
    cross_score = float(norm_cross_scores[i])

    # Multiplier (existing logic preserved)
    multiplier = compute_multiplier(c)

    # Blended score
    blended = (
        W_CROSS      * cross_score +
        W_SEMANTIC   * semantic    +
        W_STRUCTURED * structured
    )
    final = blended * multiplier

    final_scores_topk[i] = final

    # Store diagnostics
    diagnostics.append({
        "candidate_id": ids[idx],
        "semantic_score": round(semantic, 4),
        "cross_encoder_score": round(cross_score, 4),
        "cross_encoder_raw": round(float(raw_cross_scores[i]), 4),
        "structured_score": round(structured, 4),
        "multiplier": round(multiplier, 4),
        "final_score": round(final, 4),
        **{k: round(v, 4) for k, v in struct_info.items()
           if k != "structured_combined"},
    })

# ── Top 100 ────────────────────────────────────────────────────────
top100_order = np.argsort(final_scores_topk)[::-1][:100]
top100_idx_global = topk_indices[top100_order]

# ── Honeypot check on top 100 ──────────────────────────────────────
# Submission is disqualified if >10 honeypots in top 100
honeypot_in_top100 = sum(
    1 for idx in top100_idx_global if is_honeypot(cands[idx])
)
print(f"\nHoneypots in top 100: {honeypot_in_top100} "
      f"({'⚠️  RISK' if honeypot_in_top100 > 5 else '✅ OK'})")


# ── Reasoning ──────────────────────────────────────────────────────
def make_reasoning(c, rank):
    p   = c["profile"]
    sig = c["redrob_signals"]

    yoe     = p.get("years_of_experience", 0)
    title   = p.get("current_title", "")
    company = p.get("current_company", "")
    notice  = sig.get("notice_period_days", 90)
    active  = sig.get("last_active_date", "")
    rr      = sig.get("recruiter_response_rate", 0.5)
    icr     = sig.get("interview_completion_rate", 0.5)
    gh      = sig.get("github_activity_score", -1)
    passive = (
        sig.get("profile_views_received_30d", 0) > 100 or
        sig.get("saved_by_recruiters_30d", 0) > 30
    )

    last_active_date = datetime.strptime(active, "%Y-%m-%d").date()
    days_inactive = (date.today() - last_active_date).days

    # Skills analysis
    all_skills = c.get("skills", [])
    expert_skills = [s for s in all_skills if s["proficiency"] == "expert"]
    advanced_skills = [s for s in all_skills if s["proficiency"] == "advanced"]
    top_skills = sorted(
        expert_skills + advanced_skills,
        key=lambda s: -s.get("duration_months", 0)
    )

    top3_names = [s["name"] for s in top_skills[:3]]

    jd_hits = [
        s for s in top_skills
        if s["name"].lower() in JD_PRIORITY_SKILLS
    ]
    nice_hits = [
        s for s in top_skills
        if s["name"].lower() in JD_NICE_TO_HAVE
    ]

    jd_hit_names = [s["name"] for s in jd_hits[:3]]
    jd_hit_durations = [s.get("duration_months", 0) for s in jd_hits[:3]]

    # Career history signals
    history = c.get("career_history", [])
    recent_company = history[0].get("company", company) if history else company
    recent_desc = history[0].get("description", "")[:120] if history else ""

    # ── Build reasoning parts ──────────────────────────────────────

    # PART 1: Opening — rank-tier appropriate, varied openers
    if rank <= 5:
        openers = [
            f"One of the strongest profiles in the pool — {yoe:.1f} years as {title} at {company}",
            f"Exceptional match for this JD — {title} at {company} with {yoe:.1f} years of hands-on experience",
            f"Top-tier candidate: {yoe:.1f}yr {title} at {company}, with a profile that maps directly to what the JD is asking for",
        ]
    elif rank <= 15:
        openers = [
            f"Strong candidate — {yoe:.1f}yr {title} at {company}",
            f"Good profile overall — {title} at {company} with {yoe:.1f} years of relevant experience",
            f"{yoe:.1f}yr {title} at {company}; solid fit for the core requirements",
        ]
    elif rank <= 35:
        openers = [
            f"Decent fit with some gaps — {yoe:.1f}yr {title} at {company}",
            f"Reasonable candidate — {title} at {company} with {yoe:.1f} years, though not a perfect match",
            f"{yoe:.1f}yr {title} at {company}; covers some JD requirements but not all",
        ]
    elif rank <= 60:
        openers = [
            f"Moderate overlap with JD requirements — {yoe:.1f}yr {title} at {company}",
            f"Partial fit — {title} at {company} ({yoe:.1f}yr) has adjacent skills but misses some core JD signals",
            f"{yoe:.1f}yr {title} at {company}; included for relevant signals but ranked lower due to skill gaps",
        ]
    elif rank <= 80:
        openers = [
            f"Ranked lower despite some retrieval signals — {yoe:.1f}yr {title} at {company}; notice period and engagement concerns outweigh skill fit",
            f"Deprioritized due to operational concerns — {title} at {company} ({yoe:.1f}yr) has relevant skills but availability or engagement flags",
            f"{yoe:.1f}yr {title} at {company}; retrieval skills present but ranked lower on notice period, recency, or response rate signals",
        ]
    else:
        openers = [
            f"Below the core cutoff — {yoe:.1f}yr {title} at {company}",
            f"Weak JD alignment — {title} at {company} ({yoe:.1f}yr); included as filler given experience breadth",
            f"{yoe:.1f}yr {title} at {company}; profile doesn't strongly match retrieval/ranking mandate",
        ]

    # Use hash of candidate_id for deterministic but varied selection
    opener_idx = hash(c.get("candidate_id", "")) % len(openers)
    opening = openers[opener_idx]

    # PART 2: JD connection
    if len(jd_hits) >= 3:
        jd_part = (
            f"Directly covers the JD's core retrieval stack — "
            f"{jd_hit_names[0]} ({jd_hit_durations[0]}mo), "
            f"{jd_hit_names[1]} ({jd_hit_durations[1]}mo), "
            f"and {jd_hit_names[2]} ({jd_hit_durations[2]}mo) — "
            f"exactly the kind of production retrieval depth the role requires."
        )
    elif len(jd_hits) == 2:
        jd_part = (
            f"Covers {jd_hit_names[0]} and {jd_hit_names[1]} from the JD's must-have stack, "
            f"though missing some breadth on vector DB or hybrid search experience."
        )
    elif len(jd_hits) == 1:
        jd_part = (
            f"Only one strong JD signal — {jd_hit_names[0]} — "
            f"with the rest of the profile in adjacent but not core retrieval territory."
        )
    else:
        jd_part = (
            f"Top skills ({', '.join(top3_names)}) don't directly map to the JD's "
            f"retrieval and ranking mandate; this is a general ML profile without clear IR depth."
        )

    # PART 3: Nice-to-have or career narrative signal
    narrative_part = ""
    if recent_desc and rank <= 30:
        # Trim description to meaningful snippet
        snippet = recent_desc.strip().rstrip(".")
        narrative_part = f"Recent work at {recent_company} involved: \"{snippet}...\""
    elif nice_hits and rank <= 50:
        nice_names = [s["name"] for s in nice_hits[:2]]
        narrative_part = f"Also brings {', '.join(nice_names)} — useful for the eval framework and fine-tuning aspects of the role."

    # PART 4: Honest concerns — always included where applicable
    concerns = []

    if notice > 90:
        concerns.append(
            f"{notice}-day notice is a significant hiring risk — "
            f"likely needs negotiation or org is okay with delayed start"
        )
    elif notice > 60:
        concerns.append(f"{notice}-day notice period — workable but not immediate")

    if days_inactive > 180:
        concerns.append(
            f"profile inactive for {days_inactive} days — "
            f"unclear if still actively looking despite open-to-work flag"
        )
    elif days_inactive > 90:
        concerns.append(f"hasn't been active in {days_inactive} days — response rate may lag")

    if rr < 0.35:
        concerns.append(
            f"low recruiter response rate ({rr:.0%}) — "
            f"historically slow to engage, may not convert easily"
        )

    if icr < 0.40:
        concerns.append(f"interview completion rate of {icr:.0%} — drops out of processes")

    if len(jd_hits) == 0 and rank <= 50:
        concerns.append("no strong retrieval/ranking skills in top profile signals — JD fit is primarily semantic, not explicit")

    # PART 5: Positive signals worth noting
    positives = []
    if notice <= 30:
        positives.append("immediately or near-immediately available")
    if gh > 50:
        positives.append(f"active GitHub presence (score {gh:.0f}) — open-source signal")
    if passive:
        positives.append("being actively watched by recruiters — market-validated profile")
    if rr > 0.70:
        positives.append(f"high recruiter response rate ({rr:.0%}) — likely to engage")

    # ── Assemble final string ──────────────────────────────────────
    parts = [opening + "."]
    parts.append(jd_part)

    if narrative_part:
        parts.append(narrative_part + ".")

    if positives and rank <= 40:
        parts.append("Positive signals: " + "; ".join(positives) + ".")

    if concerns:
        if rank <= 20:
            parts.append("Worth noting: " + "; ".join(concerns) + ".")
        else:
            parts.append("Concerns: " + "; ".join(concerns) + ".")

    if not concerns and rank <= 10:
        parts.append("No significant red flags — clean profile for this JD.")

    return " ".join(parts)


# ── Build submission CSV ────────────────────────────────────────────
rows = []
for rank, order_idx in enumerate(top100_order, 1):
    global_idx = topk_indices[order_idx]
    rows.append({
        "candidate_id": ids[global_idx],
        "rank":         rank,
        "score":        round(final_scores_topk[order_idx], 4),
        "reasoning":    make_reasoning(cands[global_idx], rank)
    })

df = pd.DataFrame(rows)
df.to_csv("submission_new.csv", index=False)

# ── Diagnostics CSV ────────────────────────────────────────────────
diag_df = pd.DataFrame(diagnostics)
# Sort by final_score descending
diag_df = diag_df.sort_values("final_score", ascending=False).reset_index(drop=True)
diag_df.insert(0, "rank", range(1, len(diag_df) + 1))
diag_df.to_csv("diagnostics.csv", index=False)
print(f"\nDiagnostics saved → diagnostics.csv ({len(diag_df)} rows)")

# ── Performance summary ───────────────────────────────────────────
t_total = time.perf_counter() - t_total_start
print(f"\n{'='*60}")
print(f"PERFORMANCE SUMMARY")
print(f"{'='*60}")
print(f"  Retrieval time:      {t_retrieval:>7.2f}s")
print(f"  Reranking time:      {t_rerank:>7.2f}s")
print(f"  Total ranking time:  {t_total:>7.2f}s")
print(f"  Candidates/sec:      {len(topk_indices) / t_rerank:>7.0f}")
print(f"  Top-K retrieved:     {len(topk_indices):>7d}")
print(f"{'='*60}")

# ── Final output ──────────────────────────────────────────────────
print("\n=== TOP 10 FINAL RANKING ===")
print(df.head(10).to_string(index=False))

# Show top-10 diagnostics
print("\n=== TOP 10 DIAGNOSTICS ===")
diag_top10 = diag_df.head(10)[[
    "rank", "candidate_id", "semantic_score", "cross_encoder_score",
    "structured_score", "multiplier", "final_score"
]]
print(diag_top10.to_string(index=False))

print(f"\nSubmission saved → submission_new.csv ({len(df)} rows)")