# rank.py

import numpy as np
import json
import pandas as pd
from datetime import datetime, date
from pathlib import Path

OUT = Path("artifacts")

# ── Load artifacts ─────────────────────────────────────────────────
print("Loading artifacts...")
embs  = np.load(OUT / "candidate_baseline_embs.npy")
jd    = np.load(OUT / "jd_baseline_emb.npy")
ids   = json.load(open(OUT / "candidate_ids.json"))
cands = json.load(open(OUT / "candidates_raw.json"))

# ── Semantic scores ────────────────────────────────────────────────
print("Computing semantic scores...")
sem_scores = (embs @ jd.T).squeeze()  # (100000,)

# ── Constants ──────────────────────────────────────────────────────
CONSULTING = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "hcl", "tech mahindra",
    "mphasis", "hexaware", "ltimindtree", "persistent",
    "genpact", "dxc technology", "unisys", "kyndryl"
}

IRRELEVANT_TITLES = {
    "marketing", "sales", "hr manager", "recruiter",
    "designer", "content writer", "finance", "legal",
    "product manager", "account manager", "business development",
    "operations", "customer success", "support engineer"
}

CV_PRIMARY_SKILLS = {
    "cnn", "opencv", "yolo", "object detection", "image classification",
    "image segmentation", "computer vision", "diffusion models",
    "stable diffusion", "gan", "gans"
}

NLP_IR_SKILLS = {
    "nlp", "information retrieval", "semantic search", "embeddings",
    "sentence transformers", "faiss", "vector search", "qdrant",
    "pinecone", "weaviate", "milvus", "elasticsearch", "opensearch",
    "bm25", "retrieval", "ranking", "recommendation systems",
    "learning to rank", "haystack", "rag"
}

FRAMEWORK_ONLY_SKILLS = {"langchain", "llamaindex", "autogen", "crewai"}

TODAY = date.today()

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


# ── Scoring ────────────────────────────────────────────────────────
def score_candidate(idx):
    c   = cands[idx]
    p   = c["profile"]
    sig = c["redrob_signals"]

    # Honeypot → zero
    if is_honeypot(c):
        return 0.0

    # ── Hard gates ─────────────────────────────────────────────────
    if not sig["open_to_work_flag"]:
        return 0.0

    country = p.get("country", "India")
    if country != "India" and not sig.get("willing_to_relocate", False):
        return 0.0

    title = p.get("current_title", "").lower()
    if any(t in title for t in IRRELEVANT_TITLES):
        return 0.0

    # ── Multipliers ────────────────────────────────────────────────
    multiplier = 1.0

    # Replace current consulting block with this
    def consulting_career_weight(career_history):
        """Returns fraction of career months spent at consulting firms."""
        total_months      = 0
        consulting_months = 0

        for job in career_history:
            months = job.get("duration_months", 0)
            total_months += months
            co = job.get("company", "").lower()
            if any(kw in co for kw in CONSULTING):
                consulting_months += months

        if total_months == 0:
            return 0.0
        return consulting_months / total_months

    consulting_fraction = consulting_career_weight(c.get("career_history", []))

    if consulting_fraction >= 0.80:
        multiplier *= 0.45    # almost entirely consulting
    elif consulting_fraction >= 0.50:
        multiplier *= 0.70    # majority consulting career
    elif consulting_fraction >= 0.30:
        multiplier *= 0.85    # some consulting but product experience exists

    # 2. Recency
    last_active   = datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
    days_inactive = (TODAY - last_active).days
    if   days_inactive > 180: multiplier *= 0.55
    elif days_inactive > 90:  multiplier *= 0.80

    # 3. Outside India but willing to relocate
    if country != "India" and sig.get("willing_to_relocate", False):
        multiplier *= 0.75

    # 4. Junior seniority penalty
    JUNIOR_SIGNALS = {"junior", "associate", "intern", "trainee", "graduate"}
    if any(j in title for j in JUNIOR_SIGNALS):
        multiplier *= 0.70

    # 5. CV-primary penalty — fixed to catch equal CV/NLP counts
    cand_advanced = {s["name"].lower() for s in c["skills"]
                     if s["proficiency"] == "advanced"}
    cv_count  = sum(1 for sk in cand_advanced if sk in CV_PRIMARY_SKILLS)
    nlp_count = sum(1 for sk in cand_advanced if sk in NLP_IR_SKILLS)

    if   cv_count >= 2 and nlp_count == 0:        multiplier *= 0.40
    elif cv_count >= 2 and cv_count >= nlp_count: multiplier *= 0.60
    elif cv_count > nlp_count:                    multiplier *= 0.70

    # 6. Framework enthusiast penalty
    cand_all_skills  = {s["name"].lower() for s in c["skills"]}
    framework_skills = cand_all_skills & FRAMEWORK_ONLY_SKILLS
    real_ir_skills   = cand_all_skills & NLP_IR_SKILLS

    # Original: required 2+ framework skills
    if len(framework_skills) >= 2 and len(real_ir_skills) <= 1:
        multiplier *= 0.65

    # NEW: also penalize if LangChain is the FIRST listed advanced skill
    # First listed = highest priority skill per candidate's own ordering
    first_advanced = next(
        (s["name"].lower() for s in c["skills"] if s["proficiency"] == "advanced"),
        ""
    )
    if first_advanced in FRAMEWORK_ONLY_SKILLS:
        multiplier *= 0.75    # framework as top skill = concerning

    # ── Component scores ───────────────────────────────────────────

    # 1. Semantic similarity
    sem = float(sem_scores[idx])

    # 2. Experience fit
    yoe = p.get("years_of_experience", 0)
    if   5 <= yoe <= 9:   exp = 1.00
    elif 9 < yoe <= 12:   exp = 0.85
    elif 3 <= yoe < 5:    exp = 0.70
    elif yoe > 12:        exp = 0.65
    else:                 exp = 0.35

    # 3. Notice period
    notice = sig.get("notice_period_days", 90)
    if   notice <= 30:  notice_s = 1.00
    elif notice <= 60:  notice_s = 0.75
    elif notice <= 90:  notice_s = 0.45
    else:               notice_s = 0.15

    # 4. Engagement
    rr  = sig.get("recruiter_response_rate", 0)
    icr = sig.get("interview_completion_rate", 0)
    oar = sig.get("offer_acceptance_rate", -1)
    if oar < 0:
        oar = 0.5
    engagement = 0.45 * rr + 0.35 * icr + 0.20 * oar

    # 5. Recency
    recency = float(np.exp(-days_inactive / 60))

    # 6. GitHub
    gh = sig.get("github_activity_score", -1)
    github_s = 0.30 if gh < 0 else min(1.0, gh / 60)

    # 7. Assessment credibility
    assessed = sig.get("skill_assessment_scores", {})
    penalties = sum(
        1 for s in c["skills"]
        if s["proficiency"] == "advanced"
        and s["name"] in assessed
        and assessed[s["name"]] < 55
    )
    credibility = max(0.0, 1.0 - penalties * 0.20)

    # ── Composite ──────────────────────────────────────────────────
    score = (
        0.35 * sem         +
        0.15 * exp         +
        0.15 * notice_s    +
        0.15 * engagement  +
        0.10 * recency     +
        0.05 * github_s    +
        0.05 * credibility
    )

    return score * multiplier


# ── Score all candidates ───────────────────────────────────────────
print("Scoring 100K candidates...")
final_scores = np.array([
    score_candidate(i) for i in range(len(cands))
])

zeroed = (final_scores == 0.0).sum()
print(f"Zeroed out: {zeroed:,} / {len(cands):,} "
      f"({100 * zeroed / len(cands):.1f}%)")

active_scores = final_scores[final_scores > 0]
print(f"Active candidates: {len(active_scores):,}")
print(f"Score range: {active_scores.min():.4f} → {active_scores.max():.4f}")
print(f"Score mean:  {active_scores.mean():.4f}")

# ── Top 100 ────────────────────────────────────────────────────────
top100_idx = np.argsort(final_scores)[::-1][:100]

# ── Honeypot check on top 100 ──────────────────────────────────────
# Submission is disqualified if >10 honeypots in top 100
honeypot_in_top100 = sum(
    1 for idx in top100_idx if is_honeypot(cands[idx])
)
print(f"\nHoneypots in top 100: {honeypot_in_top100} "
      f"({'⚠️  RISK' if honeypot_in_top100 > 5 else '✅ OK'})")


# ── Reasoning ──────────────────────────────────────────────────────
def make_reasoning(c, rank):
    p   = c["profile"]
    sig = c["redrob_signals"]

    adv    = [s["name"] for s in c["skills"]
              if s["proficiency"] == "advanced"][:3]
    yoe    = p.get("years_of_experience", 0)
    title  = p.get("current_title", "")
    notice = sig.get("notice_period_days", 90)
    active = sig.get("last_active_date", "")
    rr     = sig.get("recruiter_response_rate", 0)
    co     = p.get("current_company", "")
    loc    = p.get("location", "")

    skills_str = ", ".join(adv) if adv else "general ML background"

    concerns = []
    if notice > 60:
        concerns.append(f"{notice}-day notice")
    if rr < 0.30:
        concerns.append(f"low response rate ({rr:.0%})")

    concern_str = ("; concern: " + ", ".join(concerns)) if concerns else ""

    return (
        f"{yoe:.1f}yr {title} at {co} ({loc}); "
        f"strong in {skills_str}; "
        f"active {active}, notice {notice}d"
        f"{concern_str}."
    )


# ── Build submission CSV ────────────────────────────────────────────
rows = []
for rank, idx in enumerate(top100_idx, 1):
    rows.append({
        "candidate_id": ids[idx],
        "rank":         rank,
        "score":        round(final_scores[idx], 4),
        "reasoning":    make_reasoning(cands[idx], rank)
    })

df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)

print("\n=== TOP 10 FINAL RANKING ===")
print(df.head(10).to_string(index=False))
print(f"\nSubmission saved → submission.csv ({len(df)} rows)")