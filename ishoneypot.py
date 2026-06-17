# inspect_middle.py
import json, pandas as pd

cands     = json.load(open("artifacts/candidates_raw.json"))
ids       = json.load(open("artifacts/candidate_ids.json"))
sub       = pd.read_csv("submission.csv")
id_to_idx = {cid: i for i, cid in enumerate(ids)}

CONSULTING_KEYWORDS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "hcl", "tech mahindra",
    "mphasis", "hexaware", "ltimindtree", "persistent",
    "genpact", "dxc", "unisys", "kyndryl"
}

CV_PRIMARY = {
    "cnn", "opencv", "yolo", "object detection", "image classification",
    "image segmentation", "computer vision", "diffusion models",
    "stable diffusion", "gan", "gans"
}

NLP_IR = {
    "nlp", "information retrieval", "semantic search", "embeddings",
    "sentence transformers", "faiss", "vector search", "qdrant",
    "pinecone", "weaviate", "milvus", "elasticsearch", "opensearch",
    "bm25", "retrieval", "ranking", "recommendation systems",
    "learning to rank", "haystack", "rag"
}

print("=== RANKS 11-50 FULL INSPECTION ===\n")

for _, row in sub[(sub["rank"] >= 11) & (sub["rank"] <= 50)].iterrows():
    idx = id_to_idx[row["candidate_id"]]
    c   = cands[idx]
    p   = c["profile"]
    sig = c["redrob_signals"]

    adv     = [s["name"] for s in c["skills"] if s["proficiency"] == "advanced"]
    adv_low = {s.lower() for s in adv}
    cv_c    = sum(1 for s in adv_low if s in CV_PRIMARY)
    nlp_c   = sum(1 for s in adv_low if s in NLP_IR)

    # Consulting fraction
    total, consulting = 0, 0
    for job in c.get("career_history", []):
        m  = job.get("duration_months", 0)
        co = job.get("company", "").lower()
        total += m
        if any(kw in co for kw in CONSULTING_KEYWORDS):
            consulting += m
    cf = consulting / total if total > 0 else 0

    # Flag issues
    flags = []
    if cf >= 0.50:
        flags.append(f"consulting {cf:.0%}")
    if cv_c >= 2 and cv_c >= nlp_c:
        flags.append(f"cv-primary (cv={cv_c},nlp={nlp_c})")
    if p.get("years_of_experience", 0) < 4:
        flags.append(f"underyoe ({p.get('years_of_experience')}yr)")
    if sig.get("notice_period_days", 0) > 90:
        flags.append(f"long notice ({sig.get('notice_period_days')}d)")

    flag_str = f"  ⚠️  {', '.join(flags)}" if flags else "  ✅"

    print(f"#{row['rank']:02d} {row['candidate_id']} | {row['score']:.4f}")
    print(f"    {p.get('current_title','')} @ {p.get('current_company','')}")
    print(f"    {p.get('years_of_experience',0)}yr | "
          f"notice={sig.get('notice_period_days',0)}d | "
          f"active={sig.get('last_active_date','')}")
    print(f"    advanced: {', '.join(adv[:4])}")
    print(flag_str)
    print()