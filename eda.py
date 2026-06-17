# # eda.py — run this FIRST, understand your data
# import gzip, json
# import pandas as pd
# import numpy as np
# from collections import Counter
# from datetime import datetime, date

# print("Loading candidates...")
# candidates = []

# # Using standard open() instead of gzip.open()
# with open("candidates.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         if line.strip():
#             candidates.append(json.loads(line))
            
# print(f"Total: {len(candidates)}")
# # ── Basic profile stats ───────────────────────────────────────────
# countries  = [c["profile"].get("country","") for c in candidates]
# titles     = [c["profile"].get("current_title","") for c in candidates]
# industries = [c["profile"].get("current_industry","") for c in candidates]
# companies  = [c["profile"].get("current_company","") for c in candidates]
# yoe        = [c["profile"].get("years_of_experience", 0) for c in candidates]

# print("\n=== COUNTRY (top 15) ===")
# print(pd.Series(countries).value_counts().head(15))

# print("\n=== CURRENT INDUSTRY (top 15) ===")
# print(pd.Series(industries).value_counts().head(15))

# print("\n=== YOE DISTRIBUTION ===")
# print(pd.Series(yoe).describe())
# print("In 5-9yr band:", sum(1 for y in yoe if 5 <= y <= 9))

# # ── Signal distributions ──────────────────────────────────────────
# open_to_work   = [c["redrob_signals"]["open_to_work_flag"] for c in candidates]
# last_active    = [c["redrob_signals"]["last_active_date"] for c in candidates]
# response_rates = [c["redrob_signals"]["recruiter_response_rate"] for c in candidates]
# notice         = [c["redrob_signals"]["notice_period_days"] for c in candidates]
# github         = [c["redrob_signals"]["github_activity_score"] for c in candidates]

# print("\n=== OPEN TO WORK ===")
# print(pd.Series(open_to_work).value_counts())

# # How stale is the pool?
# today = date.today()
# days_inactive = [
#     (today - datetime.strptime(d, "%Y-%m-%d").date()).days
#     for d in last_active
# ]
# print("\n=== DAYS SINCE LAST ACTIVE ===")
# print(pd.Series(days_inactive).describe())
# print("Active <30d:",  sum(1 for d in days_inactive if d < 30))
# print("Active <90d:",  sum(1 for d in days_inactive if d < 90))
# print("Inactive >180d:", sum(1 for d in days_inactive if d > 180))

# print("\n=== NOTICE PERIOD ===")
# print(pd.Series(notice).describe())
# print("Sub-30d:", sum(1 for n in notice if n <= 30))

# print("\n=== RESPONSE RATE ===")
# print(pd.Series(response_rates).describe())

# print("\n=== GITHUB ACTIVITY (-1 = no github) ===")
# has_github = [g for g in github if g >= 0]
# print(f"No GitHub linked: {sum(1 for g in github if g == -1)}")
# print(pd.Series(has_github).describe())

# # ── Honeypot detection preview ────────────────────────────────────
# print("\n=== POTENTIAL HONEYPOTS (preview) ===")
# honeypot_count = 0
# for c in candidates:
#     skills   = c["skills"]
#     assessed = c["redrob_signals"].get("skill_assessment_scores", {})
#     advanced = [s for s in skills if s["proficiency"] == "advanced"]
    
#     # Flag: many advanced claims but low assessments
#     low_scores = sum(
#         1 for s in advanced
#         if s["name"] in assessed and assessed[s["name"]] < 50
#     )
#     if len(advanced) >= 6 and low_scores >= 3:
#         honeypot_count += 1

# print(f"Candidates with skill inflation pattern: {honeypot_count}")

# # ── How many are plausibly relevant? ─────────────────────────────
# ML_TITLE_KEYWORDS = [
#     "ml", "machine learning", "ai ", "data scientist",
#     "nlp", "research", "applied", "ranking", "search",
#     "recommendation", "retrieval", "platform engineer",
#     "backend engineer", "software engineer"
# ]
# plausible = sum(
#     1 for t in titles
#     if any(kw in t.lower() for kw in ML_TITLE_KEYWORDS)
# )
# print(f"\n=== PLAUSIBLY RELEVANT TITLES ===")
# print(f"Count: {plausible} / {len(candidates)} "
#       f"({100*plausible/len(candidates):.1f}%)")

# india_candidates = sum(1 for c in countries if c == "India")
# print(f"\n=== IN INDIA ===")
# print(f"{india_candidates} / {len(candidates)} "
#       f"({100*india_candidates/len(candidates):.1f}%)")

# title_eda.py
import json
from collections import Counter
import pandas as pd

cands = json.load(open("artifacts/candidates_raw.json"))

# ── 1. Raw title distribution ─────────────────────────────────────
titles = [c["profile"].get("current_title", "MISSING") for c in cands]

print("=== RAW TITLE DISTRIBUTION (top 50) ===\n")
title_counts = Counter(titles)
for title, count in title_counts.most_common(50):
    pct = 100 * count / len(cands)
    print(f"  {count:5d} ({pct:4.1f}%)  {title}")

# ── 2. Title keyword frequency ────────────────────────────────────
print("\n\n=== TITLE KEYWORD FREQUENCY ===\n")
words = []
for title in titles:
    for word in title.lower().split():
        word = word.strip("(),.-")
        if len(word) > 2:
            words.append(word)

word_counts = Counter(words)
for word, count in word_counts.most_common(40):
    pct = 100 * count / len(cands)
    print(f"  {count:5d} ({pct:4.1f}%)  {word}")

# ── 3. ML-relevant vs irrelevant title breakdown ──────────────────
print("\n\n=== TITLE CATEGORY BREAKDOWN ===\n")

ML_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "senior data scientist", "applied scientist",
    "research engineer", "nlp engineer", "search engineer",
    "recommendation systems engineer", "ranking engineer",
    "applied ml engineer", "senior ml engineer", "staff ml engineer",
    "principal ml engineer", "senior ai engineer", "ai specialist",
    "senior software engineer (ml)", "software engineer (ml)",
    "senior applied scientist", "lead ai engineer", "lead ml engineer",
    "computer vision engineer", "senior nlp engineer",
    "staff machine learning engineer", "junior ml engineer"
}

IRRELEVANT_TITLES = {
    "accountant", "mobile developer", "data analyst",
    "civil engineer", "mechanical engineer", "content writer",
    "project manager", "brand designer", "ui developer",
    "ios developer", "android developer", "frontend developer",
    "marketing manager", "sales manager", "hr manager",
    "recruiter", "designer", "finance", "legal",
    "account manager", "business development", "operations",
    "customer success", "support engineer"
}

categories = {
    "clearly_ml":       0,
    "adjacent_ml":      0,  # engineer/scientist/developer but not ML-specific
    "clearly_irrelevant": 0,
    "ambiguous":        0
}

clearly_irrelevant_titles  = []
ambiguous_titles           = []
clearly_ml_titles          = []
adjacent_titles            = []

for title in titles:
    tl = title.lower().strip()

    if any(irr in tl for irr in IRRELEVANT_TITLES):
        categories["clearly_irrelevant"] += 1
        clearly_irrelevant_titles.append(title)

    elif tl in ML_TITLES or any(ml in tl for ml in [
        "machine learning", "ml engineer", "ai engineer",
        "data scientist", "nlp", "search engineer",
        "recommendation", "applied scientist", "ranking"
    ]):
        categories["clearly_ml"] += 1
        clearly_ml_titles.append(title)

    elif any(adj in tl for adj in [
        "engineer", "scientist", "developer", "architect",
        "specialist", "researcher", "analyst", "lead", "staff"
    ]):
        categories["adjacent_ml"] += 1
        adjacent_titles.append(title)

    else:
        categories["ambiguous"] += 1
        ambiguous_titles.append(title)

for cat, count in categories.items():
    pct = 100 * count / len(cands)
    print(f"  {cat:25s}: {count:6,} ({pct:4.1f}%)")

# ── 4. Clearly irrelevant titles — what are they? ─────────────────
print("\n\n=== CLEARLY IRRELEVANT TITLES (top 30) ===\n")
irr_counts = Counter(clearly_irrelevant_titles)
for title, count in irr_counts.most_common(30):
    print(f"  {count:5d}  {title}")

# ── 5. Ambiguous titles — need manual review ──────────────────────
print("\n\n=== AMBIGUOUS TITLES (top 30 — review these) ===\n")
amb_counts = Counter(ambiguous_titles)
for title, count in amb_counts.most_common(30):
    print(f"  {count:5d}  {title}")

# ── 6. How many of each category are open to work? ───────────────
print("\n\n=== OPEN TO WORK BY CATEGORY ===\n")

category_open = {
    "clearly_ml":          {"open": 0, "total": 0},
    "adjacent_ml":         {"open": 0, "total": 0},
    "clearly_irrelevant":  {"open": 0, "total": 0},
    "ambiguous":           {"open": 0, "total": 0},
}

for c in cands:
    title = c["profile"].get("current_title", "").lower().strip()
    open_flag = c["redrob_signals"].get("open_to_work_flag", False)

    if any(irr in title for irr in IRRELEVANT_TITLES):
        cat = "clearly_irrelevant"
    elif any(ml in title for ml in [
        "machine learning", "ml engineer", "ai engineer",
        "data scientist", "nlp", "search engineer",
        "recommendation", "applied scientist", "ranking"
    ]):
        cat = "clearly_ml"
    elif any(adj in title for adj in [
        "engineer", "scientist", "developer", "architect",
        "specialist", "researcher", "analyst", "lead", "staff"
    ]):
        cat = "adjacent_ml"
    else:
        cat = "ambiguous"

    category_open[cat]["total"] += 1
    if open_flag:
        category_open[cat]["open"] += 1

for cat, data in category_open.items():
    total = data["total"]
    open_ = data["open"]
    pct   = 100 * open_ / total if total > 0 else 0
    print(f"  {cat:25s}: {open_:5,} / {total:6,} open ({pct:.1f}%)")

# ── 7. Titles currently in your top 100 ───────────────────────────
print("\n\n=== TITLES IN YOUR CURRENT TOP 100 ===\n")
try:
    import pandas as pd_
    sub       = pd_.read_csv("submission.csv")
    ids       = json.load(open("artifacts/candidate_ids.json"))
    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    top100_titles = []
    for cid in sub["candidate_id"]:
        idx   = id_to_idx[cid]
        title = cands[idx]["profile"].get("current_title", "MISSING")
        top100_titles.append(title)

    top100_counts = Counter(top100_titles)
    for title, count in top100_counts.most_common():
        print(f"  {count:3d}x  {title}")

    # Flag any irrelevant titles that slipped into top 100
    print("\n\n=== ⚠️  IRRELEVANT TITLES IN TOP 100 ===\n")
    found_irrelevant = False
    for rank, (_, row) in enumerate(sub.iterrows(), 1):
        idx   = id_to_idx[row["candidate_id"]]
        title = cands[idx]["profile"].get("current_title", "").lower()
        if any(irr in title for irr in IRRELEVANT_TITLES):
            found_irrelevant = True
            print(f"  #{rank:03d} {row['candidate_id']} — "
                  f"{cands[idx]['profile'].get('current_title','')}")

    if not found_irrelevant:
        print("  ✅ None found")

except FileNotFoundError:
    print("  submission.csv not found — run rank.py first")

# ── 8. Summary stats ──────────────────────────────────────────────
print("\n\n=== SUMMARY ===\n")
print(f"  Total candidates:        {len(cands):,}")
print(f"  Unique titles:           {len(title_counts):,}")
print(f"  Clearly ML titles:       {categories['clearly_ml']:,} "
      f"({100*categories['clearly_ml']/len(cands):.1f}%)")
print(f"  Adjacent ML titles:      {categories['adjacent_ml']:,} "
      f"({100*categories['adjacent_ml']/len(cands):.1f}%)")
print(f"  Clearly irrelevant:      {categories['clearly_irrelevant']:,} "
      f"({100*categories['clearly_irrelevant']/len(cands):.1f}%)")
print(f"  Ambiguous:               {categories['ambiguous']:,} "
      f"({100*categories['ambiguous']/len(cands):.1f}%)")