# eda.py — run this FIRST, understand your data
import gzip, json
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, date

print("Loading candidates...")
candidates = []

# Using standard open() instead of gzip.open()
with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            candidates.append(json.loads(line))
            
print(f"Total: {len(candidates)}")
# ── Basic profile stats ───────────────────────────────────────────
countries  = [c["profile"].get("country","") for c in candidates]
titles     = [c["profile"].get("current_title","") for c in candidates]
industries = [c["profile"].get("current_industry","") for c in candidates]
companies  = [c["profile"].get("current_company","") for c in candidates]
yoe        = [c["profile"].get("years_of_experience", 0) for c in candidates]

print("\n=== COUNTRY (top 15) ===")
print(pd.Series(countries).value_counts().head(15))

print("\n=== CURRENT INDUSTRY (top 15) ===")
print(pd.Series(industries).value_counts().head(15))

print("\n=== YOE DISTRIBUTION ===")
print(pd.Series(yoe).describe())
print("In 5-9yr band:", sum(1 for y in yoe if 5 <= y <= 9))

# ── Signal distributions ──────────────────────────────────────────
open_to_work   = [c["redrob_signals"]["open_to_work_flag"] for c in candidates]
last_active    = [c["redrob_signals"]["last_active_date"] for c in candidates]
response_rates = [c["redrob_signals"]["recruiter_response_rate"] for c in candidates]
notice         = [c["redrob_signals"]["notice_period_days"] for c in candidates]
github         = [c["redrob_signals"]["github_activity_score"] for c in candidates]

print("\n=== OPEN TO WORK ===")
print(pd.Series(open_to_work).value_counts())

# How stale is the pool?
today = date.today()
days_inactive = [
    (today - datetime.strptime(d, "%Y-%m-%d").date()).days
    for d in last_active
]
print("\n=== DAYS SINCE LAST ACTIVE ===")
print(pd.Series(days_inactive).describe())
print("Active <30d:",  sum(1 for d in days_inactive if d < 30))
print("Active <90d:",  sum(1 for d in days_inactive if d < 90))
print("Inactive >180d:", sum(1 for d in days_inactive if d > 180))

print("\n=== NOTICE PERIOD ===")
print(pd.Series(notice).describe())
print("Sub-30d:", sum(1 for n in notice if n <= 30))

print("\n=== RESPONSE RATE ===")
print(pd.Series(response_rates).describe())

print("\n=== GITHUB ACTIVITY (-1 = no github) ===")
has_github = [g for g in github if g >= 0]
print(f"No GitHub linked: {sum(1 for g in github if g == -1)}")
print(pd.Series(has_github).describe())

# ── Honeypot detection preview ────────────────────────────────────
print("\n=== POTENTIAL HONEYPOTS (preview) ===")
honeypot_count = 0
for c in candidates:
    skills   = c["skills"]
    assessed = c["redrob_signals"].get("skill_assessment_scores", {})
    advanced = [s for s in skills if s["proficiency"] == "advanced"]
    
    # Flag: many advanced claims but low assessments
    low_scores = sum(
        1 for s in advanced
        if s["name"] in assessed and assessed[s["name"]] < 50
    )
    if len(advanced) >= 6 and low_scores >= 3:
        honeypot_count += 1

print(f"Candidates with skill inflation pattern: {honeypot_count}")

# ── How many are plausibly relevant? ─────────────────────────────
ML_TITLE_KEYWORDS = [
    "ml", "machine learning", "ai ", "data scientist",
    "nlp", "research", "applied", "ranking", "search",
    "recommendation", "retrieval", "platform engineer",
    "backend engineer", "software engineer"
]
plausible = sum(
    1 for t in titles
    if any(kw in t.lower() for kw in ML_TITLE_KEYWORDS)
)
print(f"\n=== PLAUSIBLY RELEVANT TITLES ===")
print(f"Count: {plausible} / {len(candidates)} "
      f"({100*plausible/len(candidates):.1f}%)")

india_candidates = sum(1 for c in countries if c == "India")
print(f"\n=== IN INDIA ===")
print(f"{india_candidates} / {len(candidates)} "
      f"({100*india_candidates/len(candidates):.1f}%)")