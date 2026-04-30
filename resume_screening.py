# ===============================
# Professional Resume Screening System
# Future Interns - Task 3
# ===============================

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Sample Data (You can replace with CSV later)
# -------------------------------
data = {
    "Candidate": ["Candidate 1", "Candidate 2", "Candidate 3"],
    "Resume": [
        "Python Machine Learning SQL Data Analysis",
        "Java HTML CSS Web Development",
        "Python Deep Learning TensorFlow NLP"
    ]
}

job_description = "Python Machine Learning Data Analysis"

# Convert to DataFrame
df = pd.DataFrame(data)

# -------------------------------
# 2. Text Preprocessing Function
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# Apply preprocessing
df["Clean_Resume"] = df["Resume"].apply(preprocess)
job_clean = preprocess(job_description)

# -------------------------------
# 3. TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer()

all_text = df["Clean_Resume"].tolist() + [job_clean]
X = vectorizer.fit_transform(all_text)

# -------------------------------
# 4. Similarity Calculation
# -------------------------------
scores = cosine_similarity(X[:-1], X[-1]).flatten()

# Add scores to DataFrame
df["Score"] = scores

# -------------------------------
# 5. Ranking Candidates
# -------------------------------
df = df.sort_values(by="Score", ascending=False)

print("\n===== Candidate Ranking =====")
print(df[["Candidate", "Score"]])

# -------------------------------
# 6. Skill Extraction & Missing Skills
# -------------------------------
job_skills = set(job_clean.split())

print("\n===== Missing Skills =====")
for index, row in df.iterrows():
    resume_skills = set(row["Clean_Resume"].split())
    missing = job_skills - resume_skills
    print(f"{row['Candidate']} Missing Skills: {missing}")

# -------------------------------
# 7. Best Candidate
# -------------------------------
best_candidate = df.iloc[0]

print("\n===== Best Candidate =====")
print(best_candidate["Candidate"])

# -------------------------------
# 8. Save Results (Professional Touch)
# -------------------------------
df.to_csv("ranking_results.csv", index=False)

print("\nResults saved to ranking_results.csv")
