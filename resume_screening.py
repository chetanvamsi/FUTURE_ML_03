from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample resumes
resume1 = "Python Machine Learning SQL"
resume2 = "Java HTML CSS"

# Job description
job = "Python Machine Learning"

# Convert text to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([resume1, resume2, job])

# Calculate similarity
similarity = cosine_similarity(X)

# Print scores
score1 = similarity[0][2]
score2 = similarity[1][2]

print("Candidate 1 Score:", score1)
print("Candidate 2 Score:", score2)

# Ranking
if score1 > score2:
    print("Best Candidate: Candidate 1")
else:
    print("Best Candidate: Candidate 2")

# Missing skills (for Candidate 1)
job_skills = set(job.lower().split())
resume_skills = set(resume1.lower().split())

missing = job_skills - resume_skills
print("Missing skills for Candidate 1:", missing)