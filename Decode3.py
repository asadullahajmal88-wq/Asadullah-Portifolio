import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# create mock dataset
data = {
    "job_role": [
        "DevOps Engineer",
        "Data Scientist",
        "Backend Developer",
        "Frontend Developer",
        "Machine Learning Engineer",
        "Cloud Architect",
        "Database Administrator",
        "Full Stack Developer",
    ],
    "skills": [
        "Cloud Computing AWS Docker Kubernetes CI/CD Automation Linux Terraform",
        "Python Statistics Machine Learning Pandas NumPy SQL Data Visualization",
        "Python Java Node.js SQL REST API Microservices Backend Databases",
        "JavaScript React CSS HTML UI UX Frontend Web Design",
        "Python Machine Learning TensorFlow PyTorch Deep Learning Automation Cloud Computing",
        "Cloud Computing AWS Azure GCP Architecture Networking Automation Security",
        "SQL Database Design Indexing Backup Recovery Performance Tuning Linux",
        "Python JavaScript React Node.js SQL REST API Cloud Computing",
    ],
}
df = pd.DataFrame(data)
df.to_csv("raw_skills.csv", index=False)

df = pd.read_csv("raw_skills.csv")

# step 1: ingestion
user_skills = ["Python", "Cloud Computing", "Automation"]
user_profile = " ".join(user_skills)

# step 2: scoring
corpus = df["skills"].tolist() + [user_profile]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

job_vectors = tfidf_matrix[:-1]
user_vector = tfidf_matrix[-1]

scores = cosine_similarity(user_vector, job_vectors).flatten()
df["similarity"] = scores

# step 3: sorting
df = df.sort_values(by="similarity", ascending=False).reset_index(drop=True)

# step 4: filtering top 3
top3 = df.head(3).copy()
top3["match"] = (top3["similarity"] * 100).round(2)

print("User profile:", user_profile)
print()
for i, row in top3.iterrows():
    print(f"{row['job_role']} - {row['match']}% match")