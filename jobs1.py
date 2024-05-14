import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds

def clean_experience(experience):
    numbers = re.findall("\d+", experience)
    return [int(numbers[0]), int(numbers[-1])] if numbers else [0, 0]

data = pd.read_csv("jobs_info.csv")
data["Experience Range"] = data["Job Experience"].apply(clean_experience)

# Collaborative Filtering
def collaborative_filtering(data):
    user_item_matrix = pd.pivot_table(data, values='Rating', index='User_ID', columns='Job_ID', fill_value=0)
    U, sigma, Vt = svds(user_item_matrix, k=50)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_item_matrix.columns)
    return preds_df

# Content-Based Filtering
skills_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
title_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

tfidf_skills = skills_vectorizer.fit_transform(data["Key Skills"])
tfidf_titles = title_vectorizer.fit_transform(data["Job Title"])

# Additional job-related features for content-based filtering
additional_features = data[['Job Description', 'Company', 'Salary', 'Location']]  # Add more features as needed

def content_based_filtering(query_skills, query_title):
    query_skills_vec = skills_vectorizer.transform([query_skills])
    query_title_vec = title_vectorizer.transform([query_title])

    skills_similarity = cosine_similarity(query_skills_vec, tfidf_skills).flatten()
    title_similarity = cosine_similarity(query_title_vec, tfidf_titles).flatten()

    # Normalize similarities
    skills_similarity = (skills_similarity - skills_similarity.min()) / (skills_similarity.max() - skills_similarity.min() + 1e-5)
    title_similarity = (title_similarity - title_similarity.min()) / (title_similarity.max() - title_similarity.min() + 1e-5)

    combined_similarity = (skills_similarity + title_similarity) / 2

    return combined_similarity

def recommend_jobs(query_skills, query_title, query_experience, user_id=None):
    # Collaborative Filtering
    collaborative_predictions = collaborative_filtering(data)
    
    # Content-Based Filtering
    content_based_scores = content_based_filtering(query_skills, query_title)
    
    # Combine scores from collaborative and content-based filtering
    combined_scores = 0.7 * collaborative_predictions + 0.3 * content_based_scores
    
    # Apply experience similarity and adjust combined score
    experience_scores = np.array([experience_similarity(query_experience, x) for x in data["Experience Range"]])
    combined_scores *= experience_scores
    
    # Get top recommendations
    indices = np.argsort(-combined_scores)[:10]
    if len(indices) == 0 or combined_scores[indices[0]] == 0:  # Check if there are no recommendations
        return []

    results = data.iloc[indices]

    return results.to_dict(orient='records')

# Example usage
# results = recommend_jobs('java sql linux', 'software developer', 2)
# print(results)
# results = recommend_jobs('python sql', 'data analyst', 3)
# print(results)
