# Task 3: Movie Recommendation System (Content-Based Filtering)

# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 1. Load Dataset
# ---------------------------
# Download MovieLens dataset from Kaggle (movies.csv + ratings.csv)
movies = pd.read_csv("movies.csv")   # contains movieId, title, genres

# ---------------------------
# 2. Preprocessing
# ---------------------------
# Fill missing values in genres
movies['genres'] = movies['genres'].fillna('')

# Create TF-IDF Matrix for genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reset index to map between titles and indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ---------------------------
# 3. Recommendation Function
# ---------------------------
def recommend_movies(title, num_recommendations=5):
    if title not in indices:
        return f"❌ Movie '{title}' not found in dataset."

    idx = indices[title]  # index of the movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # skip the movie itself

    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# ---------------------------
# 4. User Input
# ---------------------------
user_movie = input("🎬 Enter a movie you like: ")
recommendations = recommend_movies(user_movie, 5)

print("\n✅ You may also like:")
for i, rec in enumerate(recommendations, start=1):
    print(f"{i}. {rec}")
