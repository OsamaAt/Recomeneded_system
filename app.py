import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation Demo (No Surprise!)")

# -------------------------
# Load the data
# -------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# -------------------------
# Create user-movie matrix
# -------------------------
@st.cache_data
def build_matrix(ratings):
    user_movie_matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)
    return user_movie_matrix

user_movie_matrix = build_matrix(ratings)

# Compute cosine similarity
@st.cache_data
def compute_similarity(matrix):
    cosine_sim = cosine_similarity(matrix)
    return cosine_sim

similarity = compute_similarity(user_movie_matrix)

# -------------------------
# Recommendation function
# -------------------------
def recommend_movies(user_id, top_n=5):
    if user_id not in user_movie_matrix.index:
        return []

    user_idx = user_movie_matrix.index.tolist().index(user_id)
    sim_scores = list(enumerate(similarity[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1 : top_n + 1]
    similar_users = [user_movie_matrix.index[i] for i, score in sim_scores]

    rec_movies = (
        ratings[ratings["userId"].isin(similar_users)]
        .groupby("movieId")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    return movies[movies["movieId"].isin(rec_movies)]["title"].tolist()


# -------------------------
# Streamlit UI
# -------------------------
st.subheader("Enter a user ID to get recommendations")

user_id = st.number_input("User ID", min_value=1, step=1)

if st.button("Recommend"):
    results = recommend_movies(int(user_id))
    if results:
        st.success("Recommended Movies:")
        for m in results:
            st.write(f"- {m}")
    else:
        st.warning("No recommendations for this user.")
