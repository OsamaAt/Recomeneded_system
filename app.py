import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Uploading and preparing the dataset
ratings = pd.read_csv("data/u.data", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv("data/u.item", sep='|', encoding='latin_1', usecols=[0, 1], names=['item_id', 'title'])

# Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)
trainset = data.build_full_trainset()

# Training the SVD model
model = SVD()
model.fit(trainset)

# Recommendation function
def recommend_movies(user_id, n=5):
    all_items = ratings['item_id'].unique()  
    rated = ratings[ratings['user_id'] == user_id]['item_id']
    unrated = [item for item in all_items if item not in rated.values]

    predictions = [(item, model.predict(user_id, item).est) for item in unrated]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_items = [item for item, _ in predictions[:n]]
    
    return movies[movies['item_id'].isin(top_items)]

#  Streamlit
st.title("🎬 Movie Recommendation System")

user_id = st.number_input("Enter User ID (1–943):", min_value=1, max_value=943, step=1)

if st.button("Recommend Movies"):
    recommendations = recommend_movies(int(user_id))
    st.write("🎥 Recommended Movies:")
    st.table(recommendations[['title']].reset_index(drop=True))
