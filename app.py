import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pandas as pd 

# Reading Rating Data
ratings = pd.read_csv("data/u.data", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp']) #Enter The Path 

# Reading Movies Info
movies=pd.read_csv("data/u.data",sep='|' , encoding='latin_1',usecols=[0,1] , names=['item_id','title']) #Enter The Path

#Merging Them Together
data=pd.merge(ratings , movies , on='item_id')

print(data.head())

print(f'Number Of Users : {data["user_id"].nunique()}')

print(f'number of movies : {data["item_id"].nunique()}'.capitalize())

#Distrubution Ratings
print(data["rating"].value_counts())

#The Intermediate Evaluation Of Each Movie
avg_rating=data.groupby("title")["rating"].mean().sort_values(ascending=False)
print(avg_rating.head(10))

movie_cols=['item_id' , 'title']+[f'genre_{i}' for i in range(19)]

movies_full=pd.read_csv("C:\\Users\\Asus\\Downloads\\ml-100k\\ml-100k\\u.item", sep='|', encoding='latin_1', usecols=range(21) , names=movie_cols)

# We collect the genres for each movie into a single text string.
def extract_genres(row):
    genres=[]
    genres_labels = ['Unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi',
                'Thriller', 'War', 'Western']
    for i in range(19):
        if row[f'genre_{i}']==1:
            genres.append(genres_labels[i])
    return ' '.join(genres)

movies_full['genres']=movies_full.apply(extract_genres , axis=1)
movies_content=movies_full[['item_id','title','genres']]

print(movies_content.head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Digital text representation
tfidf=TfidfVectorizer()
tfdif_matrix=tfidf.fit_transform(movies_content['genres'])

#Similarity Matrix
cos_sim=cosine_similarity(tfdif_matrix)

# Build a filter function
indices=pd.Series(movies_content.index , index=movies_content['title'])

def Recommend(title,n=5):
    idx=indices[title]
    sim_scores=list(enumerate(cos_sim[idx]))
    sim_scores=sorted(sim_scores , key=lambda x: x[1] , reverse=True)
    sim_scores=sim_scores[1:n+1]
    movie_indecis=[i[0] for i in sim_scores]
    return movies_content.iloc[movie_indecis][['title' , 'genres']]
print(Recommend ('Toy Story (1995)',5))

# Preparing The Data For The Model
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

#Select The Form Of Data 
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# Training , Testing The Data
trainset, testset = train_test_split(data, test_size=0.2 , random_state=42 , shuffle=True)

# Build The Model With SVD
from surprise import SVD
from surprise import accuracy

model=SVD()
model.fit(trainset)

prediction=model.test(testset)

print(f' The RMSE is : {accuracy.rmse(prediction)}')

# Function For Recommended User 
def recommended_for_user(user_id , n=5):
    all_items=ratings['user_id'].unique()
    rated_items=ratings[ratings['user_id'] == user_id]['item_id']
    unrated_items=[item for item in all_items if item not in rated_items.values]
    
    predictions= [ (item , model.predict(user_id , item).est)for item in unrated_items]
    predictions.sort(key=lambda x: x[1] , reverse=True)

    top_items=[item for item ,  _ in predictions[:n]]
    return movies[movies['item_id'].isin(top_items)]
recommended_for_user(5)
# تحميل البيانات
ratings = pd.read_csv("data/u.data", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp']) #Enter The Path
movies = pd.read_csv("data/u.data", sep='|', encoding='latin-1', usecols=[0, 1], names=['item_id', 'title']) #Enter The Path

# تجهيز Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)
trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)

# دالة التوصية
def recommend_movies(user_id, n=5):
    all_items = ratings['item_id'].unique()
    rated = ratings[ratings['user_id'] == user_id]['item_id']
    unrated = [item for item in all_items if item not in rated.values]
    
    predictions = [(item, model.predict(user_id, item).est) for item in unrated]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_items = [item for item, _ in predictions[:n]]
    return movies[movies['item_id'].isin(top_items)]

# واجهة Streamlit
st.title("🎬 Recommended Movies System")

user_id = st.number_input("Enter Number Of User:", min_value=1, max_value=943, step=1)

if st.button("Recommended Movies"):
    recommendations = recommend_movies(int(user_id))
    st.write("🎥 Recommended Movies:")
    st.table(recommendations[['title']].reset_index(drop=True))
