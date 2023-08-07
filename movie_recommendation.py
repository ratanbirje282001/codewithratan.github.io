# movie_recommendation.py

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# Read data from CSV files
# movie_recommendation.py



# Read the movies and credits CSV files from the static folder
movies = pd.read_csv('static/tmdb_5000_movies.csv')
credits = pd.read_csv('static/tmdb_5000_credits.csv')

# Rest of your code here...

# Merge data on the 'title' column
movies = movies.merge(credits, on='title')

# Select required columns
movies = movies[['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]

# Preprocess the data
def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def conv_cast(obj):
    l = []
    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:
            l.append(i['name'])
            count = count + 1
        else:
            break
    return l

movies['cast'] = movies['cast'].apply(conv_cast)

def fetch_dir(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            l.append(i['name'])
            break
    return l

movies['crew'] = movies['crew'].apply(fetch_dir)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tag'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies['tag'] = movies['tag'].apply(lambda x: " ".join(x))
movies['tag'] = movies['tag'].apply(lambda x: x.lower())

ps = PorterStemmer()

def stem(text):
    y = []
    for j in text.split():
        y.append(ps.stem(j))
    return " ".join(y)

movies['tag'] = movies['tag'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tag']).toarray()
similarity = cosine_similarity(vectors)

# Function to get recommended movies
def get_recommended_movies(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    
    return recommended_movies


