from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Function to convert data from dataframe
def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

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

def fetch_dir(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            l.append(i['name'])
            break
    return l

def stem(text):
    y = []
    for j in text.split():
        y.append(ps.stem(j))
    return " ".join(y)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [new_df.iloc[i[0]].title for i in movie_list]
    return recommended_movies

# Read the data
credits = pd.read_csv('C:/Users/Sachin/Desktop/machine learning/movie recomondation/tmdb_5000_credits.csv')
movies = pd.read_csv('C:/Users/Sachin/Desktop/machine learning/movie recomondation/tmdb_5000_movies.csv')

# Merge the dataframes on 'title' column
movies = movies.merge(credits, on='title')

# Select the columns of interest
movies = movies[['movie_id', 'genres', 'keywords', 'title', 'overview', 'cast', 'crew']]

# Remove rows with missing values
movies.dropna(inplace=True)

# Convert string representation of lists to actual lists
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(conv_cast)
movies['crew'] = movies['crew'].apply(fetch_dir)

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tag'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tag']]

# ... Your previous code ...

# ... Your previous code ...

# Convert the 'tag' column using .apply() and .loc to avoid warnings
new_df['tag'] = new_df['tag'].apply(lambda x: " ".join(x))
new_df['tag'] = new_df['tag'].apply(lambda x: x.lower())

# ... The rest of your code ...


# ... The rest of your code ...

# ... The rest of your code ...


ps = PorterStemmer()
new_df['tag'] = new_df['tag'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tag']).toarray()

similarity = cosine_similarity(vectors)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/execute', methods=['POST'])
def execute_code():
    input_movie = request.form['input_movie']

    # TODO: Validate and sanitize the input_movie

    try:
        # Call the recommend function to get the list of recommended movies
        recommended_movies = recommend(input_movie)

        return render_template('result.html', input_movie=input_movie, recommended_movies=recommended_movies)
    except Exception as e:
        # Handle any exceptions that might occur during code execution
        error_message = str(e)
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
