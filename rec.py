#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[ ]:


credits=pd.read_csv('C:\Users\Sachin\Desktop\machine learning\movie recomondation\tmdb_5000_credits.csv')
movies=pd.read_csv('C:\Users\Sachin\Desktop\machine learning\movie recomondation\tmdb_5000_movies.csv')


# In[ ]:


movies.head(1)


# In[ ]:


credits.head(1)


# In[ ]:


movies=movies.merge(credits,on='title')


# In[ ]:


movies.head(1)


# In[ ]:


#genres
#id
#keywords
#title
#overview
#cast
#crew

movies=movies[['movie_id','genres','keywords','title','overview','cast','crew']]


# In[ ]:


movies.head(1)


# In[ ]:


movies.isnull().sum()


# In[ ]:


movies.dropna(inplace=True)


# In[ ]:


movies.duplicated().sum()


# In[ ]:


movies.iloc[0].genres


# In[ ]:


import ast
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[ ]:


movies['genres']=movies['genres'].apply(convert)


# In[ ]:


movies.head()


# In[ ]:


movies['keywords']=movies['keywords'].apply(convert)


# In[ ]:


movies.head()


# In[ ]:


import ast
def conv_cast(obj):
    l=[]
    count=0
    for i in ast.literal_eval(obj):
        if count!=3:
            l.append(i['name'])
            count=count+1
        else:
            break
    return l
movies['cast']=movies['cast'].apply(conv_cast)


# In[ ]:


movies.head()


# In[ ]:


import ast
def fetch_dir(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=="Director":
            l.append(i['name'])
            break
    return l

movies['crew']=movies['crew'].apply(fetch_dir)


# In[ ]:


movies.head()


# In[ ]:


movies['overview']=movies['overview'].apply(lambda x: x.split())


# In[ ]:


movies.head()


# In[ ]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[ ]:


movies.head()


# In[ ]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[ ]:


movies.head()


# In[ ]:


movies['tag']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[ ]:


movies.head()


# In[ ]:


new_df=movies[['movie_id','title','tag']]


# In[ ]:


new_df.head()


# In[ ]:


new_df['tag']=new_df['tag'].apply(lambda x: " ".join(x))


# In[ ]:


new_df.head()


# In[ ]:


new_df['tag']=new_df['tag'].apply(lambda x: x.lower())


# In[ ]:


new_df.head()


# In[ ]:


import nltk


# In[ ]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[ ]:


def stem(text):
    y=[]
    for j in text.split():
        y.append(ps.stem(j))
    return " ".join(y)


# In[ ]:


new_df['tag']=new_df['tag'].apply(stem)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[ ]:


vectors=cv.fit_transform(new_df['tag']).toarray()


# In[ ]:


vectors[0]


# In[ ]:


cv.get_feature_names_out()


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


similarity=cosine_similarity(vectors)


# In[ ]:


similarity[0]


# In[ ]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[ ]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distance=similarity[movie_index]
    movie_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
    


# In[ ]:


recommend('Avatar')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from flask import Flask, render_template, request

# Add your recommend function here (from the previous code)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movie():
    movie = request.form['movie']
    recommended_movies = recommend(movie)
    return render_template('recommendations.html', movie=movie, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
