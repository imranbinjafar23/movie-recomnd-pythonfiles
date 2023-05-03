import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import pickle
import warnings
import ast
import nltk

# Set up warnings
warnings.filterwarnings("ignore")

# Read in the data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['movie_id','title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies = movies.dropna()

# Define conversion functions
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def stem(text):
    ps = PorterStemmer()
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Apply conversion functions to data
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['tag'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] 

# Create new dataframe for processed data
new_df = movies[['movie_id', 'title', 'tag']]
new_df['tag'] = new_df['tag'].apply(lambda x: " ".join(x))
new_df['tag'] = new_df['tag'].apply(lambda x:x.lower())

# Convert tags to vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
new_df['tag'] = new_df['tag'].apply(stem)
vectors = cv.fit_transform(new_df['tag']).toarray()
similarity = cosine_similarity(vectors)

# Define recommendation function
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Save processed data and similarity matrix to file
pickle.dump(new_df, open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))
