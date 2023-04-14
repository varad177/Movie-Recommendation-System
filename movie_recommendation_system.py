import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# getting databases

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# merging databses
movies = movies.merge(credits, on='title')

# print(movies.head(1))


# we will want
# genres
# id
# keywords
# title
# overview
# cast
# crew
#tags
movies = movies[['movie_id', 'title', 'overview',
                 'genres', 'keywords', 'cast', 'crew']]


# print(movies.isnull().sum())
# checking null
movies.dropna(inplace=True)
# print(movies.isnull().sum())

# print(movies)

# making function for preprocessing


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# print(movies.head())
# print(movies['keywords'])

# print(movies['keywords'])

# now for cast


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


movies['cast'] = movies['cast'].apply(convert3)
# print(movies['cast'])


# for crew
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)
# print(movies['crew'])

# for overview
movies['overview'] = movies['overview'].apply(lambda x: x.split())
# print(movies['overview'])

# removing spaces
movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x])
# print(movies['keywords'])
# print(movies['cast'])
# print(movies['crew'])

# print(movies.head())

movies['tags'] = movies['overview'] + movies['genres'] + \
    movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
# print(new_df)


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
# print(new_df.head())

new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
# print(new_df['tags'])


# text vectorization

# now we have to give 5 movies when user enter the name of movies
# so on the basis of taags we have to give the tags
# so for comparing the tags we have to convert the tags the into vector
# for converting the into vector
# we will use the technique i.e beg of words
# after converting tags into vector
# we have to extract the vector the words which are similar in the both
# and the basis of this we show that 5 movies
# in this process we will avoid the stops the stop words (e.g and, or )

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
# print(vectors)

# there are word which have same meaning
# so  for that i am using stemming
ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# print(ps.stem('loved'))


new_df['tags'] = new_df['tags'].apply(stem)
print(new_df['tags'])
# print(cv.get_feature_names())


# so now we have 4806 movies so we have 4806 vector
# now we have to calulate the distance between two movies
# so now as distance is long , the simlarity is less\
# we will not calculate the distance tip to tip
# so we will use cosine distance
# i.e is the theta between two movie vector


similarity = cosine_similarity(vectors)
# print(similarity)


# function for showing the 5 similar  movies

# print(new_df[new_df['title'] == 'Batman Begins'].index[0])

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True, key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)
       


# print(recommend('Avatar'))
# print(recommend('Batman Begins'))


#now data model is ready 
# so now converting into website 

# now for sharing the data we have to use pickle lib
pickle.dump(new_df, open('movies.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))


