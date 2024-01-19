##############################
# Content Based Recommendation
#################

#################
# Developing Recommendations Based on Movie Reviews
#################

#1. Creating the TF-IDF Matrix
#2. Creating the Cosine Similarity Matrix
#3. Making Suggestions Based on Similarities
# 4. Preparation of Working Script

####################
#1. Creating the TF-IDF Matrix
####################


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', 100)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Recommendation Systems\datasets\the_movies_dataset\movies_metadata.csv", low_memory=False ) # DtypeWarning to turn off.
df.head()
df.shape

# Our main focus is the overview variable for similarity calculations
df["overview"].head()

# The method just imported is being called.
# Too much space is made to get ahead of the situation. It has no measurement value (in, on, and..).
tfidf= TfidfVectorizer(stop_words="english")

# Let's get rid of the shortcomings in the "overview" variable.

df[df["overview"].isnull()]
# Let's fill the unimportant ones with a space.
df["overview"]=df["overview"].fillna("")

# Overview transformation

tfidf_matrix= tfidf.fit_transform(df["overview"])

tfidf_matrix.shape
# rows contain descriptions, columns contain unique words

# verification that the lines are movies
df["title"].shape

# So what is at the intersection of the two? There are tfidf scores.

tfidf.get_feature_names()

# Filtering
df = df[~df["title"].duplicated(keep="last")]
df = df[~df["title"].isna()]
df = df[~df["overview"].isna()]

# Halving the size of the matrix in memory.
tfidf_matrix = tfidf_matrix.astype(np.float32)

# We want to access their intersections.

tfidf_matrix.toarray()

####################
#2. Creating the Cosine Similarity Matrix
####################

cosine_sim=cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
cosine_sim[1]

####################
#3. Making Suggestions Based on Similarities
####################

# We will create a pandas series and place the titles of the movies in this series.
indices= pd.Series(df.index, index=df["title"])

indices.index.value_counts()

# We are destroying the previous movies. We will take the last one of the multiplexed names. We want the last movie shot. We are getting rid of the multiplexed records.
indices=indices[~indices.index.duplicated(keep="last")]

indices["Cinderella"]

indices["Sherlock Holmes"]

# Let's keep a movie index
movie_index=indices["Sherlock Holmes"]

cosine_sim[movie_index]

# We calculated the similarity scores of the Sherlock Holmes movie and the movies I had. And placed them in the Database.
similarity_scores=pd.DataFrame(cosine_sim[movie_index],
                                columns=["Score"])

# If we want to bring the 10 movies with the highest score
movie_indices= similarity_scores.sort_values("Score",
                                              ascending=False)[1:11].index

# I want to go to the names of the movies.
df["title"].iloc[movie_indices]

####################
# 4. Preparation of Working Script
####################

def content_based_recommender(title, cosine_sim, dataframe):
     # create indexes
     indices = pd.Series(dataframe.index, index=dataframe['title'])
     indices = indices[~indices.index.duplicated(keep='last')]
     # Capture title's index
     movie_index = indices[title]
     # calculate similarity scores based on title
     similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
     # don't bring the top 10 movies except himself
     movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
     return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes",cosine_sim,df)

content_based_recommender("The Matrix",cosine_sim,df)

content_based_recommender("The Dark Knight Rises",cosine_sim,df)

content_based_recommender("The Godfather",cosine_sim,df)

def calculate_cosine_sim(dataframe):
     tfidf = TfidfVectorizer(stop_words='english')
     dataframe['overview'] = dataframe['overview'].fillna('')
     tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
     return cosine_sim

cosine_sim = calculate_cosine_sim(df)