#######################
# Item-Based Collaborative Filtering
#######################

# Dataset: https://grouplens.org/datasets/movielens/

# Step 1: Preparing the Data Set
# Step 2: Creating User Movie Df
# Step 3: Making Item-Based Movie Recommendations
# Step 4: Preparing the Working Script

#######################
# Step 1: Preparing the Data Set
#######################
import pandas as pd
pd.set_option('display.max_columns', 20)
movie = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/rating.csv')

# We merge two csv files according to MoveId.
df = movie.merge(rating, how="left", on="movieId")
df.head()

#######################
# Step 2: Creating User Movie Df
#######################

# The general problem is sparsity in the matrix.

df.head()
df.shape

# How many unique movies are there
df["title"].nunique()

# Let's see which movie has what rate
df["title"].value_counts().head()

# Let's convert value counts to df
comment_counts= pd.DataFrame(df["title"].value_counts())

# Let's see the movies that are less than a certain number first.
comment_counts[comment_counts["title"] <= 1000]

# To access the names here
rare_movies=comment_counts[comment_counts["title"] <= 1000].index

# In order to get rid of the low rated movies here, there should be a title that does not contain these names.

common_movies= df[~df["title"].isin(rare_movies)]

common_movies.shape

common_movies["title"].nunique()
df["title"].nunique()

# The reduction process has been completed and we have movies that have received more than 1000 ratings.
# Let users be in the rows and titles in the columns.

user_movie_df= common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.columns

#######################
# Step 3: Making Item-Based Movie Recommendations
#######################
pd.set_option('display.max_columns', 500)

movie_name= "Matrix, The (1999)"
movie_name= user_movie_df[movie_name]

# We use correlation formula
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# Let's try for another movie
movie_name= "Ocean's Twelve (2004)"
movie_name= user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# Access by selecting random movies
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name= user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# Random movie options Functionality;
def check_movie(keyword, user_movie_df):
     return[col for col in user_movie_df.columns if keyword in col]

check_film("Sherlock", user_movie_df)
check_film("Insomnia", user_movie_df)

#######################
# Step 4: Preparing the Working Script
#######################

def create_user_movie_df():
     import pandas as pd
     movie = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/movie.csv')
     rating = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/rating.csv')
     df = movie.merge(rating, how="left", on="movieId")
     comment_counts = pd.DataFrame(df["title"].value_counts())
     rare_movies = comment_counts[comment_counts["title"] <= 1000].index
     common_movies = df[~df["title"].isin(rare_movies)]
     user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
     return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
     movie_name = user_movie_df[movie_name]
     return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("The Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)