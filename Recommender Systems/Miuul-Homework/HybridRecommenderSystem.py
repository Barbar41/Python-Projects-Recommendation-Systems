#################
# Association RuleBased Recommender System
##############

# Business Problem:
# Make 10 movie recommendations for the user whose ID is given, using the item-based and user-based recommender methods.

##############
# Dataset Story
##############
# The dataset is provided by MovieLens, a movie recommendation service. It contains the movies as well as the rating points made for these movies.
# Includes 2,000,0263 ratings across 27,278 movies. This data set was created on October 17, 2016.
# 138,493 users and includes data between January 09, 1995 and March 31, 2015. Users were selected randomly.
# It is known that all selected users voted for at least 20 movies.


#######################
# TASK 1--Preparing the Data
#######################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.expand_frame_repr', False)

# Step 1: Read the movie and rating data sets.
movie = pd.read_csv(r"Recommendation Systems/datasets/movie_lens_dataset/movie.csv")
movie.head()
movie.shape

rating = pd.read_csv(r"Recommendation Systems/datasets/movie_lens_dataset/rating.csv")
rating.head()
rating.shape
rating["userId"].nunique()

# Step 2: Add the movie names and genres of the Ids from the movie data set to the rating data set.
df=movie.merge(rating, how="left", on="movieId")
df.head(1)
df.head(50)
df.shape

# Step3: Keep the names of the movies whose total number of votes is below 1000 in the list and remove them from the data set.
comment_counts=pd.DataFrame(df["title"].value_counts())
comment_counts

rare_movies=comment_counts[comment_counts["title"] <= 1000].index

common_movies=df[~df["title"].isin(rare_movies)]
common_movies.head(2)
common_movies.shape


# Step 4: Create a pivot table for the dataframe with userIDs in the index, movie names in the columns, and ratings as values.
user_movie_df=common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()

# Step 5: Functionalize all operations performed.
def create_user_movie_df():
     import pandas as pd
     movie = pd.read_csv(r"Recommendation Systems/datasets/movie_lens_dataset/movie.csv")
     rating = pd.read_csv(r"Recommendation Systems/datasets/movie_lens_dataset/rating.csv")
     df = movie.merge(rating, how="left", on="movieId")
     comment_counts = pd.DataFrame(df["title"].value_counts())
     rare_movies = comment_counts[comment_counts["title"] <= 1000].index
     common_movies = df[~df["title"].isin(rare_movies)]
     user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
     return user_movie_df

user_movie_df=create_user_movie_df()

#######################
# Task 2: Determining the Movies Watched by the User to Make a Recommendation
#######################
# Step 1: Choose a random user ID.
random_user= 108170

# Step 2: Create a new dataframe named random_user_df consisting of observation units of the selected user.
random_user_df= user_movie_df[user_movie_df.index == random_user] #There are indexes in the pivot table, so index
random_user_df.head(2)
random_user_df.isnull().sum()

# Step 3: Assign the movies voted by the selected users to a list called movies_watched
movies_watched=random_user_df.columns[random_user_df.notna().any()].tolist()

#######################
# Task 3: Accessing the Data and IDs of Other Users Watching the Same Movies
#######################
# Step 1: Select the columns of the movies watched by the selected user from user_movie_df and create a new dataframe named movies_watched_df.
user_movie_df.head()
movies_watched_df= user_movie_df[movies_watched]
movies_watched_df.head(2)


# Step 2: Create a new dataframe named user_movie_count, which contains information about how many movies each user has watched.
user_movie_count= movies_watched_df.T.notnull().sum()
user_movie_count=user_movie_count.reset_index() #reset index converts index to column
user_movie_count.columns=["userId", "movie_count"]
user_movie_count.index

# Step 3: Create a list named users_same_movies from the user IDs of those who watched 60 percent or more of the movies voted by the selected user.
perc= len(movies_watched) * 60/ 100
random_user_df.shape

users_same_movies= user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


#######################
# Task 4: Determining the Users Most Similar to the User to Make a Recommendation
#######################

# Step 1: Filter the movies_watched_df dataframe to find the IDs of users that are similar to the selected user in the user_same_movies list.
final_df= movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()

# Step 2: Create a new corr_df dataframe in which the correlations of users with each other will be found.
corr_df= final_df.T.corr().unstack().sort_values() # - There is a negative correlation in the first values
corr_df=pd.DataFrame(corr_df, columns=["corr"]) # we are transferring dataframe.
corr_df.index.names=["userId_1","userId_2"]#we name the indexes
corr_df=corr_df.reset_index()


# Step 3: Create a new dataframe named top_users by filtering out users with high correlation (over 0.65) with the selected user.
corr_df[corr_df["userId_1"]== random_user]
top_users= corr_df[(corr_df["userId_1"]== random_user)& (corr_df["corr"] >= 0.65)][["userId_2", "corr"]].reset_index(drop=True)

# Step 4: Merge the top_users dataframe with the rating data set.
top_users.rename(columns={"userId_2":"userId"},inplace=True)

top_users_ratings=top_users.merge(rating[["userId","movieId", "rating"]],how="inner")
top_users_ratings["userId"].nunique()

#######################
# Task 5: Calculating the Weighted Average Recommendation Score and Keeping the Top 5 Movies
#######################
# Step 1: Create a new variable named weighted_rating, which is the product of each user's corr and rating values.

top_users_ratings["weighted_rating"]= top_users_ratings["corr"]* top_users_ratings["rating"]


# Step 2: Create a new dataframe named recommendation_df, which contains the movie id and the average value of all users' weighted ratings for each movie.
recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating":"mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()


# Step 3: Select the movies with a weighted rating greater than 3.5 in recommendation_df and sort them according to the weighted rating.
recommendation_df[recommendation_df["weighted_rating"]> 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"]> 3.5].sort_values("weighted_rating", ascending=False)

# Step 4: Bring the movie names from the movie data set and select the top 5 movies to recommend.
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][0:5]

#######################
#### Item Based Recommendation #####
#######################
#######################
# TASK 1--Make an item-based recommendation based on the last and highest rated movie the user watched.
#######################

user = 108170

# Step 1: Read the movie and rating data sets
movie = pd.read_csv(r"Recommendation Systems/datasets/movie_lens_dataset/movie.csv")
rating = pd.read_csv(r"Recommendation Systems/datasets/movie_lens_dataset/rating.csv")

# Step 2: Get the ID of the movie with the most current score among the movies that the selected user gave 5 points.
movie_id= rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1 ].values[0]

# Step 3: Filter the user_movie_df dataframe created in the User based recommendation section according to the selected movie id.
movie_df=user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]


# Step 4: Using the filtered dataframe, find the correlation between the selected movie and other movies and rank them.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Functionalization
def item_based_recommender(movie_name, user_movie_df):
     movie=user_movie_df[movie_name]
     return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Step 5: Give the first 5 movies as suggestions, apart from the selected movie itself.
movies_from_item_based = item_based_recommender(movie[movie["movieId"]== movie_id]["title"].values[0],user_movie_df)
# 1 through 6.
# 0 has the movie itself.
# We left it out
movies_from_item_based[1:6].index