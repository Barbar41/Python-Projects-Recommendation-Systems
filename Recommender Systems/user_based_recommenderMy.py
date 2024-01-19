#######################
# User-Based Collaborative Filtering
#######################

# Step 1: Preparing the Data Set
# Step 2: Determining the Movies Watched by the User to Make a Recommendation
# Step 3: Accessing the Data and IDs of Other Users Watching the Same Movies
# Step 4: Determining the Users with the Most Similar Behavior to the User to Make a Recommendation
# Step 5: Calculating the Weighted Average Recommendation Score
# Step 6: Functionalization of the Work


# Step 1: Preparing the Data Set
#######################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


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

#######################
# Step 2: Determining the Movies Watched by the Main User to Make Recommendations
#######################

import pandas as pd
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# First of all, to find out which movies the main user has watched.

random_user
user_movie_df

# Movies that the Main User has watched and not watched
random_user_df= user_movie_df[user_movie_df.index == random_user]

# Movie names watched by the Main User
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Let's do validation, let's do selection in rows and columns
user_movie_df.loc[user_movie_df.index == random_user,
                   user_movie_df.columns == "Sense and Sensibility (1995)"]

# How many movies has the Main User watched?
len(movies_watched)

#######################
# Step 3: Accessing the Data and IDs of Other Users Watching the Same Movies
#######################

# First, let's reduce the data set by asking the list of watched movies from user movie df.
# So now we have information about the movies watched

movies_watched_df = user_movie_df[movies_watched]
# We customize the data and move from general to possible solutions that are as close to custom as possible.

# We are in the title of accessing the IDs of users who watch the same movies.
# So, should all users who have watched at least one movie be entered into this data?
# Those who watch a small number of movies together will not benefit, so a limit should be determined, such as 10 movies together.
# Has a user watched the movie or not?
user_movie_count= movies_watched_df.T.notnull().sum()

# We convert the UserId in this index into a variable.
user_movie_count=user_movie_count.reset_index()

# Let's name it
user_movie_count.columns=["userId", "movie_count"]

Let's bring in the ones greater than #20.

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

# How many users have watched all the movies watched by the main user.
user_movie_count[user_movie_count["movie_count"] == 33].count()

# We need the IDs of these users
users_same_movies= user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

# To make this section more programmatic
# perc= len(movies_watched)*60/100
# users_same_movies=user_movie_count[user_movie_count["movie_count"]> perc]["userId"]

#######################
# Step 4: Determining the Users with the Most Similar Behavior to the User to Make a Recommendation
#######################

# For this we will perform 3 steps:
# 1. We will aggregate the data of the Main User and other users.
# 2. We will create the correlation df.
# 3. We will find the most similar users (Top Users)

# Let's bring together the previous data sets.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                     random_user_df[movies_watched]])

# We need to include users in the columns.
corr_df= final_df.T.corr().unstack().sort_values().drop_duplicates()

# Let's convert a classical df, then let's name this df in a way that touches the nomenclatures

corr_df=pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names= ["user_id_1", "user_id_2"]

corr_df= corr_df.reset_index


# High correlations with the main user are required.
# If we take Main User for UserId_1 and other users for UserId_2 and
# If we say bring users whose correlation is above a certain ratio,
# We find users who show similar behavior to the main user.
# There should be a positive correlation above 65%. Let's capture the main user relationship.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
     ["user_id_2", "corr"]].reset_index(drop=True)

# Let's sort it.
top_users= top_users.sort_values(by="corr", ascending=False)

# Let's name it
top_users.rename(columns={"user_id_2":"userId"},inplace=True)

# It is not known how many points these users gave to which movie.

# Let's combine the rating file with top users.
rating= pd.read_csv("Recommendation Systems/datasets/movie_lens_dataset/rating.csv")
top_users_ratings= top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

# Let's remove the main user from the work.
top_users_ratings= top_users_ratings[top_users_ratings["userId"] != random_user]

#######################
# Step 5: Calculating the Weighted Average Recommendation Score
#######################

# Only if there are ratings and users similar to the main user, we can recommend the ones with high ratings. But the short correlations are different.
# Let's sort by correlation and rank, but some are small and large;
# Let's consider the effect of correlation and rating at the same time.
top_users_ratings["weighted_rating"]= top_users_ratings["corr"] * top_users_ratings["rating"]

# To reach their final values;
top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

# Let's save it to df for a more structured form and solve the index problem.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

# Let's say the main user's scores greater than 3 can be reduced by a higher score. It is an approximation, not a full score.

recommendation_df[recommendation_df["weighted_rating"]> 3.5]

# Let's transfer it to Df
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# To find out which movies these are;

movie = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])

#######################
# Step 6: Functionalization of the Work
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

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
     import pandas as pd
     random_user_df = user_movie_df[user_movie_df.index == random_user]
     movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
     movies_watched_df = user_movie_df[movies_watched]
     user_movie_count = movies_watched_df.T.notnull().sum()
     user_movie_count = user_movie_count.reset_index()
     user_movie_count.columns = ["userId", "movie_count"]
     perc = len(movies_watched) * ratio / 100
     users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

     final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                           random_user_df[movies_watched]])

     corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
     corr_df = pd.DataFrame(corr_df, columns=["corr"])
     corr_df.index.names = ['user_id_1', 'user_id_2']
     corr_df = corr_df.reset_index()

     top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
         ["user_id_2", "corr"]].reset_index(drop=True)


     top_users = top_users.sort_values(by='corr', ascending=False)
     top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
     rating = pd.read_csv("Recommendation Systems/datasets/movie_lens_dataset/rating.csv")
     top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")
     top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('Recommender Systems/datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])



random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, cor_th=0.65, score=3)








