#################
# Model-Based Collaborative Filtering: Matrix Factorization
#################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Step 1: Preparing the Data Set
# Step 2: Modeling
# Step 3: Model Tuning
# Step 4: Final Model and Prediction

#################
# Step 1: Preparing the Data Set
#################

movie = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")

# In terms of traceability, 4 movies and 4 movie IDs.
movie_ids=[130219, 356, 4422,541]
movies = ["The Dark Knight (2011)",
           "Cries and Whispers (Viskningar och rop) (1972)",
           "Forrest Gump (1994)",
           "Blade Runner (1982)"]

# We create a data set according to the ids here.
sample_df=df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

# We create our df through these.
user_movie_df=sample_df.pivot_table(index=["userId"],
                                     columns=["title"],
                                     values=["rating"])
user_movie_df.shape

# We are making an information entry.
reader= Reader(rating_scale=(1,5))

# We brought our own data to the data format requested by the Suprise library.
data = Dataset.load_from_df(sample_df[['userId',
                                        'movieId',
                                        'rating']], reader)
#################
# Step 2: Modeling
#################
# We Divide the Data.
trainset, testset = train_test_split(data, test_size=.25)

# Model object created
svd_model=SVD()

# We fit and build the model. There is a model created with the matrix factorization method.
svd_model.fit(trainset)

# Let's use this on Testset
predictions = svd_model.test(testset)

# We can learn with Accuracy import. We will find the error difference.
accuracy.rmse(predictions)

# Let's make a prediction for one user.
svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)

# For first user selection
sample_df[sample_df["userId"]==1]

# In this way, when we enter the ID and movie ID of any user we want;
# We will obtain how many points users can give when they watch this movie.


#################
# Step 3: Model Tuning
#################
# OPTIMIZING THE MODEL IS TRYING TO INCREASE THE PREDICTION PERFORMANCE OF THE MODEL.
# How to optimize the model's hyperparameters (specified by the user).

# By combining the number of iterations and the number of epochs, we can test the model with parameter set inputs.

param_grid = {'n_epochs': [5, 10, 20],
               'lr_all': [0.002, 0.005, 0.007]}

# We call the method.
gs = GridSearchCV(SVD will use #matrix factorization method
                   param_grid, # will use parameter metrics (customizable)
                   measures=['rmse', 'mae'],# take the average of the squares of the differences between the actual values and the predicted values, or take the square root of this average
                   cv=3,# cross validation (divide the data set into 3, build a model with 1 piece and test 2 pieces and combine them, then take the average
                   n_jobs=-1,# use full performance cpu
                   joblib_verbose=True) # report at that time

gs.fit(data)

gs.best_score["rmse"]
gs.best_params["rmse"]

#################
# Step 4: Final Model and Prediction
#################


# call svd model object
dir(svd_model)

# let's call n_epochs
svd_model.n_epochs

# Create the model with new values
svd_model = SVD(**gs.best_params['rmse'])

# Let's show all the data. We turned it into a full train set.
# We saw our error rate and found the best values of the hyperparameter. We created the model object according to these values.

data = data.build_full_trainset()
svd_model.fit(data)

# Let's ask for the model of the prediction.
svd_model.predict(uid=1.0, iid=541, verbose=True)
# Blade Runner movie was given a rating of 4, we found it to be 4.20, not very good but not bad, 20% is open to discussion.