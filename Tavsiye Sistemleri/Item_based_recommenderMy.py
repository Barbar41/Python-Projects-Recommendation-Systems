###########################################
# Item-Based Collaborative Filtering
###########################################

# Veri seti: https://grouplens.org/datasets/movielens/

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 20)
movie = pd.read_csv('Tavsiye Sistemleri/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Tavsiye Sistemleri/datasets/movie_lens_dataset/rating.csv')

# MoveId ye göre iki csv dosyasını birleştiriyoruz.
df = movie.merge(rating, how="left", on="movieId")
df.head()

######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

# Genel problem matristeki seyreklik durumudur.

df.head()
df.shape

# Eşssiz kaç film var
df["title"].nunique()

# Hangi film kaç rate aldığına erişelim
df["title"].value_counts().head()

# Value counts df ye cevırelım
comment_counts= pd.DataFrame(df["title"].value_counts())

# Belirli bir sayıdan az olan filmleri önce bir gidelim
comment_counts[comment_counts["title"] <= 1000]

# Buradaki isimlere ulaşmak için ise
rare_movies=comment_counts[comment_counts["title"] <= 1000].index

# Buradaki az rate alan filmelrden kurtulmak için bu isimlerin olmadıgı title gelmesi lazım

common_movies= df[~df["title"].isin(rare_movies)]

common_movies.shape

common_movies["title"].nunique()
df["title"].nunique()

# Indırgeme işlemi tamamlandı ve 1000 den fazla rate almıs fılmler elımızde.
# Satırlarda kullanıcılar sutunlarda ise title olsun.

user_movie_df= common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.columns

######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################
pd.set_option('display.max_columns', 500)

movie_name= "Matrix, The (1999)"
movie_name= user_movie_df[movie_name]

# Korelasyon formulu kullanıyoruz
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# Başka bir film için deneyelim
movie_name= "Ocean's Twelve (2004)"
movie_name= user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# Rastgele filmler seçerek erişmek
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name= user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# Rastgele fılm secenekleri Fonksiyonlaşması;
def check_film(keyword, user_movie_df):
    return[col for col in user_movie_df.columns if keyword in col]

check_film("Sherlock", user_movie_df)
check_film("Insomnia", user_movie_df)

######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Tavsiye Sistemleri/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Tavsiye Sistemleri/datasets/movie_lens_dataset/rating.csv')
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

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)