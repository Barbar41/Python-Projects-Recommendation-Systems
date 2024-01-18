###############
# Association RuleBased Recommender System
##############

# İş Problemi:
# ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak 10 film önerisi yapınız.

##############
# Veri Seti Hikayesi
#############
# Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını barındırmaktadır.
# 27.278 filmde 2.000.0263 derecelendirme içermektedir. Bu veri seti ise 17 Ekim 2016 tarihinde oluşturulmuştur.
# 138.493 kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasında verileri içermektedir. Kullanıcılarrastgele seçilmiştir.
# Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.


###########################################
# GÖREV 1--Veriyi Hazırlama
###########################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.expand_frame_repr', False)

# Adım 1: movie, rating veri setlerini okutunuz.
movie = pd.read_csv(r"Tavsiye Sistemleri/datasets/movie_lens_dataset/movie.csv")
movie.head()
movie.shape

rating = pd.read_csv(r"Tavsiye Sistemleri/datasets/movie_lens_dataset/rating.csv")
rating.head()
rating.shape
rating["userId"].nunique()

# Adım 2: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
df=movie.merge(rating, how="left", on="movieId")
df.head(1)
df.head(50)
df.shape

# Adım3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.
comment_counts=pd.DataFrame(df["title"].value_counts())
comment_counts

rare_movies=comment_counts[comment_counts["title"] <= 1000].index

common_movies=df[~df["title"].isin(rare_movies)]
common_movies.head(2)
common_movies.shape


# Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz.
user_movie_df=common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()

# Adım 5: Yapılan tüm işlemleri fonksiyonlaştırınız.
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv(r"Tavsiye Sistemleri/datasets/movie_lens_dataset/movie.csv")
    rating = pd.read_csv(r"Tavsiye Sistemleri/datasets/movie_lens_dataset/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df=create_user_movie_df()

###########################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
###########################################
# Adım 1: Rastgele bir kullanıcı id’si seçiniz.
random_user= 108170

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df= user_movie_df[user_movie_df.index == random_user] #pivot tabloda indexler var o yuzden index
random_user_df.head(2)
random_user_df.isnull().sum()

# Adım 3: Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız
movies_watched=random_user_df.columns[random_user_df.notna().any()].tolist()

###########################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
###########################################
# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturunuz.
user_movie_df.head()
movies_watched_df= user_movie_df[movies_watched]
movies_watched_df.head(2)


# Adım 2: Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
user_movie_count= movies_watched_df.T.notnull().sum()
user_movie_count=user_movie_count.reset_index() #reset index indexi sutuna cevırır
user_movie_count.columns=["userId", "movie_count"]
user_movie_count.index

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden users_same_movies adında bir liste oluşturunuz.
perc= len(movies_watched) * 60/ 100
random_user_df.shape

users_same_movies= user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


###########################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
###########################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
final_df= movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df= final_df.T.corr().unstack().sort_values() # - yonlu negaıtf korelasyon var ılk degerlerde
corr_df=pd.DataFrame(corr_df, columns=["corr"]) # dataframe aktarıyoruz.
corr_df.index.names=["userId_1","userId_2"]#indexlerı ısımlendırıyoruz
corr_df=corr_df.reset_index()


# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
corr_df[corr_df["userId_1"]== random_user]
top_users= corr_df[(corr_df["userId_1"]== random_user)& (corr_df["corr"] >= 0.65)][["userId_2", "corr"]].reset_index(drop=True)

# Adım 4: top_users dataframe’ine rating veri seti ile merge ediniz.
top_users.rename(columns={"userId_2":"userId"},inplace=True)

top_users_ratings=top_users.merge(rating[["userId","movieId", "rating"]],how="inner")
top_users_ratings["userId"].nunique()

###########################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
###########################################
# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users_ratings["weighted_rating"]= top_users_ratings["corr"]* top_users_ratings["rating"]


# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir dataframe oluşturunuz.
recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating":"mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()


# Adım 3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
recommendation_df[recommendation_df["weighted_rating"]> 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"]> 3.5].sort_values("weighted_rating", ascending=False)

# Adım 4: movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][0:5]

####################################
####  Item Based Recommendation #####
#####################################
###########################################
# GÖREV 1--Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız.
###########################################

user = 108170

# Adım 1:  movie, rating veri setlerini okutunuz
movie = pd.read_csv(r"Tavsiye Sistemleri/datasets/movie_lens_dataset/movie.csv")
rating = pd.read_csv(r"Tavsiye Sistemleri/datasets/movie_lens_dataset/rating.csv")

# Adım 2:  Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sini alınız.
movie_id= rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Adım 3:  User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
movie_df=user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]


# Adım 4:  Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Fonksiyonlastırılması
def item_based_recommender(movie_name, user_movie_df):
    movie=user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Adım 5:  Seçili film’in kendisi haricinde ilk 5 filmi öneri olarak veriniz.
movies_from_item_based = item_based_recommender(movie[movie["movieId"]== movie_id]["title"].values[0],user_movie_df)
# 1'den 6'ya kadar.0'da filmin kendisi var.Onu dışarıda bıraktık
movies_from_item_based[1:6].index


