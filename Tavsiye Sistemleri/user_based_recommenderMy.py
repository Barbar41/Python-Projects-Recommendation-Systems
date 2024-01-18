############################################
# User-Based Collaborative Filtering
#############################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması


# Adım 1: Veri Setinin Hazırlanması
#############################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


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

#############################################
# Adım 2: Öneri Yapılacak Ana Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

import pandas as pd
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# Öncelikle bu Ana kullanıcı hangi filmleri izlemiş bunu bulmak.

random_user
user_movie_df

# Ana Kullanıcı izlediği ve izlemediği filmler
random_user_df= user_movie_df[user_movie_df.index == random_user]

# Ana Kullanıcının izlediği film isimleri
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Dogrulama yapalım satır ve sutunalrda secım ıslemı yapalım
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Sense and Sensibility (1995)"]

# Ana Kullanıcı kaç film izlemiş
len(movies_watched)

#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Öncelikle user movie df den izlenen filmler listesini sorarak veri setini indirgeyelim.
# Böylece elimzde artık izlenen filmlere ilişkin bilgi var

movies_watched_df = user_movie_df[movies_watched]
# Veriyi özelleştirip genelden mümkün oldugu kadar custom a daha yakın olası cozumlere gıdıoruz.

# Aynı filmleri izleyen kullanıcıların Idsini erişmek başlığındayız.
# Peki en az bir tane dahi film izleyen tum kullanıcılar bu veriye girmelimi?
# Az adette bırlıkte fılm ızleyenler fayda saglamaz bu yuzden bır sınır belırlenmeleı ortak 10 fılm gıbı
# Bir kullanıcı filmi izlemişmi izlemişmi?
user_movie_count= movies_watched_df.T.notnull().sum()

# Bu indexte yer alan UserId yi değişkene cevırıyoruz.
user_movie_count=user_movie_count.reset_index()

# İsimlendirmesini yapalım
user_movie_count.columns=["userId", "movie_count"]

# 20 den buyuk olanları getırelım.
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

# Ana kullanıcnın ızledıgı tum fılmelerı ızleyen kac kullanıcı vardır
user_movie_count[user_movie_count["movie_count"] == 33].count()

# Bize bu kullanıcıların Id leri lazım
users_same_movies= user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

# Bu bölümü daha programatik yapmak için
# perc= len(movies_watched)*60/100
# users_same_movies=user_movie_count[user_movie_count["movie_count"]> perc]["userId"]

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Ana Kullanıcı ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız

# Daha önceki veri setlerini bir araya getirelim.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                    random_user_df[movies_watched]])

# Sutunlara kullanıcıları almalıyız.
corr_df= final_df.T.corr().unstack().sort_values().drop_duplicates()

# Klasik bir df cevırelım daha sonra bu df yı ısımlendırmelere dokuncak sekılde ısımlendırelım
corr_df=pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names= ["user_id_1", "user_id_2"]

corr_df= corr_df.reset_index()

# Ana kullanıcı ile yuksek korelasyonluluar lazım.
# UserId_1 için Ana Kullanıcıyı UserId_2 için diğer kullanıcıları alırsak ve
# Korelasyonu bellı oran ustu kullanıcıalrı getır dersek
# Ana kullanıcı ile benzer davranısları gosteren kullanıcıları bulmus oluruz.
# %65 üstü pozitif yönlü bir korelasyon olmalıkı ana kullanıcı ilşikisini yakalayalım.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

# Sıralamasını yapalım.
top_users= top_users.sort_values(by="corr", ascending=False)

# İsimlendirmesini yapalım
top_users.rename(columns={"user_id_2":"userId"},inplace=True)

# Bu kullanıcıların hangi filme kaç puan verdiği bilgisi yok

# Rating dosyasını top users ıle bırlestırelım.
rating= pd.read_csv("Tavsiye Sistemleri/datasets/movie_lens_dataset/rating.csv")
top_users_ratings= top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

# Ana kullancıyı calısmadan cıkaralım.
top_users_ratings= top_users_ratings[top_users_ratings["userId"] != random_user]

#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################

# Sadece ratingler ve ana kullanıcı ıle benzer kullanıcılar varsa yuksek reytınge sahıp olanları onerebılırız.Fakat kısı korelasyonları farklı.
# Korelasyona gore sıralayıp ratinge gore sıralasak ama bazıları kucuk ve buyuk;
# Korelasyonun ve ratingin etksıını aynı anda göz önünde bulundurabılelım.
top_users_ratings["weighted_rating"]= top_users_ratings["corr"] * top_users_ratings["rating"]

# Nihai değerlerine ulaşmak için;
top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

# Daha yapısal bir form ici df ye kaydedelım index problemınıde cozelım
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

# Ana kullanıcnın 3 ten buyuk olan skorları getır dıyelım daha yuksek rakla azaltılabılır.Tam skor degıl de yaklasımdır.

recommendation_df[recommendation_df["weighted_rating"]> 3.5]

# Df ye aktaralım
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Bu filmlerin hangi filmler oldugunu ogrenmek için;

movie = pd.read_csv('Tavsiye Sistemleri/datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])

#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

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
    rating = pd.read_csv("Tavsiye Sistemleri/datasets/movie_lens_dataset/rating.csv")
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('Tavsiye Sistemleri/datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])



random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, cor_th=0.65, score=3)








