#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', 100)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Tavsiye Sistemleri\datasets\the_movies_dataset\movies_metadata.csv", low_memory=False ) # DtypeWarning kapamak icin
df.head()
df.shape

# Ana odağımız Benzerlik hesapları için overview değişkeni
df["overview"].head()

# Az önce import edilen metod çağırılıor.
# Cokfazla bosluk durumun onunce gecmek adına yapılıyor.Ölçüm değeri taşımayan(in,on,and..)
tfidf= TfidfVectorizer(stop_words="english")

# "overview" değişkeni içindeki eksikliklerden kurtulalım.

df[df["overview"].isnull()]
# Boslukla dolduralım degersızlerı
df["overview"]=df["overview"].fillna("")

# Overview dönüşümü

tfidf_matrix= tfidf.fit_transform(df["overview"])

tfidf_matrix.shape
# satırlarda açıklamalar,sütunlarda eşssiz kelimeler var

# satırlardakiler film oldugu dogrulaması
df["title"].shape

# Öyle ise ikisinin kesişiminde ne vardır?tfidf skorları vardır.

tfidf.get_feature_names()

# Filtreleme
df = df[~df["title"].duplicated(keep="last")]
df = df[~df["title"].isna()]
df = df[~df["overview"].isna()]

# Matrisin hafızadaki boyutunu yarıya indirme
tfidf_matrix = tfidf_matrix.astype(np.float32)

# Kesişimleri ile ilgili erişmek istiyoruz.

tfidf_matrix.toarray()

#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

cosine_sim=cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
cosine_sim[1]

#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

# Bir pandas serisi oluşturup bu serinin içine filmlerin title yerlestırecegız.
indices= pd.Series(df.index, index=df["title"])

indices.index.value_counts()

# Önceki filmleri ucuruyoruz.Coklama isimlerin en sonundakini alacagız.En son cekılen fılmı ıstıyoruz.Çoklama Kayıtlardan kurtuluyoruz.
indices=indices[~indices.index.duplicated(keep="last")]

indices["Cinderella"]

indices["Sherlock Holmes"]

# Bir film indexini tutalım
movie_index=indices["Sherlock Holmes"]

cosine_sim[movie_index]

# Sherlock Holmes filmi ile elimdeki filmlerin benzerlik skorlarını çıkardık.Ve Database a yerlestırıdk.
similarity_scores=pd.DataFrame(cosine_sim[movie_index],
                               columns=["Score"])

# En yüksek skora sahip 10 filmi getirmek istersek
movie_indices= similarity_scores.sort_values("Score",
                                             ascending=False)[1:11].index

# Filmlerin isimlerine gitmek istiyorum.
df["title"].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesaplama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
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
