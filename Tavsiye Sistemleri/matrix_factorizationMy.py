#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('Tavsiye Sistemleri/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Tavsiye Sistemleri/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")

# Takip edilebilirlik acısından 4 fılm 4 fılm ıdsi.
movie_ids=[130219, 356, 4422,541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

# Buradaki idlere göre bir veri seti olusuturuyoruz.
sample_df=df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

# Bunlar üzerinden df mizi olusturuyoruz.
user_movie_df=sample_df.pivot_table(index=["userId"],
                                    columns=["title"],
                                    values=["rating"])
user_movie_df.shape

# Bilgilendirme girişi yapıyoruz.
reader= Reader(rating_scale=(1,5))

# Suprise kutuphanesinin istediği veri formatına kendi verimizi getirdik.
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)
##############################
# Adım 2: Modelleme
##############################
# Veriyi Bölüyoruz.
trainset, testset = train_test_split(data, test_size=.25)

# Model nesnesi oluşturuldu
svd_model=SVD()

# Fit edip modeli kuruyoruz.Matrix factorization yöntemiyle oluşturulmus model var.
svd_model.fit(trainset)

# Testset üzerinde bunu kullanalım
predictions = svd_model.test(testset)

# Accuracy import ile ogrenebiliriz.Hata farkımızı bulacagız.
accuracy.rmse(predictions)

# Bir tane kullanıcı için bir tahminde bulunalım.
svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)

# Birinci kullanıcı seçimi için
sample_df[sample_df["userId"]==1]

# Bu şeklilde istediğimiz herhangi bir kullanıcının Idsini ve film idsini girdiğimizde;
# Kullanıcıların bu filmi izlediklerinde kaç puaan verebileceklerini elde etmiş oalcagız.


##############################
# Adım 3: Model Tuning
##############################
# MODEL OPTİMİZE ETMEK MODELİN TAHMİN PERFORMANSINI ARTTIRMAYA ÇALIŞMAKTIR.
# Modelin hiperparametrelerinin(kullanıcı tarafından belirlenen)nasıl optimize edeceğimiz.

# İterasyon sayısı ve epochs sayısılarını kombine ederek parametre seti girdileri ile modeli sınayabiliriz.

param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

# Method çağırırıyoruz.
gs = GridSearchCV(SVD, #matrix factorization method kullanacak
                  param_grid, # parametre olculerı kullanacak(kişiselleştirilebilir)
                  measures=['rmse', 'mae'],# gercek degerler ıle tahmın edılen degerlerın farklarının karelerının ortalamasını al yada bu ortalamanın karekökünü al
                  cv=3,# çapraz dogrulama(veri setini 3 e böl 1 parçasıyla test 2 parcasıyla model kur.ve kombine et(kombinle) sonra ortalamasını al
                  n_jobs=-1,# full performans cpu kullan
                  joblib_verbose=True) # o sırada raporlama yap

gs.fit(data)

gs.best_score["rmse"]
gs.best_params["rmse"]

##############################
# Adım 4: Final Model ve Tahmin
##############################


# svd model nesnesı cagıralım
dir(svd_model)

# n_epochs cagıralım
svd_model.n_epochs

# Modeli yeni değerleri ile oluşturmak
svd_model = SVD(**gs.best_params['rmse'])

# Bütün veriyi gösterelim.Full train sete cevırdık.
# Hata oranımızı görduk hiperparametrenın en ıyı degerlenı bulduk.Bu degerlere göre model nesnesını olusturduk.

data = data.build_full_trainset()
svd_model.fit(data)

# Tahmın modeli isteyelim.
svd_model.predict(uid=1.0, iid=541, verbose=True)
# Blade Runner filmi için 4 deger verılmıs bız 4.20 bulduk cok ıyı degıl ama kotude degıl %20 tartısmaya acık

