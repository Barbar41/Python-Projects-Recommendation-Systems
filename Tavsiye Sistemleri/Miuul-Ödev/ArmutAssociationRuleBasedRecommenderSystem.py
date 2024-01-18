###############
# Association RuleBased Recommender System
##############

# İş Problemi:
# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak Association
# Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.
##############
# Veri Seti Hikayesi
#############
# Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.Alınan her hizmetin tarih ve saat bilgisini içermektedir.
# 4 Değişken 162.523 Gözlem 5 MB
# UserId--Müşteri numarası
# ServiceId--Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId--Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate--Hizmetin satın alındığı tarih


###########################################
# GÖREV 1--Veriyi Hazırlama
###########################################
import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.expand_frame_repr', False)


# Adım 1:armut_data.csv dosyasını okutunuz
df_ = pd.read_csv(r"Tavsiye Sistemleri/Miuul-Ödev/armut_data.csv")
df = df_.copy()
df.describe().T
df.head()


# Adım 2:ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.
# Elde edilmesi gereken çıktı:

# UserId ServiceId CategoryId    CreateDate    Hizmet
#  25446     4          5     6.08.2017 16:11    4_5
#  22948    48          5     6.08.2017 16:12   48_5
#  10618     0          8     6.08.2017 16:13    0_5
#   7256     9          4     6.08.2017 16:14    9_4
#  25446    48          5     6.08.2017 16:16   48_5

df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()


# Adım 3:Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır. Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 25446 id'li müşteri 2017'in 8.ayında aldığı 4_5, 48_5, 6_7, 47_7 hizmetler bir sepeti; 2017'in 9.ayında aldığı 17_5, 14_7 hizmetler başka bir sepeti ifade etmektedir.
# Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız. Elde edilmesi gereken çıktı:

# UserId    ServiceId   CategoryId     CreateDate       Hizmet   New_Date    SepetID
# 25446        4            5        6.08.2017 16:11     4_5     2017-08    25446_2017-08
# 22948       48            5        6.08.2017 16:12    48_5     2017-08    22948_2017-08
# 10618        0            8        6.08.2017 16:13     0_5     2017-08    10618_2017-08
# 7256         9            4        6.08.2017 16:14     9_4     2017-08     7256_2017-08
# 25446       48            5        6.08.2017 16:16    48_5     2017-08    25446_2017-08

df.info()
df["CreateDate"]= pd.to_datetime(df["CreateDate"])
df["NEW_DATE"]= df["CreateDate"].dt.strftime("%Y-%m")
df.head(5)

df["SepetID"]= [str(row[0])+ "_"+ str(row[5]) for row in df.values]
df.head()


###########################################
# Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
###########################################

# Adım 1:Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz

# Hizmet        0_8   10_9   11_11   12_7   13_11   14_7..
# SepetID
# 0_2017-08      0     0       0      0       0       0..
# 0_2017-09      0     0       0      0       0       0..
# 0_2018-01      0     0       0      0       0       0..
# 0_2018-04      0     0       0      0       0       1..
# 10000_2017-08  0     0       0      0       0       0

invoice_product_df=df.groupby(["SepetID", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x:1 if x> 0 else 0)

# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets=apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules=association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

# Adım 3: arl_recommender fonksiyonunu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(rules, "2_0", 1)