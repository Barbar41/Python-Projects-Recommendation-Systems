# Association Rule Based Recommender System

# İş Problemi:
# Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir.
# Bu sepet bilgilerine en uygun ürün önerisini birliktelik kuralı kullanarak yapınız.
# Ürün önerileri 1 tane ya da 1'den fazla olabilir.
# Karar kurallarını 2010-2011 Germany müşterileri üzerinden türetiniz.
# Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
# Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
# Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747

##############
# Veri Seti Hikayesi
#############

# Online Retail II veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009-09/12/2011 tarihlerinde ki online satış işlemlerini içeriyor.
# Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi mevcuttur.

# 8 Değişken 541.909 Gözlem 45.6MB
# InvoiceNo Fatura Numarası ( Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder )
# StockCode Ürün kodu ( Her bir ürün için eşsiz )
# Description Ürün ismi
# Quantity Ürün adedi ( Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate Fatura tarihi
# UnitPrice Fatura fiyatı ( Sterlin )
# CustomerID Eşsiz müşteri numarası
# Country Ülke ismi

###########################################
# GÖREV 1--Veriyi Hazırlama
###########################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width',50)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.
df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Tavsiye Sistemleri\datasets\online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()
df.shape


# Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)
df[df["StockCode"] == "POST"]
df = df[~df["StockCode"].str.contains("POST", na=False)]



# Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
df.dropna(inplace=True)

# Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
df = df[~df["Invoice"].str.contains("C", na=False)]

# Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
df= df[df["Price"] > 0]


# Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit( - degerler olmadıgı ıcın )
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


###########################################
# GÖREV 2--Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
###########################################

df_gr=df[df["Country"] == "Germany"]

# Adım 1: Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.
df.head()
df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)
df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]
df_gr.groupby(["Invoice", "StockCode"]).agg({"Quantity":"sum"}).unstack().fillna(0).\
   applymap(lambda x: 1 if x> 0 else 0).iloc[0:5, 0:5]
df_gr.groupby(["Invoice", "Description"]).agg({"Quantity":"sum"}).unstack().fillna(0).\
   applymap(lambda x: 1 if x> 0 else 0).iloc[0:5, 0:5]
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

gr_inv_pro_df = create_invoice_product_df(df_gr)

gr_inv_pro_df = create_invoice_product_df(df_gr, id=True)

# Description  **NINE DRAWER OFFICE TIDY **SET 2 TEA TOWELS I LOVE LONDON **SPACEBOY BABY GIFT SET…
# Invoice
# 536370                0                          1                               0
# 536852                1                          0                               1
# 536974                0                          0                               0
# 537065                1                          0                               0
# 537463                0                          0                               1

# Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz.
frequent_item_sets = apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_item_sets.sort_values("support", ascending=False).head(20)
rules = association_rules(frequent_item_sets,
                          metric="support",
                          min_threshold=0.01)
rules.head(10)

def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country] # Bu ulkeye göre veriyi indirge
    dataframe = create_invoice_product_df(dataframe, id) #  create_invoice_product_df dataframe i olustur id ture yani stock codelara göre olsutur
    frequent_item_sets = apriori(dataframe, min_support=0.01, use_colnames=True) # apriori fonkisyonunu çağır min_support=0.01 e göre olası ürünlerin çiftlerinin frekansları ile oalsılıkları hesapla
    rules = association_rules(frequent_item_sets, metric="support", min_threshold=0.01)# bu hesaplamaları kullanarak association_rules tablomu getir rules adında
    return rules
rules = create_rules(df_gr)

###########################################
# GÖREV 3--Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
###########################################
# Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)
check_id(df_gr, 10125)


# Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, 22556, 1)
arl_recommender(rules, 22551, 2)
arl_recommender(rules, 22326, 3)



# Adım 3: Önerilecek ürünlerin isimlerine bakınız.

check_id(df_gr, 22556)
check_id(df_gr, 22551)
check_id(df_gr, 22326)
