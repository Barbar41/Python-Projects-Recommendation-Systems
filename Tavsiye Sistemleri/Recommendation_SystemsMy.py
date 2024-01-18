##########################
# ASSOCIATION RULE LEARNING(BİRLİKTELİK KURALI ÖĞRENİMİ)
#########################

# 1.Veri ön işleme
# 2.ARL Veri Yapısını Hazırlama(Invoice-Product Matrix)
# 3.Birliktelik Kurallarının Çıkarılması
# 4.Çalışmanın Scriptini Hazırlama
# 5.Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

######################
# 1.Veri Ön İşleme
#####################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width',50)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Tavsiye Sistemleri\datasets\online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()
df.shape

# - Değerlerden kurtulmak için fonk olusturuyoruz.

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe= dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe= dataframe[dataframe["Quantity"] > 0]
    dataframe= dataframe[dataframe["Price"] > 0]
    return dataframe

df=retail_data_prep(df)

# Değişkenler için eşik değer hesaplayıp bu eşik değerin üzerinde kalan değeri bu eşik değer ile değiştirebiliriz.
# Bu durumda baskılama yöntemi kullanmış oluruz.Öyle bir fonk yazacağız ki;Bu değişkenler için eşik değer belirliyor olacak.
# Daha sonra başka bir fonksiyon ile değişkenlerdeki değerleri bu eşik değerlere göre kontrol edecek
# Eğer bu eşik değerlerden aşağı yada yukarı değer varsa bunları ucurucak ve yerine eşik dğerleri yerleştiricek.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01) # %25 olarak
    quartile3 = dataframe[variable].quantile(0.99) # %75 olarak.Ama derin bir dokunus olmaması için (0.01 ve 0.099 degerleri)
    interquantile_range = quartile3 - quartile1  # değişkenin değişimini ifade eden aralıktır.
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Gerekli baskılamayı yapacağız.

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df= retail_data_prep(df)
df.isnull().sum()

############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

df.head()

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

# Herhangı bır ulkeye indirigiyoruz.
# Mesela almanyaya urun satmak ıstıyoruz.Orada hıc musterımız yok oraya nasıl davranacagımızı almanya ıle benzer davranıslar sergılemesını bekledıgımız
# Fransadan ogrendıgımız birliktelik kurallarını almanyadan gelen msuterılere uygulabılme yaklasımı yakalarız.Farklı dikeylerde de düşünebilir.

df_fr = df[df['Country'] == "France"] # Fransa musterılerının bırlıktelık kurallarını turetmıs olacagız.

# Invoice product matrisi oluşturmak ama veriseti yapısı buna cok uygun gibi gozukmuyor.Bunu nasıl yapacagız.
# Önce Invoice lara göre groupby alalım gözlemlemek adına description alalım.Normalde stok code göre gideceğiz,invoice a göre groupby alalım ondan sonra descrptionların groupby alaınmasını saglayalım
# yanı fatura urun ve hangı urunden kacar tane alınmısı hesaplamak ıcın groupbya invoice ve description u koydukdan sonra quantity sum alalım

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

# Groupby işleminde sonra pivot yapıp bunları sütüna geçirmemiz gerekmekte.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]


# Öyle bir işlemle eksik degerlerın yerıne 0 doluların yerıne 1 yazılmasını ıstıyoruz.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# Dolu olan mesela 24 yazan yere 1 yazmamız gerekıyor.

df_fr.groupby(["Invoice", "Description"]).agg({"Quantity":"sum"}).unstack().fillna(0).\
   applymap(lambda x: 1 if x> 0 else 0).iloc[0:5, 0:5]

# apply satır yada sutun bilgisi verilir bir fonksiyonu satır yada sutundaotomatık olarak uygular gezer.
# applymap ise bütün gözlemleri gezer.

# Öncelikle Invoice Product dataframe ını matriisini olusturack fonk yazalım ve bu fonksıyona ozellık verıp.
# İstersek Stok kodlara gore istersek Descrption lara göre getirsin

df_fr.groupby(["Invoice", "StockCode"]).agg({"Quantity":"sum"}).unstack().fillna(0).\
   applymap(lambda x: 1 if x> 0 else 0).iloc[0:5, 0:5]

# Bunun ıcın gereklı fonksiyon;

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


fr_inv_pro_df = create_invoice_product_df(df_fr)

# Argüman girerek işlem kontrolü
fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)

# Bu calısma üzerinde HIZLI VERİMLİ calısmak için;

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_fr, 10120)

############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################

frequent_item_sets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)

frequent_item_sets.sort_values("support", ascending=False).head(20)

rules = association_rules(frequent_item_sets,
                          metric="support",
                          min_threshold=0.01)

rules.head(10)

# Tablonun yorumlanması ise;
# antecedents--önceki ürün
# consequents--ikinci ürün
# antecedents support-- ilk ürün tek basına görülme olasılığı
# consequents support-- ikinci ürün tek basına görülme olasılığı
# support-- Her iki ürünün birlikte görülme olasılığı
# confidence-- X ürün alındıgında Y'nin alınması olasılıgı
# lift --X ürünü satın alındıgında Y'nin satın alınması olasılığı.17 kat artar gibi.
#    lift ise daha az sıklıkta olmasına ragmen bazı ılıskılerı yakalayabılır.Daha degerlıdır yansız bır metrıktır.
# leverage-- Kaldıraç etkisi lift e benzer supportu yüksek değerlere destek verme eğilimindedir.Bundan dolayı yanlıdır.
# conviction-- Y olmadan X ürünün beklenen değeridir,frkansıdır.Yada diğer taraftan X ürünü olmadan Y ürünün beklenen frekansıdır.(çok odakta değil)

# Sıralamaları dilersek lift support yada vs yapabiliriz.Mesela support degerı su deger ustunde lift destegı su degerde olsun gibi olası kombinasyonlar yapılabilir.

rules[(rules["support"]> 0.05) & (rules["confidence"]> 0.1) & (rules["lift"]> 5) ]

# Ürün adları için ise
check_id(df_fr, 21086)

# Olasılıgı daha yuksek daha buyuk olasılıga erısmek ıcın

rules[(rules["support"]> 0.05) & (rules["confidence"]> 0.1) & (rules["lift"]> 5) ].sort_values("confidence", ascending=False)

# confidence daha guvenılır bır metrık bırı satın alındıgında dıgerının satın alınması olasılıgı ıfade edıyor.Eğer Kullanıcı:
#-- antecedents(21080, 21094) ıkı urunu ekledıyse bu durumda consequents (21086) ürününü önereceğiz.Çünkü ikisini birlikte görülme olasılıgı support(0.100257)
# Tamam ama ilk ikili o ürün alındıgında üçüncü ürün alınması olasılıgı confidence(0.97500)
# O zaman bu kısı bunu alır cok yogun ıcerık ve urun var neyı onermem gerek bilmıyorsak bu olasık kullanılır.

############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country] # Bu ulkeye göre veriyi indirge
    dataframe = create_invoice_product_df(dataframe, id) #  create_invoice_product_df dataframe i olustur id ture yani stock codelara göre olsutur
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True) # apriori fonkisyonunu çağır min_support=0.01 e göre olası ürünlerin çiftlerinin frekansları ile oalsılıkları hesapla
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)# bu hesaplamaları kullanarak association_rules tablomu getir rules adında
    return rules

df= df_.copy()

df= retail_data_prep(df)
rules = create_rules(df)

############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

# Örnek:
# Kullanıcı örnek ürün id: 22492

product_id= 22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)
# alternatif olarak yaklasım olarak sorted_rules = rules.sort_values("confidence", ascending=False)

# antecedent bolumunde geecegız ve burada yakalamıs oldugumuz urunelrı aynı ındexteki diger kısımdaki bu degıskendekı degerı gordugumuzde bunları yakalayacagız

recommendation_list=[]
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

check_id(df, 22556)
check_id(df, 22551)
check_id(df, 22326)

#########################
#-Fonksiyonlaştırılması
##########################


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)

