# Association Rule Based Recommender System

# Business Problem:
# Below is the basket information of 3 different users.
# Make the product suggestion that best suits this basket information using the association rule.
# Product recommendations can be 1 or more than 1.
# Derive decision rules based on 2010-2011 Germany customers.
# ID of the product in User 1's cart: 21987
# ID of the product in User 2's cart: 23235
# ID of the product in User 3's cart: 22747

##############
# Dataset Story
##############

# Online Retail II data set includes online sales transactions of a UK-based retail company between 01/12/2009-09/12/2011.
# The company's product catalog includes gift items and it is known that most of its customers are wholesalers.

#8 Variable 541,909 Observations 45.6MB
# InvoiceNo Invoice Number (If this code starts with C, it means that the transaction has been cancelled)
# StockCode Product code (unique for each product)
# Description Product name
# Quantity Number of products (How many of the products on the invoices were sold)
# InvoiceDate Invoice date
# UnitPrice Invoice price ( Sterling )
# CustomerID Unique customer number
# Country Country name

#######################
# TASK 1--Preparing the Data
#######################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width',50)
# ensures the output is on a single line.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Read the 2010-2011 sheet from the Online Retail II data set.
df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Recommendation Systems\datasets\online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()
df.shape


# Step 2: Drop the observation units whose StockCode is POST. (POST is the price added to each invoice, it does not refer to the product.)
df[df["StockCode"] == "POST"]
df = df[~df["StockCode"].str.contains("POST", na=False)]



# Step 3: Drop the observation units containing empty values.
df.dropna(inplace=True)

# Step 4: Remove the values containing C in the Invoice from the data set. (C indicates cancellation of the invoice.)
df = df[~df["Invoice"].str.contains("C", na=False)]

# Step 5: Filter the observation units whose Price value is less than zero.
df= df[df["Price"] > 0]


# Step 6: Examine the outliers of the Price and Quantity variables and suppress them if necessary.
def outlier_thresholds(dataframe, variable):
     quartile1 = dataframe[variable].quantile(0.01)
     quartile3 = dataframe[variable].quantile(0.99)
     interquantile_range = quartile3 - quartile1
     up_limit = quartile3 + 1.5 * interquantile_range
     low_limit = quartile1 - 1.5 * interquantile_range
     return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
     low_limit, up_limit = outlier_thresholds(dataframe, variable)
     # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit( - since there are no values)
     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


#######################
# TASK 2--Creating Association Rules through German Customers
#######################

df_gr=df[df["Country"] == "Germany"]

# Step 1: Define the create_invoice_product_df function that will create the invoice product pivot table as follows.
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

# Step 2: Define the create_rules function that will create the rules and find the rules for German customers.
frequent_item_sets = apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_item_sets.sort_values("support", ascending=False).head(20)
rules = association_rules(frequent_item_sets,
                           metric="support",
                           min_threshold=0.01)
rules.head(10)

def create_rules(dataframe, id=True, country="Germany"):
     dataframe = dataframe[dataframe['Country'] == country] # Reduce data by this country
     dataframe = create_invoice_product_df(dataframe, id) # create_invoice_product_df create dataframe according to id ture, i.e. stock codes
     frequent_item_sets = apriori(dataframe, min_support=0.01, use_colnames=True) # call apriori function calculate probabilities and frequencies of pairs of possible items based on min_support=0.01
     rules = association_rules(frequent_item_sets, metric="support", min_threshold=0.01)# use these calculations to get my association_rules table called rules
     return rules
rules = create_rules(df_gr)

#######################
# TASK 3--Making Product Recommendations to Users Given Product Ids in the Cart
#######################
# Step 1: Find the names of the given products using the check_id function.
def check_id(dataframe, stock_code):
     product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
     print(product_name)
check_id(df_gr, 10125)


# Step 2: Recommend products for 3 users using the arl_recommender function.

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



# Step 3: Look at the names of the products to be recommended.

check_id(df_gr, 22556)
check_id(df_gr, 22551)
check_id(df_gr, 22326)

