#############################
# ASSOCIATION RULE LEARNING
#############################

#1.Data preprocessing
# 2.Preparing the ARL Data Structure (Invoice-Product Matrix)
# 3. Issuance of Association Rules
# 4.Preparing the Script of the Work
# 5. Making Product Recommendations to Users in the Basket Stage

##########################
#1.Data Preprocessing
#######################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width',50)
# ensures the output is on a single line.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Recommendation Systems\datasets\online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()
df.shape

# - We create a function to get rid of the values.

def retail_data_prep(dataframe):
     dataframe.dropna(inplace=True)
     dataframe= dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
     dataframe= dataframe[dataframe["Quantity"] > 0]
     dataframe= dataframe[dataframe["Price"] > 0]
     return dataframe

df=retail_data_prep(df)

# We can calculate the threshold value for variables and replace the value above this threshold value with this threshold value.
# In this case, we will use the suppression method. We will write such a function that will determine the threshold value for these variables.
# Then, another function will check the values in the variables according to these threshold values.
# If there are values below or above these threshold values, it will remove them and replace them with threshold values.

def outlier_thresholds(dataframe, variable):
     quartile1 = dataframe[variable].quantile(0.01) # as 25%
     quartile3 = dataframe[variable].quantile(0.99) # As 75%. But to avoid a deep touch (0.01 and 0.099 values)
     interquantile_range = quartile3 - quartile1 # is the range that expresses the change of the variable.
     up_limit = quartile3 + 1.5 * interquantile_range
     low_limit = quartile1 - 1.5 * interquantile_range
     return low_limit, up_limit

# We will do the necessary suppression.

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

#######################
# 2. Preparing the ARL Data Structure (Invoice-Product Matrix)
#######################

df.head()

# Description NINE DRAWER OFFICE TIDY SET 2 TEA TOWELS I LOVE LONDON SPACEBOY BABY GIFT SET
#Invoice
# 536370 0 1 0
# 536852 1 0 1
# 536974 0 0 0
# 537065 1 0 0
# 537463 0 0 1

# We reduce it to any country.
# For example, we want to sell products to Germany. We do not have any customers there.
# We take the approach of applying the rules of association we learned from France to customers from Germany. We can also think in different verticals.

df_fr = df[df['Country'] == "France"] # We will have derived the association rules of France clients.

# Creating an invoice product matrix, but the dataset structure does not seem very suitable for this. How do we do this?
# First, let's get a groupby according to the invoices and get a description to observe. Normally, we will go according to the stock code, let's get a groupby according to the invoice, then let's get the descriptions to be groupbyed.
# In order to calculate the invoice product and how many of the product were purchased, let's get the quantity sum after putting the invoice and description into the groupbya.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

# After the groupby process, we need to pivot and pass them to the column.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]


# With such an operation, we want to write 0 instead of missing values and 1 instead of filled ones.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# We need to write 1 in the filled field, for example 24.

df_fr.groupby(["Invoice", "Description"]).agg({"Quantity":"sum"}).unstack().fillna(0).\
    applymap(lambda x: 1 if x> 0 else 0).iloc[0:5, 0:5]

# apply The row or column information is given, it automatically applies a function in the row or column.
# applymap browses all observations.

# First of all, let's write a function to create the matrix of the Invoice Product dataframe and give properties to this function.
# If we want, it can be fetched according to stock codes or according to Descrptions.

df_fr.groupby(["Invoice", "StockCode"]).agg({"Quantity":"sum"}).unstack().fillna(0).\
    applymap(lambda x: 1 if x> 0 else 0).iloc[0:5, 0:5]

# The required function for this is;

def create_invoice_product_df(dataframe, id=False):
     if id:
         return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
             applymap(lambda x: 1 if x > 0 else 0)
     else:
         return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
             applymap(lambda x: 1 if x > 0 else 0)


fr_inv_pro_df = create_invoice_product_df(df_fr)

# Process control by entering arguments
fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)

# To work FAST AND EFFICIENT on this study;

def check_id(dataframe, stock_code):
     product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
     print(product_name)


check_id(df_fr, 10120)

#######################
# 3. Issuance of Association Rules
#######################

frequent_item_sets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)

frequent_item_sets.sort_values("support", ascending=False).head(20)

rules = association_rules(frequent_item_sets,
                           metric="support",
                           min_threshold=0.01)

rules.head(10)

# Interpretation of the table is;
# antecedents--previous item
#consequences--second product
# antecedents support-- possibility of seeing the first product alone
# consequences support-- probability of second product appearing alone
# support-- Possibility of seeing both products together
#confidence--Probability of purchasing Y when product X is purchased
# lift --When product X is purchased, the probability of purchasing Y increases by 17 times.
# lift, although less frequent, can capture some relationships. It is more valuable and an unbiased metric.
# leverage-- Leverage effect is similar to lift, its support tends to support high values. Therefore, it is biased.
#conviction-- It is the expected value, the frequency, of product X without Y. Or, on the other hand, it is the expected frequency of product Y without product

# If we wish, we can make the rankings like lift support or etc. Possible combinations can be made, for example, the support value should be above the water value and the lift support should be at the same value.

rules[(rules["support"]> 0.05) & (rules["confidence"]> 0.1) & (rules["lift"]> 5) ]

# For product names
check_id(df_fr, 21086)

# To reach a higher probability

rules[(rules["support"]> 0.05) & (rules["confidence"]> 0.1) & (rules["lift"]> 5) ].sort_values("confidence", ascending=False)

#confidence is a more reliable metric that expresses the probability of purchasing another when one is purchased. If the User:
#-- If antecedents(21080, 21094) added two items, then we will recommend consequents (21086). Because the probability of seeing both together is support(0.100257)
# Okay, but when the first two items are purchased, the probability of purchasing the third item is confidence(0.97500)
# Then this person will buy this. There is a lot of content and product. If we do not know what to offer, this possibility is used.

#######################
# 4. Preparing the Script of the Work
#######################

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
     dataframe = dataframe[dataframe['Country'] == country] # Reduce data by this country
     dataframe = create_invoice_product_df(dataframe, id) # create_invoice_product_df create dataframe according to id type, that is, stock codes
     frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True) # call apriori function calculate probabilities and frequencies of pairs of possible items based on min_support=0.01
     rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)# use these calculations to get my association_rules table called rules
     return rules

df= df_.copy()

df= retail_data_prep(df)
rules = create_rules(df)

#######################
# 5. Making Product Recommendations to Users in the Cart Stage
#######################

# Example:
# User sample product id: 22492

product_id=22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)
# alternatively approach sorted_rules = rules.sort_values("confidence", ascending=False)

# We will go through the antecedent section and when we see the value of this variable in the other section of the same index, we will capture the products we have captured here.

recommendation_list=[]
for i, product in enumerate(sorted_rules["antecedents"]):
     for j in list(product):
         if j == product_id:
             recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

check_id(df, 22556)
check_id(df, 22551)
check_id(df, 22326)

#############################
#-Functionalization
#############################


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

