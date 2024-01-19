#################
# Association RuleBased Recommender System
##############

# Business Problem:
# Armut, Turkey's largest online service platform, brings together service providers and those who want to receive service.
# You can easily access services such as cleaning, renovation and transportation with a few taps on your computer or smartphone.
# provides access.
# Using the data set containing service users and the services and categories these users receive, Association
# It is desired to create a product recommendation system with Rule Learning.
##############
# Dataset Story
##############
# The data set consists of the services received by customers and the categories of these services. It contains the date and time information of each service received.
#4 Variable 162,523 Observations 5 MB
# UserId--Customer number
# ServiceId--Anonymized services belonging to each category. (Example: Sofa washing service under the cleaning category)
# A ServiceId can be found under different categories and represents different services under different categories.
# (Example: The service with CategoryId 7 and ServiceId 4 is radiator cleaning, while the service with CategoryId 2 and ServiceId 4 is furniture assembly)
# CategoryId--Anonymized categories. (Example: Cleaning, transportation, renovation category)
# CreateDate--The date the service was purchased


#######################
# TASK 1--Preparing the Data
#######################
import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.expand_frame_repr', False)


# Step 1: Read the armut_data.csv file
df_ = pd.read_csv(r"Recommendation Systems/Miuul-Homework/armut_data.csv")
df = df_.copy()
df.describe().T
df.head()


# Step 2: ServiceID represents a different service for each CategoryID.
# Create a new variable to represent these services by combining ServiceID and CategoryID with "_".
# Output to be obtained:

# UserId ServiceId CategoryId CreateDate Service
# 25446 4 5 6.08.2017 16:11 4_5
# 22948 48 5 6.08.2017 16:12 48_5
# 10618 0 8 6.08.2017 16:13 0_5
# 7256 9 4 6.08.2017 16:14 9_4
# 25446 48 5 6.08.2017 16:16 48_5

df["Service"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()


# Step 3: The data set consists of the date and time the services were received, there is no basket definition (invoice, etc.). In order to apply Association Rule Learning, a basket (invoice, etc.) definition must be created.
# Here, the basket definition is the services that each customer receives monthly.
# For example; The customer with ID 25446 received a basket of 4_5, 48_5, 6_7, 47_7 services in the 8th month of 2017; The 17_5 and 14_7 services received in the 9th month of 2017 represent another basket.
# Carts must be identified with a unique ID. To do this, first create a new date variable that contains only the year and month.
# Combine UserID and the date variable you just created with "_" and assign it to a new variable called ID. The output you should get is:

# UserId ServiceId CategoryId CreateDate Service New_Date CartID
# 25446 4 5 6.08.2017 16:11 4_5 2017-08 25446_2017-08
# 22948 48 5 6.08.2017 16:12 48_5 2017-08 22948_2017-08
# 10618 0 8 6.08.2017 16:13 0_5 2017-08 10618_2017-08
# 7256 9 4 6.08.2017 16:14 9_4 2017-08 7256_2017-08
# 25446 48 5 6.08.2017 16:16 48_5 2017-08 25446_2017-08

df.info()
df["CreateDate"]= pd.to_datetime(df["CreateDate"])
df["NEW_DATE"]= df["CreateDate"].dt.strftime("%Y-%m")
df.head(5)

df["CartID"]= [str(row[0])+ "_"+ str(row[5]) for row in df.values]
df.head()


#######################
# Task 2: Create Association Rules and Make Suggestions
#######################

# Step 1: Create the basket and service pivot table as below

# Service 0_8 10_9 11_11 12_7 13_11 14_7..
# CartID
# 0_2017-08 0 0 0 0 0 0..
# 0_2017-09 0 0 0 0 0 0..
# 0_2018-01 0 0 0 0 0 0..
# 0_2018-04 0 0 0 0 0 1..
# 10000_2017-08 0 0 0 0 0 0

invoice_product_df=df.groupby(["CartID", "Service"])["Service"].count().unstack().fillna(0).applymap(lambda x:1 if x> 0 else 0)

# Step 2: Create association rules.

frequent_itemsets=apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules=association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

# Step 3: Use the arl_recommender function to recommend a service to a user who has received the 2_0 service in the last month.

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