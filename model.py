#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import pandas as pd
import pickle

#1. Model saved with pickle
MODEL_PATH = 'models/log_best_reg.pkl'
print(" MODEL_PATH :",MODEL_PATH)
with open(MODEL_PATH, 'rb') as pickle_file:
    sentiment_model_load = pickle.load(pickle_file)
# 2. Load tfidf vectorizer    
tfidf_vect_path = 'models/tfidf_vect.pkl'
with open(tfidf_vect_path, 'rb') as pickle_file:
    tfidf_vectorizer = pickle.load(pickle_file)
# 3. Load  user prediction and processed reviews
recommendation_PATH = 'models/recommednation_user_final_rating.csv'
processed_reviews_path = 'data/sentiment_analysis_processed_reviews.csv'
user_final_rating = pd.read_csv(recommendation_PATH ,index_col='user_name')
processed_prod_df = pd.read_csv(processed_reviews_path)

# featurization of processed reviews
def convert_text_features(processed_reviews):
    
    # TFIDF vectorizer
    #vect = TfidfVectorizer()
    tfidf_vect=tfidf_vectorizer.transform(processed_reviews)
    train_features = pd.DataFrame(tfidf_vect.toarray(),columns=tfidf_vectorizer.get_feature_names())
    train_features.reset_index(drop=True,inplace=True)
    X = train_features
    return X

# to predict the sentiments
def sentiment_model_predict(X_features):
    predicted_sentiments = sentiment_model_load.predict(X_features)
    return predicted_sentiments
def top20_prod_final_rating(username):
    top20_products = user_final_rating.loc[username].sort_values(ascending=False)[0:20]
    return top20_products

def top20_products_reviews(username):#, recommendation_PATH,processed_reviews_path):
    # Top 20 products are 
    top20_products = top20_prod_final_rating(username)
    top20prod_df = top20_products.to_frame().reset_index().rename(columns={'index': 'prod_name'})
    # merging with processed df to get the processed reviews for top 20 products
    reviews_top20_df= pd.merge(top20prod_df,processed_prod_df,left_on='prod_name',right_on='name', how = 'left')
    return reviews_top20_df[['prod_name','reviews_text_processed']]

def perform_sentiment_on_top20ProdReviews(username):# ,MODEL_PATH, recommendation_PATH,processed_reviews_path):
    top20_prod_reviews = top20_products_reviews(username)# , recommendation_PATH,processed_reviews_path)
    reviews_txt=[doc for doc in top20_prod_reviews['reviews_text_processed']]
    X_features = convert_text_features(reviews_txt)
    predicted_sentiment = sentiment_model_predict(X_features)
    final_sentiment_df = pd.DataFrame(predicted_sentiment,columns=['predicted_sentiment'])
    final_sentiment_df['prod_name'] = top20_prod_reviews['prod_name']
    return final_sentiment_df

def cal_top5_prod(username):    
    sentiment_top20_prods = perform_sentiment_on_top20ProdReviews(username)
    df = sentiment_top20_prods.pivot_table(index ='prod_name' , columns='predicted_sentiment' , values ='predicted_sentiment',
                              aggfunc ={ 'predicted_sentiment':'count'} )
    df = df.reset_index()
    df = df.rename(columns={0: 'class0', 1: 'class1'})
    df['positive_percentage'] = (df['class1']/(df['class0']+df['class1']))*100
    df = df.sort_values(by = 'positive_percentage' , ascending = False)  
    #df = df.reset_index().reset_index()
    #df = df.rename(columns ={'level_0':'Sno.' , 'prod_name':'ProductName'})
    
    return df['prod_name'][0:5]

