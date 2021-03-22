import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def tf_idf_result(reviews, business_df, location_list, amount = 20):
    print('-------- Running TF-IDF Recommendation method --------')
    # process the business csv
    business_df[['categories']] = business_df[['categories']].fillna(value= '')
    restaurants_business = business_df[business_df.categories.str.contains("Restaurants")]
    for i in location_list:
        print('Start Processing TF-IDF on City: ' + i)
        bus_rev_city = (restaurants_business[restaurants_business.city == i]).merge(reviews, on = 'business_id')
        rest_df = preprocess_group_restaurant_review(bus_rev_city, restaurants_business)
        rest_top = generate_rest_recommend(rest_df, amount)        
        rest_top_sn = rest_df[['business_id','stars','name']].merge(rest_top)
        result_df = restaurants_business[['business_id','categories','review_count','address']].merge(
            rest_top_sn).drop_duplicates().reset_index(drop = True)
        print('Store the ' + i + ' Recommendation Result into Reference Folder...')
        file_name = 'LV_rest_info.csv' if i == 'Las Vegas' else 'PX_rest_info.csv'
        result_df.to_csv('reference/dataframe/' + file_name)
        print('City ' + i + ' TF-IDF finished')
        print('\n')
    
    # make recommendation based on the random rest and user id, store the recommendation list in the reference folder
    #make_recommendations(random_rest, rest_feature_matrix, reviews[['name','business_id','categories']], k=amount).to_csv(
    #'./reference/dataframe/restaurant_recommendation.csv')
    print('------- TF-IDF Recommendation method finished --------')
    return 

def prepare_review(rest_path, business_path):
    return pd.read_csv(rest_path), pd.read_csv(business_path)

def preprocess_group_restaurant_review(review_df, business_df):
    restaurant_df = review_df[['text','stars','business_id']].groupby('business_id').agg({'text':list,
                                                                                          'stars':np.mean}
                                                                                        ).reset_index()
    restaurant_df.text = restaurant_df.text.apply(lambda t: "".join(re.sub(r'[^\w\s]',' ',str(t))).replace("\n"," "))
    return restaurant_df.merge(business_df[['name', 'business_id']], on = 'business_id')

def generate_rest_recommend(grouped_df, n):
    # initialize and train the tf-idf vectorizer
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(grouped_df.text.apply(lambda x: x.lower()))
    # calculate the cosine similarity of tf-idf result
    cosine_sim = cosine_similarity(tfidf_matrix)
    # match the matrix with the restaurant
    rest_df = pd.DataFrame(cosine_sim, columns=grouped_df.business_id, index=grouped_df.business_id)
    # generate top 20 recommended restaurant dataframe for all restaurants
    column = ['business_id']
    for i in range(n):
        column_name = 'Top_' + str(i)
        column.append(column_name)
    top_df = pd.DataFrame(columns = column)
    # iterate through all restaurants in the dataframe
    for i in rest_df.index:
        ix = rest_df.loc[:,i].to_numpy().argpartition(range(-1,-n,-1))
        high_recomm = rest_df.columns[ix[-1:-(n+2):-1]]
        top_df = top_df.append(pd.Series(high_recomm.values, index = column),ignore_index=True)
    return top_df
