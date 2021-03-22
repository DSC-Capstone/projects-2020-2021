import os
import numpy as np
import shutil
import pandas as pd
import pdb
import json
import subprocess

import pyspark.ml as M
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

ori_dir = "/AutoPhrase"
dir = os.getcwd()

spark = SparkSession \
    .builder \
    .appName("yelp-reccomender") \
    .getOrCreate()


def reviews_by_city(city_name, review_path, business_path):
    '''Create a subset for the given city'''
    print(' --------Creating subset for: ' + city_name + ' --------')
    # Set up schema for pyspark
    business_schema = T.StructType([
    T.StructField("name", T.StringType(), True),
    T.StructField("business_id", T.StringType(), True),   
    T.StructField("city", T.StringType(), True),   
    T.StructField("categories", T.StringType(), True),
    T.StructField("address", T.StringType(), True),
    T.StructField("review_count", T.IntegerType(), True),
    T.StructField("hours", T.StringType(), True),                  
    ])

    reviews_schema = T.StructType([
        T.StructField("review_id", T.StringType(), True),   
        T.StructField("business_id", T.StringType(), True),   
        T.StructField("text", T.StringType(), True),      
        T.StructField("stars", T.FloatType(), True),
        T.StructField("user_id", T.StringType(),  True),                  
    ])
    
    # Read in data with pyspark
    review_path = "./" + review_path
    business_path = "./" + business_path
    business = spark.read.csv(business_path, header=True, multiLine=True, schema=business_schema, quote="\"", escape="\"")
    reviews = spark.read.csv(review_path, header=True, multiLine=True, schema=reviews_schema, quote="\"", escape="\"")
    
    # Filtering process for city; categories must include restaurants/food
    df = business.filter(business.city == city_name.replace('_', ' '))\
.filter(business.categories.contains("Restaurants")|business.categories.contains('Food'))\
.join(reviews, business.business_id == reviews.business_id, 'inner')\
.drop(reviews.business_id)
#.drop(reviews._c01).drop(business._c01)

    city_name = city_name.replace(' ', '_')
    
    # Save city as CSV
    res = df.toPandas().drop_duplicates(subset='review_id').reset_index(drop=True)
    res.to_csv('./data/tmp/' + city_name + '_subset.csv')
    
    # Annotate the reviews text with this delimiter for phrasal segmentation
    with open('data/tmp/reviews/' + city_name + '.txt', 'w') as f:
        content = res['text'].str.cat(sep="REVIEW_DELIMITER")
        f.write(json.dumps(content))   

    print('Subset Created')
    
def autophrase_reviews(txt_list, path='data/tmp/reviews'):
    '''Perform AutoPhrase on the Reviews Text'''
    print('Starting AutoPhrase')
    # Matching directories with the AutoPhrase Repo
    try:
        shutil.copytree(path, dir + ori_dir + '/data/EN/reviews')
    except:
        shutil.rmtree(dir + ori_dir + '/data/EN/reviews')
        shutil.copytree(path, dir + ori_dir + '/data/EN/reviews')
        
    # Adjust the AutoPhrase bash for our requirements
    for name in txt_list:
        if name is not None:
            name += '.txt'
            with open(dir+ ori_dir + "/auto_phrase.sh",'r') as f , open(dir + ori_dir+ "/tmp_autophrase.sh",'w') as new_f:
                autophrased = [next(f) for x in range(146)] # get the autophase part
                index = 0
                for i in autophrased:
                    if index != 23:
                        new_f.write(i)
                    else:
                        new_f.write('DEFAULT_TRAIN=${DATA_DIR}/EN/reviews/' + name + '\n')
                    index += 1
            with open(dir+ ori_dir + "/phrasal_segmentation.sh",'r') as f , open(dir + ori_dir+ "/tmp_autophrase.sh",'a') as new_f:
                autophrased = [next(f) for x in range(90)] 
                index = 0
                new_f.write('\n')
                for i in autophrased:
                    if index != 14:
                        new_f.write(i)
                    else:
                        new_f.write('TEXT_TO_SEG=${TEXT_TO_SEG:- ${DATA_DIR}/EN/reviews/' + name + '}\n')
                    index += 1
        # change the access of the bash script
        os.chmod("./AutoPhrase/tmp_autophrase.sh", 509)
        os.chdir(dir + ori_dir)
        subprocess.run(["./tmp_autophrase.sh"])

        # move the result to the result folder
        shutil.copy(dir + ori_dir + '/models/DBLP/segmentation.txt', dir+ '/reference/AutoPhrase_result/' + name)
        os.chdir(dir)
        print('Autophrase for ' + name + ' is Done!')
            
    # remove the temporary bash script
    os.remove(dir + ori_dir + "/tmp_autophrase.sh")
    shutil.rmtree(dir + ori_dir + '/data/EN/reviews')
    return
    

def split_data(test_txt, test_user, business_csv, review_test, **kwargs):
    '''Return txt back to rows, each row being a review'''
    with open(test_txt, 'r') as f:
        reviews_string = f.read()
        reviews_list = reviews_string.split("REVIEW_DELIMITER")

    reviews_list = pd.Series(reviews_list)[:-1] # last entry is always empty
    dtypes = {
        'review_id' : np.str,
        'business_id': np.str,
        'text': np.str,
        'stars': np.float64,
        'user_id': np.str
        
    }
    return reviews_list , pd.read_csv(test_user, dtype = dtypes), pd.read_csv(business_csv), pd.read_csv(review_test)

def json_to_csv(review_path_in, review_path_out, business_path_in, business_path_out):
    '''convert the json file into the csv file we needed'''
    print("----- Start Converting Business Json file into CSV file -----")
    business_data = {"name":[],"business_id":[],"city":[],"categories":[], "address":[], "review_count":[], "hours":[]}
    with open(business_path_in) as f:
        for line in f:
            business = json.loads(line)
            business_data['name'].append(business['name'])
            business_data['business_id'].append(business['business_id'])
            business_data['city'].append(business['city'])
            business_data['categories'].append(business['categories'])
            business_data['address'].append(business['address'])
            business_data['review_count'].append(business['review_count'])
            business_data['hours'].append(business['hours'])
    business_df = pd.DataFrame(business_data)
    business_df.to_csv(business_path_out, index = False)
    print("----- Finish Converting Business Json file into CSV file -----")

    print("----- Start Converting Review Json file into CSV file -----")
    review_data = {"review_id":[],"business_id":[],"text":[],"stars":[], "user_id":[]}
    with open(review_path_in) as fi:
        for line in fi:
            review = json.loads(line)
            review_data['review_id'].append(review['review_id'])
            review_data['business_id'].append(review['business_id'])
            review_data['text'].append(review['text'])
            review_data['stars'].append(review['stars'])
            review_data['user_id'].append(review['user_id'])
    review_df = pd.DataFrame(review_data)
    review_df.to_csv(review_path_out, index = False)
    print("----- Finish Converting Review Json file into CSV file -----")
    return 

def check_result_folder(out_df, out_img, out_txt, out_autophrase, **kwargs):
    # create the result placement folder if does not exist
    print(' --------Checking the reference folder now ------')
    if os.path.isdir(out_df) is False:
        command = 'mkdir -p ' + out_df
        os.system(command)
    if os.path.isdir(out_img) is False:
        command = 'mkdir -p ' + out_img
        os.system(command)
    if os.path.isdir(out_txt) is False:
        command = 'mkdir -p ' + out_txt
        os.system(command)
    if os.path.isdir(out_autophrase) is False:
        command = 'mkdir -p ' + out_autophrase
        os.system(command)
    if os.path.isdir('data/tmp') is False:
        command = 'mkdir -p ' + 'data/tmp'
        os.system(command)
    if os.path.isdir('data/raw') is False:
        command = 'mkdir -p ' + 'data/raw'
        os.system(command)
    if os.path.isdir('data/tmp/reviews') is False:
        command = 'mkdir -p ' + 'data/tmp/reviews'
        os.system(command)
    if os.path.isdir('../AutoPhrase/data/EN/reviews') is False:
        command = 'mkdir -p ' + '../AutoPhrase/data/EN/reviews'
        os.system(command)     
    print(' --------The reference folder is ready ------')
    return

def clean_repo():
    if os.path.isdir('reference') is True:
        shutil.rmtree('reference')
    if os.path.isdir('AutoPhrase/tmp') is True:
        shutil.rmtree('AutoPhrase/tmp')
    if os.path.isdir('tmp') is True:
        shutil.rmtree('tmp')
    if os.path.isdir('AutoPhrase/data/txt') is True:
        shutil.rmtree('AutoPhrase/data/txt')
    if os.path.isdir('data/raw') is True:
        shutil.rmtree('data/raw')
    if os.path.isdir('data/tmp') is True:
        shutil.rmtree('data/tmp')
    return