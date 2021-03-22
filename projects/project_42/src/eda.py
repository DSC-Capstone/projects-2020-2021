import os
from collections import defaultdict
import re
import json
import shutil
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb

ori_dir = "/AutoPhrase"
dir = os.getcwd()

def save_eda_data(df, positive_phrases, negative_phrases, outdir, out_txt, review):
    '''Save the sentiment output in outdir for eda.'''
    df.to_csv(outdir + '/restaurant.csv')
    positive_phrases.to_csv(outdir + '/positive.csv')
    negative_phrases.to_csv(outdir + '/negative.csv')
    
    user_review_count_df, picked_user = user_distribution(review)
    fname = generate_user_review_txt(picked_user, review, out_txt)
    if any(fname):
        run_autophrase(fname, out_txt)
    else:
        print('No txt file for user ' + i) 
    
    # save the user review count dataframe
    user_review_count_df.to_csv(outdir + "/user_review_count.csv")
    
    return

def user_distribution(review):
    """
    generate the user review distribution for 20 most review and the overall distribution of reviews
    
    return a list of user id includes the most and second most reviews user, a random user
    """
    user_df = review.user_id.value_counts().sort_values(ascending=False)
    
    # generate the user distribution plot
    x = user_df.iloc[0:20] 
    plt.figure(figsize=(16,4))
    ax = sns.barplot(x.index, x.values, alpha=0.8)
    plt.title("20 Users with Most Reviews ")
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.ylabel('number of reviews', fontsize=12)
    plt.xlabel('User', fontsize=12)

    # adding the text labels
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
    print('save the user distribution image now')
    plt.savefig("./reference/img/most_20_user.png")
    
    user_list = [user_df.index[0], user_df.index[1], user_df.index[len(user_df) // 2]]
    return pd.DataFrame({'user_id':user_df.index, 'count':user_df.values}), user_list

def generate_user_review_txt(user_id_list, reviews, path):
    '''
    generate the txt file that contains all reviews of a user
    @param user_id: string of the user unique id
    @param reviews: all reviews csv file
    @return: the file path that store the txt file
    '''
    # filter out the user review record from all reviews
    txt_list = []
    
    for i in user_id_list:
        user_df = reviews.loc[reviews['user_id'] == i]
        
        # if the user does not have any reviews before
        if not len(user_df):
            print('The user does not have any previous review record')
            return None
        print('generate the txt file for User' + i) 
        
        # store the user reviews txt file under the reference/user_reviews folder
        file_path = i + '.txt'
        txt_list.append(file_path)
        user_df[['text']].to_csv(path + '/' + file_path, header=None, index=None, sep=',', mode='a')   
    return txt_list

def run_autophrase(txt_list, path):
#     pdb.set_trace()
    try:
        shutil.copytree(path, dir + ori_dir + '/data/EN/txt')
    except:
        shutil.rmtree(dir + ori_dir + '/data/EN/txt')
        shutil.copytree(path, dir + ori_dir + '/data/EN/txt')
    for name in txt_list:
        if name is not None:
            with open(dir+ ori_dir + "/auto_phrase.sh",'r') as f , open(dir + ori_dir+ "/tmp_autophrase.sh",'w') as new_f:
                autophrased = [next(f) for x in range(90)] # get the autophase part
                index = 0
                for i in autophrased:
                    if index != 15:
                        new_f.write(i)
                    else:
                        new_f.write('DEFAULT_TRAIN=${DATA_DIR}/EN/txt/' + name + '\n')
                    index += 1

        # change the access of the bash script
        os.chmod("./AutoPhrase/tmp_autophrase.sh", 509)
        os.chdir(dir + ori_dir)
        subprocess.run(["./tmp_autophrase.sh"])
        
    
        # move the result to the result folder
        shutil.copy(dir + ori_dir + '/models/DBLP/AutoPhrase.txt', dir+ '/reference/AutoPhrase_result/AutoPhrase_' + name)
        os.chdir(dir)
        print('Autophrase for User ' + name + ' is Done!')
    
    # remove the temporary bash script
    os.remove(dir + ori_dir + "/tmp_autophrase.sh")
    shutil.rmtree(dir + ori_dir + '/data/EN/txt')
    return
