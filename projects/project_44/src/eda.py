import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import numpy as np

def do_eda(out_dir,input_path,file):
    print('Creating EDA Graphs')
    if file == "DBLP.5K":
        data_kk = pd.read_csv(input_path, header=None, names=['sentence'])
        data_kk['length'] = data_kk['sentence'].apply(lambda x: len(str(x).split(' ')))
        data_kk['length_sentnece'] = data_kk['sentence'].apply(lambda x: len(str(x).split('.'))-1)
        
        #input words distribution for each sentence
        plt.semilogy(data_kk['length'],label='words length', color='purple')
        plt.title('Input Words Length Distribution for Each Line')
        plt.xlabel('Each Line')
        plt.ylabel('Length')
        plt.legend(loc = 'upper right')
        plt.savefig(out_dir+'word_distribution.png')
        plt.close()
        
        #box plot length
        plt.figure()
        sns.boxplot(x=data_kk['length']).set_title('Box Plot of Words Lengths in Each Sentence')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.savefig(out_dir+'box_plot_word_length.png')
        plt.close()
     
        #cleaned
        mean_kk = data_kk['length'].describe()['mean']
        std_kk = data_kk['length'].describe()['std']
        percent_kk = mean_kk+std_kk*3
        cleaned_kk = data_kk[data_kk['length']<percent_kk]
        plt.figure()
        plt.hist(cleaned_kk['length'])  
        plt.title('Cleaned Distribution '+file)
        plt.xlabel('Length Distribution')
        plt.ylabel('Frequency')
        plt.savefig(out_dir+'cleaned_set.png')
        plt.close()
        
        #token
        tokens_kk = data_kk['sentence'].str.split(expand=True).stack().value_counts().to_dict()
        token = pd.DataFrame.from_dict(tokens_kk,orient='index', columns = ['count'])
        plt.figure()
        token['words'] = token.index
        token = token.reset_index(drop = True)
        plt.scatter(x = list(token[:20]['count']),
                            y =list(token[:20]['words']),
                            linewidths = 2,
                            edgecolor='b',
                            alpha = 0.5)
        plt.title('Tokenization Top20 Words Frequency')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.savefig(out_dir+'tokens_top20_words.png')
        plt.close()
        
        token_arr_kk = list(tokens_kk.values())
        plt.figure()
        plt.semilogy(list(tokens_kk.values()),color = 'pink')
        plt.title('tokens frequency in descending order' )
        plt.xlabel('tokens index')
        plt.ylabel('Frequency')
        plt.savefig(out_dir+'tokens_freq.png')
        plt.close()
        
        num_rare_kk = sum(i < 5 for i in token_arr_kk)
        strs_kk = 'There are '+ str(data_kk['length_sentnece'].sum())+' sentences in this input text file. '+'The mean of the input text word length'+ ' is around ' + str(round(mean_kk)) + ' for each sentence with the standard deviation ' + str(round(std_kk)) + '. Number of Rare tokens is ' + str(num_rare_kk) +' (which defined as the the number of tokens is less than 5).'
        f = open(out_dir + "description.txt", "a")
        f.write(strs_kk)
        f.close()

        f = open(out_dir + "description.txt", "a")
        f.write(strs_kk)
        f.close()
        print('Done')

    if file == "DBLP":
        data = pd.read_csv(input_path, header=None, names=['sentence'])
        data['length'] = data['sentence'].apply(lambda x: len(str(x).split(' ')))
        #length outlier
        plt.figure()
        plt.hist(data['length'], bins = 100)  
        plt.title(file+'_outlier')
        plt.savefig(out_dir+'outlier'+'.png')
        plt.close()
        #box plot length
        plt.figure()
        sns.boxplot(x=data['length']).set_title('box plot of length sentences '+file)
        plt.savefig(out_dir+'boxplot.png')
        plt.close()

        mean = data['length'].describe()['mean']
        std = data['length'].describe()['std']
        percent = mean+std*3
        cleaned = data[data['length']<percent]
        plt.figure()
        plt.hist(cleaned['length'], bins = 20)  
        plt.title('Cleaned set '+file)
        plt.savefig(out_dir+'cleaned_set.png')
        plt.close()

        prev = 0
        output = Counter({})
        print('Tokenizing..this may take more than 10 mins')
        for i in tqdm((np.arange(10000,2773022,10000).tolist()+[2773022])):
            tokens = data['sentence'][prev:i].str.split(expand=True).stack().value_counts().to_dict()
            output += Counter(tokens)
            prev = i
        
        token_arr = list(output.values())
        num_rare = sum(i < 5 for i in token_arr)

        plt.figure()
        plt.hist(list(output.values()), bins = 80)
        plt.title('tokens distribution of '+ file)
        plt.savefig(out_dir+'tokens_distribution.png')
        plt.close()

        strs = 'Mean for length distribution of '+ file+ ' is ' + str(mean) + '. Std is ' + str(std) + '. Number of Rare tokens is ' + str(num_rare) +'.'
        f = open(out_dir+ "description.txt", "a")
        f.write(strs)
        f.close()
        print('Done')

    return
