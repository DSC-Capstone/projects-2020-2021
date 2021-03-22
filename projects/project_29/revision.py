import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def get_all_df(route):
    result={}
    cvs=os.listdir(route)
    coron=[]
    pandemic=[]
    country=[]
    for i in cvs:
        path=os.path.join(route,i)
        df=pd.read_csv(path)
        if "Corona" in i:
            coron.append(df)
        elif 'country' in i:
            country.append(df)
        else:
            pandemic.append(df)
    result['Coronavirus']=pd.concat(coron,ignore_index=True)
    result['COVID-19_pandemic_by_country']=pd.concat(country, ignore_index=True)
    result['COVID-19_pandemic']=pd.concat(pandemic, ignore_index=True)
    return result

def get_user_activities(result, output):
    if not os.path.exists(output):
        os.makedirs(output)
    top10_percentage={}
    top100_percentage={}
    for i in result.keys():
        savefile=i+'_user_pie.png'
        outfile=os.path.join(output,savefile)
        user_freq=result[i]['user'].value_counts()
        user_10=user_freq.sort_values(ascending=False).head(10)
        plt.figure()
        user_10.plot.pie(title=i+'_user_pie')
        plt.savefig(outfile)
        plt.close()
        user_100=user_freq.sort_values(ascending=False).head(100)
        top100_percentage[i]=user_100.sum()/len(result[i])
        top10_percentage[i]=user_10.sum()/len(result[i])
    df=pd.Series(top10_percentage)
    outdf='top10_user_percentage.csv'
    outdffile=os.path.join(output, outdf)
    df.to_csv(outdffile)
    df2=pd.Series(top100_percentage)
    outdf2='top100_user_percentage.csv'
    outdf2file=os.path.join(output, outdf2)
    df2.to_csv(outdf2file)
    bar1=os.path.join(output, 'top10_editor.png')
    bar2=os.path.join(output, 'top100_editor.png')
    plt.figure()
    plt.bar(x=df.index, height=df.values)
    plt.xticks(rotation=90)
    plt.xlabel('article')
    plt.ylabel('percentage')
    plt.title('top10_editor_percentage')
    plt.savefig(bar1, bbox_inches='tight')
    plt.close()
    plt.figure()
    plt.bar(x=df2.index, height=df2.values)
    plt.xticks(rotation=90)
    plt.xlabel('article')
    plt.ylabel('percentage')
    plt.title('top100_editor_percentage')
    plt.savefig(bar2, bbox_inches='tight')
    plt.close()
       
       
    return

def plot_top_words(model, feature_names, n_top_words, title,output):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    outfile=os.path.join(output, title+'.png')
    plt.savefig(outfile)
    plt.show()
    plt.close()
    return



def LDA(result, output):
    for i in result.keys():
        section=result[i]['comment']
        tfidf=TfidfVectorizer(stop_words='english')
        tf=tfidf.fit_transform(section.dropna())
        lda=LatentDirichletAllocation(n_components=10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
        lda.fit(tf)
        name=tfidf.get_feature_names()
        title=i+'_topics_in_LDA_model'
        plot_top_words(lda, name,20, title,output)
        
    
    return