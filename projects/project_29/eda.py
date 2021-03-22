import os
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

def get_dfs(road):
    '''
    get all filles with csvs in it
    '''
    result={}
   
    for filename in os.listdir(road):
        if filename.endswith('.csv'):
            fileroute=os.path.join(road,filename)
            filer=filename[:-13]
            result[filer]=pd.read_csv(fileroute)
            
    return result

def get_top_10_average_daily_view(dic, output):
    top10={}
    top10['article']=[]
    top10['average_view']=[]
    for i in dic.keys():
        top10['article'].append(i)
        top10['average_view'].append(dic[i]['pageview'].mean())
    top10er=pd.DataFrame(top10)
    top10er=top10er.sort_values(by=['average_view'], ascending=False)
    top10er=top10er.head(10)
    if not os.path.exists(output):
        os.makedirs(output)
    outfile=os.path.join(output, 'top10average_pageview.csv')
    outimage=os.path.join(output, 'top10pageview.png')
    top10er.to_csv(outfile, index=False)
    plt.figure()
    plt.bar(x=top10er['article'], height=top10er['average_view'])
    plt.xticks(rotation=90)
    plt.xlabel('article')
    plt.ylabel('average_view')
    plt.savefig(outimage, bbox_inches='tight')
    plt.close()
    return top10er

def plot_top10(dict1,dict2, outpath):
    for i in dict1['article']:
        df=dict2[i]
        timer=df['timestamp'].values
        retime=[]
        dater=[]
        for m in timer:
            m=str(m)
            retime.append(m)
        for j in retime:
            dater.append(date(int(j[:4]), int(j[4:6]),int(j[6:8])))
        outer=i+'_pageview.png'
        putpath=os.path.join(outpath, outer)
        plt.figure()
        plt.plot(dater, df['pageview'].values)
        plt.xticks(rotation=45)
        plt.xlabel('date')
        plt.ylabel('daily_pageview')
        plt.savefig(putpath, bbox_inches='tight')
        plt.close()
    return
        