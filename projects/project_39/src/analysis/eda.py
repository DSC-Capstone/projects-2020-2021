import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from collections import defaultdict

def analyze_data(arg1, year, dynamic_cols, target, wait_th, cpu_th,nominal,N):
    """
    Perform exploratory data analysis, further clean the data, and output dataset
    for the following model
    """

    def time_tf(x):
        if ((x[:4]) == year):
            return True
        else: return False

    outpath = "src/analysis"

    #data frame within the 2020 interval
    dynamic, static = arg1
    tf_list = dynamic['ts'].apply(time_tf)
    dynamic = dynamic[tf_list]

    #examine where the outlier starts
    dynamic[target].value_counts(bins = 3000)

    #feature engineering on dynamic_cols
    dynamic[dynamic_cols] = dynamic[dynamic_cols].astype(float)
    dynamic = dynamic[dynamic.before_cpuutil_max < int(cpu_th)]

    #set all values [0,1] as 0
    dynamic[dynamic_cols] = dynamic[dynamic_cols].apply(lambda x: x.apply(lambda y: 0 if (y < 1 and y > 0) else y))

    #log tranform because their distributions are strongly skewed
    dynamic['before_harddpf_max'] = np.log(dynamic['before_harddpf_max'])
    dynamic['before_diskutil_max'] = np.log(dynamic['before_diskutil_max'])
    dynamic['before_networkutil_max'] = np.log(dynamic['before_networkutil_max'])
    dynamic[dynamic_cols] = dynamic[dynamic_cols].apply(lambda x: x.apply(lambda y: 0 if y < 0 else y))

    #select the first 1200 rows of data
    subset = dynamic.head(1200)

    sns.pairplot(subset)
    #drop before_networkutil_max because it shows no pattern
    dynamic = dynamic.drop("before_networkutil_max", axis =1)

    subset = dynamic.head(1200)
    sns.pairplot(subset)
    plt.savefig(outpath+"/pairplot.png")

    plt.figure(figsize=(8, 4))
    sns.heatmap(subset.corr(), annot=True, linewidths=.5)
    plt.savefig(outpath+"/heatmap.png")

    #visualize the wait time
    dynamic['wait_secs'] = dynamic.wait_msecs.apply(lambda x: x*10**(-3))
    plt.figure(figsize = (8,4))
    plt.hist(dynamic.wait_secs)
    plt.title("Histogram of mouse wait time")
    plt.xlabel("Wait time in second")
    plt.ylabel("Frequency")
    plt.savefig(outpath + "/wait_distribution.png")

    #Scatter plot of each feature with wait time
    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.scatter(dynamic.before_cpuutil_max, dynamic.wait_secs, alpha = 0.2)
    plt.xlabel("log of maximum cpu utilization before mouse wait")
    plt.ylabel("wait sec")

    plt.subplot(1, 3, 2)
    plt.scatter(dynamic.before_harddpf_max, dynamic.wait_secs, alpha = 0.2)
    plt.xlabel("log of maximum hard page fault before mouse wait")

    plt.subplot(1, 3, 3)
    plt.scatter(dynamic.before_diskutil_max, dynamic.wait_secs, alpha = 0.2)
    plt.xlabel("log of maximum disk utilization before mouse wait")


    #create the data target
    dynamic['target'] = dynamic.wait_secs.apply(lambda x: 1 if x <= 5 else 2 if x <= 10 else 3)


    #feature engineering on static_cols
    def chi(table):
        stat, p, dof, expected = chi2_contingency(table)
        return p

    df = dynamic.merge(static, on = "guid", how = "left")
    # np.seterr(divide = 'ignore')
    pcollection = defaultdict(list)
    for i in range(int(N)):
        chi_sample = df.sample(5000)
        for j in nominal:
            data_crosstab = pd.crosstab(chi_sample[j],\
                                    chi_sample['target'], margins = False)
            pvalue = chi(data_crosstab)
            pcollection[j].append(pvalue)

    plt.figure(figsize=(5, 25))
    for i in range(len(nominal)):
        plt.subplot(len(nominal),1, i+1)
        plt.hist(pcollection[nominal[i]])
        plt.title(nominal[i])
    plt.savefig(outpath + "/pvalue_" + nominal[i] + ".png", bbox_inches="tight")

    df.to_csv("data/output/processed_data.csv", index = False)
    return df
