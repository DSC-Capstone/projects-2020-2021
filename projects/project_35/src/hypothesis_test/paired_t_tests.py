import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import scipy.stats as stats

def hypo_test_on_numerical(datapath, feature):
    # This function finds all pvalues between each persona with the given feature

    data = dd.read_csv(datapath,
                       delimiter ="\1",
                       assume_missing=True)
    results = {}

    #list for all types of users
    all_persona = list(data.persona.unique().compute())
    all_persona.remove('Unknown')
    print(all_persona)
    # test between all personas
    for i in all_persona:
        # data of ram of persona i
        d1 = data.loc[data['persona'] == i][feature]
        for j in all_persona:
            if (i,j) not in results.keys() and (i !=j):
                # data of persona i
                d2 = data.loc[data['persona'] == j][feature]
                #testing
                output = stats.ttest_ind(a = d1, b = d2)
                #put into results dict
                results[(i,j)] = output.pvalue
                print([(i,j)],':'+str(output.pvalue)+'\n')
                

            else:
                pass

    return results

def unsignificants(results_dict, significant_level):
    # This function finds all unsignificant values in the results list
    unsig = {}
    keys = results_dict.keys()
    for i in keys:
        if results_dict[i] > significant_level:
            unsig[i] = results_dict[i]
    return unsig
