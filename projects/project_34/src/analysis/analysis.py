#!/usr/bin/env python
import pandas as pd
# TN TP FN FP plots
def TFNP_analysis(chains, reg):
    def cate(x):
        r = ''
        if x.is_mal == x.pred:
            r += 'T'
        else:
            r += 'F'
        if x.pred == 1:
            r += 'P'
        else:
            r += 'N'
        return r

    chains['pred'] = reg.predict(X)
    chains['analysis'] = chains.apply(cate, axis=1)
    fp_df = chains.loc[chains.analysis == 'FP']
    fn_df = chains.loc[chains.analysis == 'FN']
    tn_df = chains.loc[chains.analysis == 'TN']
    tp_df = chains.loc[chains.analysis == 'TP']
    fp_mean = fp_df.drop(['is_mal','pred'],axis=1).describe().loc['mean']
    fp_mean.name = 'FP'
    tn_mean = tn_df.drop(['is_mal','pred'],axis=1).describe().loc['mean']
    tn_mean.name = 'TN'
    fn_mean = fn_df.drop(['is_mal','pred'],axis=1).describe().loc['mean']
    fn_mean.name = 'FN'
    tp_mean = tp_df.drop(['is_mal','pred'],axis=1).describe().loc['mean']
    tp_mean.name = 'TP'
    pd.DataFrame([fp_mean,tn_mean,fn_mean,tp_mean]).T.plot(kind='bar');


def eda(chains, plots):
    chains = chains[['android','androidx','java','javax','kotlin','self','is_mal']]
    if 'describe' in plots:
        chains.describe()

    if 'api_num_bar' in plots:
        cb = chains.loc[chains.is_mal == 0].drop(['is_mal'],axis=1).describe().loc['mean']
        cb.name = 'Benign'
        cm = chains.loc[chains.is_mal == 1].drop(['is_mal'],axis=1).describe().loc['mean']
        cm.name = 'Malware'
        pd.DataFrame([cm,cb]).T.plot(kind='bar');
