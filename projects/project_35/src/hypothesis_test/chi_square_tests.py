import pandas as pd
from scipy.stats import chi2_contingency

def data_cleaning(df):
    used_cols =['chassistype',
                'chassistype_2in1_category',
                'countryname_normalized',
                'modelvendor_normalized',
                'model_normalized',
                'ram',
                'os',
                '#ofcores',
                'age_category',
                'graphicsmanuf',
                'graphicscardclass',
                'processornumber',
                'cpuvendor',
                'cpu_family',
                'cpu_suffix',
                'screensize_category',
                'persona',
                'processor_line',
                'vpro_enabled',
                'discretegraphics']
    df = df[used_cols]

    #cleaning
    df = df.dropna()
    df = df[df.persona!= 'Unknown'].reset_index(drop=True)
    df = df[df.processornumber!= 'Unknown'].reset_index(drop=True)

    df['processornumber'] = df['processornumber'].apply(lambda x: x[:2] ).astype('int32',errors='raise')
    df['ram'] =df['ram'].astype('int32')
    df['#ofcores'] =df['#ofcores'].astype('int32',errors='raise')

    return df


def chi_square_test(data_path,alpha):
    #read the data
    sys_info = pd.read_csv(data_path, delimiter ="\1")

    #clean the data
    df = data_cleaning(sys_info)

    cat_cols = ['chassistype',
                'chassistype_2in1_category',
                'countryname_normalized',
                'modelvendor_normalized',
                'model_normalized',
                'os',
                'age_category',
                'graphicsmanuf',
                'graphicscardclass',
                'cpuvendor',
                'cpu_family',
                'cpu_suffix',
                'screensize_category',
                'processor_line',
                'vpro_enabled',
                'discretegraphics']
    out_df = {}
    for i in cat_cols:
        #create a crosstable of that categorical feature
        ct = pd.crosstab(df.persona,df[i])

        #chi square tests
        stat, p, dof, expected = chi2_contingency(ct)

        out_df[('persona',i)] = p

        # interpret p-value
        print("p value between the persona and "+i+" is: " + str(p))
        if p <= alpha:
            print('They are dependent (reject H0)')
        else:
            print('They are independent (H0 holds true)')

    out_df = pd.DataFrame.from_dict(out_df,orient='index',columns=['p-values'])
    return out_df
