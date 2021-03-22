import pandas as pd

def get_data(data_path, dynamic_name, static_name, col_to_transform, num_cols, static_cols, fill_method):
    '''
    Retrieve and clean the data
    '''

    #retrieve data
    file_path = data_path + "/"
    dynamic = pd.read_csv(file_path + dynamic_name)
    static = pd.read_csv(file_path + static_name)

    #clean data_cfg
    #age_category
    static[col_to_transform[0]] = static[col_to_transform[0]].fillna(method = fill_method).str[:1].replace({'U': '-1'})
    static[col_to_transform[1]] = static[col_to_transform[1]].str.strip(' ').str.strip('nm').replace({'Unknow': '-1'})
    static[num_cols] = static[num_cols].fillna(method = fill_method).astype(int)

    common_guid = set(dynamic.guid.unique()) & set(static.guid.unique())
    static_filter = static[static.guid.apply(lambda x: x in common_guid)].reset_index(drop = True)
    static = static_filter[static_cols + ['guid']]

    # print(dynamic.head())
    # print(static.head())

    static.to_csv("data/output/data_output.csv", index = False)

    return dynamic, static
