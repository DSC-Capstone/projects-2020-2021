from snapy import MinHash, LSH

def create_lsh(content, n_permutations, n_gram):
    labels = content.keys()
    values = content.values()
    #Create MinHash object
    minhash = MinHash(values, n_gram=n_gram, permutations=n_permutations, hash_bits=64, seed=3)
    #Create LSH model
    lsh = LSH(minhash, labels, no_of_bands=5)
    return lsh

def create_lsh_all(content_ls,n_permutations, n_gram):
    lsh_ls = {}
    for i in content_ls:
        lsh_ls[i] = create_lsh(content_ls[i], n_permutations, n_gram)
    return lsh_ls

def get_similar_ls(content_ls,n_permutations, n_gram):
    lsh_ls = create_lsh_all(content_ls,n_permutations, n_gram)
    edge_list = {}
    for i in lsh_ls:
        edge_list[i] = lsh_ls[i].edge_list(jaccard_weighted=True)
    
    similar_ls = []
    for e in edge_list:
        edges = edge_list[e]
        for i in edges:
            if type(i[1])==int and type(i[0])==str:
                similar_ls.append(i)
            elif (type(i[0])==int and type(i[1])==str):
                similar_ls.append(i)
    return similar_ls

