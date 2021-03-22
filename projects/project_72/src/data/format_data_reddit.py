from ast import literal_eval
def make_content_text(content_ls,df_ls):
    for df_id in df_ls:
        df = df_ls[df_id]
        posts = df['clean_words']
        for post_id,post in zip(posts.index,posts):
            for word_id, word in enumerate(post):
                if len(word) > 3:
                    key=df_id+'_post_'+str(post_id)+'_word_'+str(word_id)
                    word = word.lower()
                    char=word[:4]
                    if char in content_ls:
                        content_ls[char].update({key:word})
                    else: 
                        content_ls[char] = {key:word}
def make_content_search(search_terms):
    #Initialize a dictionary
    content_ls ={}
    #Add search terms to dictionary with numbered key
    for i in range(len(search_terms)):
        term = search_terms[i]
        if len(term) > 3:
            char = term[:4]
            if char in content_ls:
                content_ls[char].update({i:term.strip().lower()})
            else:
                content_ls[char] = {i:term.strip().lower()}
    return content_ls



