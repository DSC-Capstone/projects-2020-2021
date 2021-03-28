def make_content_text(content_ls,docs):
    for doc in docs:
    #Add noun to dictionary with its location as key
        for sent_id in doc.sentences:
            sent=doc.sentences[sent_id]
            for n_id,noun in enumerate(sent.noun):
                if len(noun) > 3:
                    key=doc.doc_id+'_sentence_'+str(sent_id)+'_noun_'+str(n_id)
                    noun = noun.lower()
                    char=noun[0]
                    if char in content_ls:
                        content_ls[char].update({key:noun})
                    else: 
                        content_ls[char] = {key:noun}
                        
def make_content_search(search_terms):
    #Initialize a dictionary
    content_ls ={}
    #Add search terms to dictionary with numbered key
    for i in range(len(search_terms)):
        term = search_terms[i]
        if len(term) > 3:
            char = term[0]
            if char in content_ls:
                content_ls[char].update({i:term.strip().lower()})
            else:
                content_ls[char] = {i:term.strip().lower()}
    return content_ls



