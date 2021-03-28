from nltk.corpus import stopwords
import string
import numpy as np

stpwrds = stopwords.words("english")

def bias(document):
    tmp = document.lower().translate(str.maketrans('', '', string.punctuation))
    input_str = ''
    d, r = [], []

    for i in tmp.split():
        if i not in stpwrds:
            try: 
                df = m.issue(i)
                df = df[df['year'] >= 2000]
                d.append(df['dem'].mean())
                r.append(df['rep'].mean())
            except KeyError:
                continue
    return np.mean(d) - np.mean(r)