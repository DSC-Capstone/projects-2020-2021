import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

def main(configs):
    recipes = pd.read_csv(configs['final'])
    recipes['calories'] =  [eval(x)[0] for x in recipes['nutrition']]
    recipes = recipes[eval(configs['conRecCol'])][:configs['size']]
    
    test = recipes['ingredients'].apply(lambda x: eval(x))
    
    mlb = MultiLabelBinarizer()
    mlb.fit(test)
    
    input_vector = eval(configs['sampleInput'])
    
    ingredients_transformed = mlb.transform(test)
    recipe_test_trans = mlb.transform(input_vector)

    sims = []
    for recipe in ingredients_transformed:
        sim = cosine_similarity(recipe_test_trans,recipe.reshape(-1,len(recipe)))
        sims.append(sim)

    recipes['sim'] = [x[0][0] for x in sims]
    print(recipes.set_index('name')[:5])
    return 
    
if __name__ == '__main__':
    main(sys.argv)