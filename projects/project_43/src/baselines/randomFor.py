import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def main(configs):
    train = pd.read_json(configs['train'])
    test = pd.read_json(configs['test'])
    final = pd.read_csv(configs['final'])

    train['ingredient_list'] = [','.join(x) for x in train.ingredients]
    ingredients = train['ingredient_list']

    vectorizer = TfidfVectorizer()
    tfidf_matrix= vectorizer.fit_transform(ingredients).todense()

    cuisines = train['cuisine']

    x_train, x_test, y_train, y_test = train_test_split(tfidf_matrix, cuisines, test_size=0.2)
    param_grid = {'n_estimators': [100]}

    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid.fit(x_train,y_train)

    y_pred = grid.predict(x_test)
    cuisines = train['cuisine'].value_counts().index

    test['ingredient_list'] = [','.join(x).strip() for x in test['ingredients']]
    test_ingredients = test['ingredient_list']
    test_tfidf_matrix = vectorizer.transform(test_ingredients)
    test_cuisines = grid.predict(test_tfidf_matrix)
    test['cuisine'] = test_cuisines

    final['ingredient_list'] = [''.join(x).strip() for x in final['ingredients']]
    final_ingredients = final['ingredient_list']
    final_tfidf_matrix = vectorizer.transform(final_ingredients)
    final_cuisines = grid.predict(final_tfidf_matrix)
    final['cuisine'] = final_cuisines
    final.to_csv(configs['output_location']+'/final_data.csv',index=False)

if __name__ == '__main__':
    main(sys.argv)