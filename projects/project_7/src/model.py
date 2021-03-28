import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def create_model(training_data, n_estimators, max_depth, min_samples_split):
  training_data = pd.read_csv(training_data)
  
  clf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    criterion='entropy',
    min_samples_split=min_samples_split,
    random_state=42
  )

  X, y = training_data.drop(columns=['resolution']), training_data['resolution']

  clf.fit(X, y)

  return clf


