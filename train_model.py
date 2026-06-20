import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("dataset.csv")

X = data.drop("prognosis", axis=1)
y = data["prognosis"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")