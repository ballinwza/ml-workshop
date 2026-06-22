import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from typing import cast, Any
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib

# ----Prepare data----
iris_data=cast(Any, load_iris())
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df["target"] = iris_data.target

# ----Setup Pipeline----
np.random.seed(42)
model = Pipeline(steps=[
    ("model", RandomForestClassifier())
])

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# ----Tuning hyperparameter----
pipe_grid = {
    'model__class_weight': [None, "balanced", "balanced_subsample"], 
}

gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)
gs_model.fit(X_train, y_train)

score = gs_model.score(X_test, y_test)
print(score)

# ----Export model----
joblib.dump(gs_model, filename="models/iris_model_1.joblib")