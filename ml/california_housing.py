import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from typing import cast, Any

# ----Preparing data----
california_house_data = cast(Any, fetch_california_housing())
df = pd.DataFrame(data=california_house_data.data, columns=california_house_data.feature_names)
df["target"] = california_house_data.target

np.random.seed(42)
X = df.drop("target", axis=1)
y = df["target"]

# ----Features selection----
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X,y)

X_selected = X.loc[:, selector.support_]
# ["MedInc", "AveRooms", "AveBedrms", "Latitude", "Longtitude"]

# ----Pipeline----
X_train, X_test, y_train, y_test = train_test_split(X_selected,y,test_size=0.2)

model = Pipeline(steps=[
    ("model", RandomForestRegressor(random_state=42))
])

pipe_grid={
    "model__n_estimators":[70]
}
gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)
gs_model.fit(X_train, y_train)
score = gs_model.score(X_test, y_test)
print(score)
print(gs_model.best_params_)
# Accuracy 0.82

joblib.dump(gs_model,filename="models/california_housing_model_1.joblib")


