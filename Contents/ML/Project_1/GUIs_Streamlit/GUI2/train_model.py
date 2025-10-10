# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
# from sklearn.metrics import r2_score, make_scorer
# import joblib

# # Load data
# df = pd.read_csv('housing.csv')
# prices = df['MEDV']
# features = df.drop('MEDV', axis=1)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

# # Define model and parameters
# cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
# regressor = DecisionTreeRegressor(random_state=0)
# params = {
#     'max_depth': list(range(1, 11)),
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# scoring_fnc = make_scorer(r2_score)

# # Grid Search
# grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets, n_jobs=-1)
# grid.fit(X_train, y_train)

# # Best model
# best_reg = grid.best_estimator_
# print(f"Best parameters: {grid.best_params_}")
# print(f"Training R²: {r2_score(y_train, best_reg.predict(X_train)):.3f}")
# print(f"Test R²: {r2_score(y_test, best_reg.predict(X_test)):.3f}")

# # Save the model
# joblib.dump(best_reg, 'best_model.joblib')


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import r2_score, make_scorer
import joblib

# Load data
df = pd.read_csv(r'C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_1\housing.csv')
prices = df['MEDV']
features = df.drop('MEDV', axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

# Define model and parameters
cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
rf = RandomForestRegressor(random_state=0)
params = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
scoring_fnc = make_scorer(r2_score)

# Grid Search
grid = GridSearchCV(rf, params, scoring=scoring_fnc, cv=cv_sets, n_jobs=-1)
grid.fit(X_train, y_train)

# Best model
best_rf = grid.best_estimator_
print(f"Best parameters: {grid.best_params_}")
print(f"Training R²: {r2_score(y_train, best_rf.predict(X_train)):.3f}")
print(f"Test R²: {r2_score(y_test, best_rf.predict(X_test)):.3f}")

# Save the model
joblib.dump(best_rf, 'best_rf_model.joblib')