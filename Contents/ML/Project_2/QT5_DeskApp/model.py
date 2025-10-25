import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
import joblib # Import joblib for saving

# Load the Census dataset
data = pd.read_csv(r"C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_2\census.csv")


# Success - Display the first record
display(data.head())

# --- Basic Data Exploration (No changes) ---
n_records =  data.shape[0]
n_greater_50k = data[data['income'] == '>50K'].shape[0]
n_at_most_50k = data[data['income'] == '<=50K'].shape[0]
greater_percent = round((n_greater_50k / n_records) * 100 , 2)
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))


# --- Preprocessing (No changes) ---
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

skewed = ['capital-gain', 'capital-loss']
features_log_transformed = features_raw.copy()
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))


# --- Scaling (No changes) ---
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# --- !!! MODIFICATION IS HERE !!! ---
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# We will select ONLY the columns that our UI uses
# before we do the one-hot encoding.
columns_to_keep = [
    'age', 
    'workclass', 
    'education-num', 
    'capital-gain', 
    'capital-loss', 
    'hours-per-week'
]

# Overwrite the DataFrame to only include these columns
features_subset = features_log_minmax_transform[columns_to_keep]

print(f"\n--- Shape before subsetting: {features_log_minmax_transform.shape} ---")
print(f"--- Shape after subsetting: {features_subset.shape} ---\n")


# TODO: One-hot encode the *SUBSET* data using pandas.get_dummies()
features_final = pd.get_dummies(features_subset) # Was: pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.map({'<=50K': 0, '>50K': 1})

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))
print("(This should be a much smaller number than 103)")

# Uncomment the following line to see the encoded feature names
# print(encoded)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# --- !!! END OF MODIFICATION !!! ---
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# --- All code below this line is for training ---
# --- No changes needed here, just run it all ---

# --- Naive Predictor ---
TP = np.sum(income)  # True Positives
FP = income.count() - TP  # False Positives
accuracy = TP / income.count()
precision = TP / (TP + FP)
recall = TP / (TP + 0) # FN = 0
beta = 0.5
fscore = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# --- Helper Function ---
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    results = {}
    start = time()
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    results['train_time'] = end - start
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()
    results['pred_time'] = end - start
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
    results['acc_test']  = accuracy_score(y_test, predictions_test)
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
    results['f_test']  = fbeta_score(y_test, predictions_test, beta=0.5)
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    return results


# --- Model Comparison ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

clf_A = DecisionTreeClassifier(random_state=0)
clf_B = RandomForestClassifier(random_state=0)
clf_C = AdaBoostClassifier(random_state=0)

samples_100 = len(y_train)
samples_10 = int(len(y_train) * 0.1)
samples_1 = int(len(y_train) * 0.01)

results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)


# --- Grid Search ---
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

clf = AdaBoostClassifier(random_state=42)
parameters = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.5, 1.0, 1.5]
}
scorer = make_scorer(fbeta_score, beta=0.5)
grid_obj = GridSearchCV(estimator=clf, param_grid=parameters, scoring=scorer, cv=5, n_jobs=-1)
grid_fit = grid_obj.fit(X_train, y_train)
best_clf = grid_fit.best_estimator_

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5)))
print("\nBest Parameters Found:\n", grid_fit.best_params_)


# --- Feature Importance (and reduced model test) ---
# This part is less critical now but good to keep
importances = best_clf.feature_importances_ # Use best_clf
features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)
print("\nTop 10 Features (from new model):")
print(importance_df)

from sklearn.base import clone
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]
clf_reduced = (clone(best_clf)).fit(X_train_reduced, y_train)
reduced_predictions = clf_reduced.predict(X_test_reduced)

print("\nFinal Model trained on full (subset) data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced (top 5) data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))) 


# --- SAVE THE NEW MODEL AND SCALER ---
print("\nSaving new model and scaler...")
# Save full model (which is now the subset model)
joblib.dump(best_clf, "income_predictor_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

print("Done. New .pkl files have been created.")