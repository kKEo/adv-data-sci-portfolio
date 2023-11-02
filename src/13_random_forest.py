import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# This lets us see all of the columns, preventing Juptyer from redacting them.
pd.set_option('display.max_columns', None)

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier

# This module lets us save our models once we fit them.
import pickle

df_original = pd.read_csv("../data/Churn_Modelling.csv")


# Feature engineering
churn_df = df_original.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender'], axis=1)
with_original_columns = {col for col in churn_df.columns}

# Dummy encode categoricals
churn_df2 = pd.get_dummies(churn_df, drop_first='True')
with_categorial_columns = {column for column in churn_df2.columns}

print("New columns: ", set.difference(with_categorial_columns, with_original_columns))
print("Removed: ", set.difference(with_original_columns, with_categorial_columns))

# Define the y (target) variable
y = churn_df2["Exited"]

# Define the X (predictor) variables
X = churn_df2.copy()
X = X.drop("Exited", axis=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


## Cross-validated hyperparameter tuning
rf = RandomForestClassifier(random_state=0)

cv_params = {'max_depth': [2, 3, 4, 5, None],
             'min_samples_leaf': [1, 2, 3],
             'min_samples_split': [2, 3, 4],
             'max_features': [2, 3, 4],
             'n_estimators': [75, 100, 125, 150]
             }

scoring = ('accuracy', 'precision', 'recall', 'f1')

rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='f1')

try:
    with open('../output/rf_cv_model.pickle', 'rb') as model:
        rf_cv = pickle.load(model)
except Exception as e:
    print("Exception e: ", e)
    print("Fitting model..")
    rf_cv.fit(X_train, y_train)
    with open('../output/rf_cv_model.pickle', 'wb') as to_write:
        pickle.dump(rf_cv, to_write)

print("Best params: ", rf_cv.best_params_)
print("Best score: ", rf_cv.best_score_)


def make_results(model_name, model_object):
    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(mean f1 score)
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame({'Model': [model_name],
                          'F1': [f1],
                          'Recall': [recall],
                          'Precision': [precision],
                          'Accuracy': [accuracy]
                          }
                         )

    return table


rf_cv_results = make_results('Random Forest CV', rf_cv)
print(rf_cv_results)

###############################################################
# PART 2. Hyperparameters tuned with separate validation set

# Create separate validation data
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                            stratify=y_train, random_state=10)

# Create list of split indices
split_index = [0 if x in X_val.index else -1 for x in X_train.index]

from sklearn.model_selection import PredefinedSplit

rf = RandomForestClassifier(random_state=0)

cv_params = {'max_depth': [2,3,4,5, None],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'max_features': [2,3,4],
             'n_estimators': [75, 100, 125, 150]
             }

scoring = ('accuracy', 'precision', 'recall', 'f1')

custom_split = PredefinedSplit(split_index)

rf_val = GridSearchCV(rf, cv_params, scoring=scoring, cv=custom_split, refit='f1')

rf_val.fit(X_train, y_train)

print("Best params: ", rf_val.best_params_)
rf_cv_results = make_results('Random Forest CV Validated', rf_val)
print(rf_cv_results)
