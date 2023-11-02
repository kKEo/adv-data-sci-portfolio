import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# This function displays the splits of the tree
from sklearn.tree import plot_tree

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

df_original = pd.read_csv("../data/Churn_Modelling.csv")

print(df_original['Exited'].value_counts())


avg_churned_bal = df_original[df_original['Exited']==1]['Balance'].mean()

print(df_original.head())

churn_df = df_original.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender'], axis=1)
churn_df = pd.get_dummies(churn_df, drop_first=True)

# Define the y (target) variable
y = churn_df['Exited']

# Define the X (predictor) variables
X = churn_df.copy()
X = X.drop('Exited', axis=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25, stratify=y,
                                                    random_state=42)

decision_tree = DecisionTreeClassifier(random_state=0)

# Fit the model to training data
decision_tree.fit(X_train, y_train)

# Make predictions on test data
dt_pred = decision_tree.predict(X_test)

print("Accuracy:", "%.3f" % accuracy_score(y_test, dt_pred))
print("Precision:", "%.3f" % precision_score(y_test, dt_pred))
print("Recall:", "%.3f" % recall_score(y_test, dt_pred))
print("F1 Score:", "%.3f" % f1_score(y_test, dt_pred))


def conf_matrix_plot(model, x_data, y_data):
    '''
    Accepts as argument model object, X data (test or validate), and y data (test or validate).
    Returns a plot of confusion matrix for predictions on y data.
    '''

    model_pred = model.predict(x_data)
    cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)

    disp.plot(values_format='')  # `values_format=''` suppresses scientific notation
    plt.show()

# conf_matrix_plot(decision_tree, X_test, y_test)


# Plot the tree
plt.figure(figsize=(15,12))
plot_tree(decision_tree, max_depth=2, fontsize=14, feature_names=X.columns,
          class_names={0:'stayed', 1:'churned'}, filled=True);
# plt.show()

# Cross-validated hyperparameter tuning

from sklearn.model_selection import GridSearchCV

# Assign a dictionary of hyperparameters to search over
tree_para = {'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50],
             'min_samples_leaf': [2, 5, 10, 20, 50]}

# Assign a dictionary of scoring metrics to capture
#The 'scoring' parameter of GridSearchCV must be a str among
# {'jaccard', 'neg_median_absolute_error', 'max_error',
#  'roc_auc_ovo_weighted', 'neg_mean_squared_error',
#  'neg_root_mean_squared_error', 'roc_auc', 'f1',
#  'top_k_accuracy', 'rand_score', 'jaccard_micro',
#  'fowlkes_mallows_score', 'matthews_corrcoef',
#  'completeness_score', 'precision_micro',
#  'precision_weighted', 'accuracy', 'roc_auc_ovr',
#  'average_precision', 'neg_mean_poisson_deviance',
#  'mutual_info_score', 'recall_micro', 'r2',
#  'neg_mean_squared_log_error', 'f1_macro',
#  'positive_likelihood_ratio', 'adjusted_mutual_info_score',
#  'roc_auc_ovr_weighted', 'neg_negative_likelihood_ratio',
#  'neg_log_loss', 'recall_weighted', 'homogeneity_score',
#  'f1_weighted', 'neg_brier_score', 'f1_micro', 'precision',
#  'precision_samples', 'recall', 'neg_mean_absolute_percentage_error',
#  'recall_samples', 'explained_variance', 'jaccard_weighted',
#  'roc_auc_ovo', 'v_measure_score', 'jaccard_samples',
#  'recall_macro', 'neg_mean_absolute_error', 'precision_macro',
#  'jaccard_macro', 'balanced_accuracy', 'adjusted_rand_score',
#  'normalized_mutual_info_score', 'neg_mean_gamma_deviance',
#  'f1_samples'}

scoring = ('accuracy', 'precision', 'recall', 'f1')

# Instantiate the classifier
tuned_decision_tree = DecisionTreeClassifier(random_state=42)

# Instantiate the GridSearch
clf = GridSearchCV(tuned_decision_tree,
                   tree_para,
                   scoring=scoring,
                   cv=5,
                   refit="f1")

# Fit the model
print(clf.fit(X_train, y_train))

print(f"Best estimator: {clf.best_estimator_}")


def make_results(model_name, model_object):
    '''
    Accepts as arguments a model name (your choice - string) and
    a fit GridSearchCV model object.

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean F1 score across all validation folds.
    '''

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


print(make_results("Tuned Decision Tree", clf))
