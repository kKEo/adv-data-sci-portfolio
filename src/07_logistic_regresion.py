import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

activity = pd.read_csv("../data/activity.csv")

# Load in sci-kit learn functions for constructing logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Save X and y data into variables
X = activity[["Acc (vertical)"]]
y = activity[["LyingDown"]]

# Split dataset into training and holdout datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression().fit(X_train, y_train)

print("Coefficient: ", clf.coef_)
print("Intercept: ", clf.intercept_)

sns.regplot(x="Acc (vertical)", y="LyingDown", data=activity, logistic=True)

###  Confusion matrix (Part II) ###

# Split data into training and holdout samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build regression model
clf = LogisticRegression().fit(X_train, y_train)

# Save predictions
y_pred = clf.predict(X_test)

print(clf.predict(X_test))


# Print out the predicted probabilities
print(clf.predict_proba(X_test)[::, -1])


import sklearn.metrics as metrics
# Calculate the values for each quadrant in the confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)

# Create the confusion matrix as a visualization
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()

plt.show()