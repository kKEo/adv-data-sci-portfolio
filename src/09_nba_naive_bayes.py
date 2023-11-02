import pandas as pd

data = pd.read_csv("../data/nba-players.csv", index_col=0)


print(data["target_5yrs"].value_counts(normalize=True)*100)

selected_data = data[["gp", "min", "pts", "fg", "3p", "ft", "reb", "ast", "stl", "blk", "tov", "target_5yrs"]]
print(selected_data.head())

extracted_data = selected_data.copy()

# Add a new column named `total_points`;
# Calculate total points earned by multiplying the number of games played by the average number of points earned per game
extracted_data["total_points"] = extracted_data["gp"] * extracted_data["pts"]

# Add a new column named `efficiency`. Calculate efficiency by dividing the total points earned by the total number
# of minutes played, which yields points per minute. (Note that `min` represents avg. minutes per game.)
extracted_data["efficiency"] = extracted_data["total_points"] / (extracted_data["min"] * extracted_data["gp"])

# Display the first few rows of `extracted_data` to confirm that the new columns were added.
print(extracted_data.head())

# Remove `gp`, `pts`, and `min` from `extracted_data`.
extracted_data = extracted_data.drop(columns=["gp", "pts", "min"])


### Bayes

# Define the y (target) variable.
y = extracted_data['target_5yrs']

# Define the X (predictor) variables.
X = extracted_data.drop('target_5yrs', axis=1)

from sklearn import model_selection
from sklearn import metrics
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)


# Assign `nb` to be the appropriate implementation of Naive Bayes.
from sklearn import naive_bayes
nb = naive_bayes.GaussianNB()

# Fit the model on your training data.
nb.fit(X_train, y_train)

# Apply your model to predict on your test data. Call this "y_pred".
y_pred = nb.predict(X_test)

print('Accuracy score:'), print(metrics.accuracy_score(y_test, y_pred))
print('Precision score:'), print(metrics.precision_score(y_test, y_pred))
print('Recall score:'), print(metrics.recall_score(y_test, y_pred))
print('F1 score:'), print(metrics.f1_score(y_test, y_pred))


# Construct and display your confusion matrix.
# Construct the confusion matrix for your predicted and test values.
cm = metrics.confusion_matrix(y_test, y_pred)

# Create the display for your confusion matrix.
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb.classes_)

# Plot the visual in-line.
disp.plot()


import matplotlib.pyplot as plt
plt.show()