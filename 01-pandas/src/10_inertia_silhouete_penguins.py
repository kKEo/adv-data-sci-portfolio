# Import standard operational packages.
import numpy as np
import pandas as pd

# Important tools for modeling and evaluation.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Import visualization packages.
import matplotlib.pyplot as plt
import seaborn as sns

penguins = pd.read_csv("../data//penguins.csv")

print(penguins['species'].unique())
print(penguins['species'].value_counts(dropna=False))

# Check for missing values.
print(penguins.isnull().sum())

# Drop rows with missing values
penguins_subset = penguins.dropna(axis=0).reset_index(drop = True)
print(penguins_subset.isna().sum())

# Encode data
penguins_subset['sex'] = penguins_subset['sex'].str.upper()
penguins_subset = pd.get_dummies(penguins_subset, drop_first=True, columns=['sex'])

# Get rid of location
penguins_subset = penguins_subset.drop(['island'], axis=1)
X = penguins_subset.drop(['species'], axis=1)

#Scale the features.
#Assign the scaled data to variable `X_scaled`.
X_scaled = StandardScaler().fit_transform(X)
# Fit K-means and evaluate inertia for different values of k.

num_clusters = [i for i in range(2, 11)]

def kmeans_inertia(num_clusters, x_vals):
    """
    Accepts as arguments list of ints and data array.
    Fits a KMeans model where k = each value in the list of ints.
    Returns each k-value's inertia appended to a list.
    """
    inertia = []
    for num in num_clusters:
        kms = KMeans(n_clusters=num, n_init=10, random_state=42)
        kms.fit(x_vals)
        inertia.append(kms.inertia_)
    return inertia

inertia = kmeans_inertia(num_clusters, X_scaled)
print(f"Inertia: {inertia}")


plot = sns.lineplot(x=num_clusters, y=inertia, marker='o')
plot.set_xlabel("Number of clusters")
plot.set_ylabel("Inertia")
plt.show()

# Evaluate silhouette score.
# Write a function to return a list of each k-value's score.

### YOUR CODE HERE ###

def kmeans_sil(num_clusters, x_vals):
    """
    Accepts as arguments list of ints and data array.
    Fits a KMeans model where k = each value in the list of ints.
    Calculates a silhouette score for each k value.
    Returns each k-value's silhouette score appended to a list.
    """
    sil_score = []
    for num in num_clusters:
        kms = KMeans(n_clusters=num, n_init=10, random_state=42)
        kms.fit(x_vals)
        sil_score.append(silhouette_score(x_vals, kms.labels_))

    return sil_score


sil_score = kmeans_sil(num_clusters, X_scaled)
print(f"Silhouette score: {sil_score}")

plot = sns.lineplot(x=num_clusters, y=sil_score, marker='o')
plot.set_xlabel("# of clusters")
plot.set_ylabel("Silhouette Score")
# plt.show()


# Optimal k-value

kmeans6 = KMeans(n_clusters=6, n_init=10, random_state=42)
kmeans6.fit(X_scaled)

print('Unique labels:', np.unique(kmeans6.labels_))
penguins_subset['cluster'] = kmeans6.labels_
penguins_subset.head()


penguins_subset.groupby(by=['cluster', 'species']).size()
penguins_subset.groupby(by=['cluster', 'species']).size().plot.bar(title='Clusters differentiated by species',
                                                                   figsize=(6, 5),
                                                                   ylabel='Size',
                                                                   xlabel='(Cluster, Species)');

plt.show()

# Verify if each `cluster` can be differentiated by `species` AND `sex_MALE`.
penguins_subset.groupby(by=['cluster', 'species', 'sex_MALE']).size().sort_values(ascending=False)

penguins_subset\
    .groupby(by=['cluster', 'species', 'sex_MALE']).size()\
    .unstack(level='species', fill_value=0)\
    .plot.bar(title='Clusters differentiated by species and sex',
              figsize=(6, 5),
              ylabel='Size',
              xlabel='(Cluster, Sex)')

plt.legend(bbox_to_anchor=(1.3, 1.0))
plt.show()

