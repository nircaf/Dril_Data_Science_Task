from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # for plot styling
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
import pandas as pd
from matplotlib import pyplot
import time

def normalize(data):
    result = data.copy()
    for feature_name in data.columns:
        max_value = data[feature_name].max()
        min_value = data[feature_name].min()
        result[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
    return result.dropna(axis = 1, how='any') # remove unmatching col

def excels_to_num(data):  # factorize data
    for column in data:
        if not pd.to_numeric(data[column], errors='coerce').notnull().all():
            data[column] = pd.factorize(data[column], sort=False)[0]
    return normalize(data)

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

def age_divide(age_series):
    for index,age in enumerate(age_series):
        if age < 20:
            age_series[index] = 0
        elif 40> age > 20:
            age_series[index] = 1
        elif 60> age > 40:
            age_series[index] = 2
        elif 80> age > 60:
            age_series[index] = 3
        else:
            age_series[index] = 4

def kmeans_run(engagement_percent_mean_loc):
    X = (pd.DataFrame(engagement_percent_mean_loc).drop(['date','gender','delta_location_engage'], axis=1).fillna(0))
    col_names  = X.columns
    X = excels_to_num(X).to_numpy()

    n_clusters =6
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    kmedoids = KMedoids(n_clusters=n_clusters, init='build', max_iter=500, random_state=0).fit(X)
    y_kmeans = kmedoids.labels_
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y_kmeans, s=50, cmap='viridis' )
    centers = kmedoids.cluster_centers_
    ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5)

    ax.set_xlabel(col_names[0])
    ax.set_ylabel(col_names[1])
    ax.set_zlabel(col_names[2])
    fig.savefig("Figures/kmeans_run.png", dpi=300)
    time.sleep(3) # Wait to finish save
    pyplot.close(fig)