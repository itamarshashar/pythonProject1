import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)


def load_data(path):
    """
        :parameter path: path to the data file
        :return: the pandas DataFrame
    """
    df = pd.read_csv(path)
    return df

"""""def scaling(transformed_data):
    transformed_data[:, 0] = (
                (transformed_data[:, 0] - (transformed_data[:, 0]).min()) / (transformed_data[:, 0]).sum())
    transformed_data[:, 1] = (
                (transformed_data[:, 1] - (transformed_data[:, 1]).min()) / (transformed_data[:, 1]).sum())
"""


def transform_data(df, features):
    """
        Performs the following transformations on df:
            - selecting relevant features
            - scaling
            - adding noise
        :param df: dataframe as was read from the original csv.
        :param features: list of 2 features from the dataframe
        :return: transformed data as numpy array of shape (n, 2)
        """

    transformed_data = df[features].to_numpy().T
    transformed_data[0] = (transformed_data[0] - transformed_data[0].min()) / transformed_data[0].sum()
    transformed_data[1] = (transformed_data[1] - transformed_data[1].min()) / transformed_data[1].sum()

    return add_noise(transformed_data.T)  # לבדוק אם זאת הכוונה


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.01^2)
    """
    noise = np.random.normal(loc=0, scale=0.01, size=data.shape)
    return data + noise


#########################################################################

def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset

    """""
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """

    labels = np.arange(data.shape[0])
    for i in range(data.shape[0]):
        distances = [np.linalg.norm(data[i] - centroid) for centroid in centroids]
        labels[i] = np.argmin(distances)
    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        centroids[i] = np.average(data[labels == i], axis=0)
    return centroids



def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """


    centroids = choose_initial_centroids(data, k)
    old_centroids = np.zeros((k, data.shape[1]))
    labels = assign_to_clusters(data, centroids)

    while not (np.array_equal(centroids, old_centroids)):
        labels = assign_to_clusters(data, centroids)
        old_centroids = centroids.copy()
        centroids = recompute_centroids(data, labels, k)

    return labels, centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """

    u_labels = np.unique(labels)
    plt.figure(figsize=(8, 8))
    for i in u_labels:
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=int(i), s=40, linewidths=1, alpha=1)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', edgecolors='black', marker='*',
                linewidths=1, label=f'Centroid')
    plt.xlabel("cnt")
    plt.ylabel("hum")
    plt.title(f'Results for k-means with k = {centroids.shape[0]}', loc='center', fontsize=20)
    plt.legend()
    #plt.show()
    plt.savefig(path.format(len(centroids)))
