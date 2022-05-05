import numpy as np
import random


class Kmeans(object):
    """
    K-Means Classifier via Iterative Improvement

    @attrs:
        k: The number of clusters to form as well as the number of centroids to
           generate (default = 10), an int
        tol: Value specifying our convergence criterion. If the ratio of the
             distance each centroid moves to the previous position of the centroid
             is less than this value, then we declare convergence.
        max_iter: the maximum number of times the algorithm can iterate trying
                  to optimize the centroid values, an int,
                  the default value is set to 500 iterations
        centroids: a Numpy array where each element is one of the k cluster centers
    """

    def __init__(self, num_clusters = 16, max_iter = 1000, threshold = 1e-6):
        """
        Initiate K-Means with some parameters
        """
        self.k = num_clusters
        self.tol = threshold
        self.max_iter = max_iter
        self.centroids = None

    def train(self, X):
        """
        Compute K-Means clustering on each class label and store your result in self.cluster_centers_
        :param X: inputs of training data, a 2D Numpy array
        """
        curr_centroids = np.array(random.sample(X.tolist(), self.k))
        prev_centroids = curr_centroids
        iter = 0
        while(iter < self.max_iter):
            # assigning cluster for each images
            assigned_data = np.array([np.argmin(np.sum(np.square(curr_centroids-x), axis=1)) for x in X])
            # 
            curr_centroids = np.array([np.average(X[np.where(assigned_data==i)],axis=0) for i in range(self.k)])
            if (np.linalg.norm(curr_centroids-prev_centroids)/np.linalg.norm(prev_centroids) < self.tol):
                break
            prev_centroids = curr_centroids
            iter += 1
            print(f"kmeans iteration number {iter} done")
        self.centroids = curr_centroids

    def predict(self, X):
        """
        Predicts the label of each sample in X based on the assigned centroids.

        :param X: A dataset as a 2D Numpy array
        :return: A Numpy array of predicted clusters
        """
        return [np.argmin(np.sum(np.square(self.centroids-x), axis=1)) for x in X]
