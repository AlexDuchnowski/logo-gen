from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
import random
import heapq as hq
import h5py

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

    def __init__(self, num_clusters = 16, max_iter = 2000, threshold = 1e-8):
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
        m = 64
        closest_data = []
        for centroid in self.centroids:
            pq=[]
            for i in range(len(X)):
                hq.heappush(pq, (np.sum(np.square(centroid-X[i])), i))
            temp_data = []
            for i in range(m):
                temp_data.append(hq.heappop(pq)[1])
            closest_data.append(temp_data)
        np.savetxt(f"icon-clusters/{self.k}_best_clusters_data.csv", closest_data, delimiter=",")
        print(f"save best clusters image for {self.k}")

    def predict(self, X):
        """
        Predicts the label of each sample in X based on the assigned centroids.

        :param X: A dataset as a 2D Numpy array
        :return: A Numpy array of predicted clusters
        """
        return np.array([np.argmin(np.sum(np.square(self.centroids-x), axis=1)) for x in X])

def extract_resnet_features(images):
    """
    Extract the output of the final pooling layer from the ResNet50V2 network with 
    pretrained weights from ImageNet to get a 2048-dimensional feature vector for each
    image in the dataset.

    :param images: 4D input image data with shape (n, 224, 224, 3)
    :return 2D features vector output as numpy array with shape (n, 2048)
    """
    # convert RGB to BGR and normalize each channel to fit the ImageNet dataset
    inputs = preprocess_input(images)
    # initialize Keras model for ResNet50V2 with pretrained model from ImgaeNet
    resnet_model = ResNet50V2(include_top=False, input_shape=(32,32,3), pooling='avg')
    # get the last pooling layer
    final_pooling = resnet_model.get_layer('avg_pool').output
    # make customized model with final_pooling as last layer
    customized_model = Model(inputs=resnet_model.input, outputs=final_pooling)
    # print out the summary of the model
    #customized_model.summary()
    # return the feature vectors output
    return customized_model.predict(inputs)

def makeImageGenerator(data_file_path, batch_size, max_epoch=10):
    hdf5_file = h5py.File(data_file_path, 'r')['data']
    epoch = 0
    while epoch < max_epoch:
        epoch += 1
        for i in range(0, len(hdf5_file), batch_size):
            end_idx = i+batch_size
            if i + batch_size >= len(hdf5_file):
                batch_size = len(hdf5_file)
            # transpose from NCHW to NHWC
            images = tf.cast(tf.transpose(hdf5_file[i:end_idx], [0, 2, 3, 1]), tf.int32)
            #images = tf.image.resize(images, [224, 224])
            images = tf.cast(images, tf.float32) / 255
            yield images

def cluster_images(images, num_clusters=16):
    """
    Extract the output of the final pooling layer from the ResNet50V2 network with 
    pretrained weights from ImageNet to get a 2048-dimensional feature vector for each
    image in the dataset.

    :param images: Generater/Dataset of 4D input image data with shape (n, 224, 224, 3)
    :param num_clusters: Number of clusters to classify into
    :return numpy array including the cluster indices for images
    """
    # Apply ResNet feature extraction to each batch
    image_batch = next(images)
    extracted_images = None
    count = 0
    while True:
        extracted_image_batch = extract_resnet_features(image_batch)
        count += extracted_image_batch.shape[0]
        if extracted_images is None:
            extracted_images = extracted_image_batch
        else:
            extracted_images = np.append(extracted_images, extracted_image_batch, axis=0)
        print(f"extracted {count}")
        try:
            image_batch = next(images)
        except StopIteration as e:
            break

    print(f"ResNet extraction done with shape {extracted_images.shape}")
    # Fit then apply pply sklearn PCA reduction to extracted images
    reduction = PCA(n_components=256)
    reduced_images = reduction.fit_transform(extracted_images)
    print(f"PCA reduction done with shape {reduced_images.shape}")    

    # Fit then make clusters for reduced images using kmeans
    image_clusters = []
    for num_cluster in num_clusters:
        classifier = Kmeans(num_clusters=num_cluster)
        classifier.train(reduced_images)
        print(f"done kmeans trainning with {num_cluster} clusters")
        image_clusters.append(np.reshape(classifier.predict(reduced_images), (-1,1)))
    # Return the clusters with same indices as image data
    return np.array(image_clusters)

data_file_path = 'LLD-icon-data.hdf5'
gen = makeImageGenerator(data_file_path, 512, 1)
n_clusters = [4,6,8,10,12,16,24,32,64]
clusters = cluster_images(gen, num_clusters=n_clusters)
for i in range(len(n_clusters)):
    n = n_clusters[i]
    np.savetxt(f"icon-clusters/{n}_clusters_data.csv", clusters[i], delimiter=",")