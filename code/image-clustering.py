from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
from kmeans import Kmeans

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
    resnet_model = ResNet50V2(pooling='avg')
    # get the last pooling layer
    final_pooling = resnet_model.get_layer('avg_pool').output
    # make customized model with final_pooling as last layer
    customized_model = Model(inputs=resnet_model.input, outputs=final_pooling)
    # print out the summary of the model
    customized_model.summary()
    # return the feature vectors output
    return customized_model.predict(inputs)

def cluster_images(images):
    """
    Extract the output of the final pooling layer from the ResNet50V2 network with 
    pretrained weights from ImageNet to get a 2048-dimensional feature vector for each
    image in the dataset.

    :param images: Generater/Dataset of 4D input image data with shape (n, 224, 224, 3)
    :return numpy array including the cluster indices for images
    """
    image_batch = next(images)
    extracted_images = np.array([])
    while image_batch is not None:
        extracted_image_batch = extract_resnet_features(images)
        np.append(extracted_images, extracted_image_batch)

    reduction = PCA(n_components=64)
    reduced_images = reduction.fit_transform(extracted_images)

    classifier = Kmeans(num_clusters=16)
    classifier.train(reduced_images)
    image_clusters = classifier.predict(reduced_images)

    return image_clusters
