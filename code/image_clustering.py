from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
import h5py
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
    #customized_model.summary()
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
    # Apply ResNet feature extraction to each batch
    image_batch = next(images)
    extracted_images = np.array([])
    count = 0
    while image_batch is not None:
        print(type(image_batch))
        print(image_batch.shape)
        extracted_image_batch = extract_resnet_features(image_batch)
        np.append(extracted_images, extracted_image_batch)
        count += 128
        print(f"extracted {count}")
        image_batch = next(images)
    print("extracted all")
    # Fit then apply pply sklearn PCA reduction to extracted images
    reduction = PCA(n_components=64)
    reduced_images = reduction.fit_transform(extracted_images)
    print("reduced all")    

    # Fit then make clusters for reduced images using kmeans
    classifier = Kmeans(num_clusters=16)
    classifier.train(reduced_images)
    print("done kmeans trainning")
    image_clusters = classifier.predict(reduced_images)

    # Return the clusters with same indices as image data
    return image_clusters

def makeImageGenerator(data_file_path, batch_size, max_epoch=10):
    hdf5_file = h5py.File(data_file_path, 'r')['data'][:1000]
    epoch = 0
    while epoch < max_epoch:
        epoch += 1
        for i in range(0, len(hdf5_file), batch_size):
            if i + batch_size >= len(hdf5_file):
                break
            # transpose from NCHW to NHWC
            images = tf.cast(tf.transpose(hdf5_file[i:i+batch_size], [0, 2, 3, 1]), tf.int32)
            images = tf.image.resize(images, [224, 224])
            images = tf.cast(images, tf.float32) / 255
            yield images
            print(i)
        print("e done")

gen = makeImageGenerator('LLD-logo.hdf5', 128, 1)
print("generator done")
clusters = cluster_images(gen)
print(clusters[:100])