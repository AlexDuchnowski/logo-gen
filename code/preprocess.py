import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import googletrans 

WINDOW_SIZE = 30

def make_input_generator(data_file_path, batch_size, epochs):
    """
    Returns a generator on batches of data.

    :param data_file_path: file path to the hdf5 file containing the data
    :param batch_size: batch size for splitting up the model inputs
    :return: the final generator yields (images, descriptions, names)
    """
    hdf5_file = h5py.File(data_file_path, 'r')
    for n in range(epochs):
        for i in range(0, len(hdf5_file['data']), batch_size):
            if i + batch_size >= len(hdf5_file['data']):
                break

            # transpose from NCHW to NHWC
            images = tf.cast(tf.transpose(hdf5_file['data'][i:i+batch_size], [0, 2, 3, 1]), tf.int32)
            users = [json.loads(user_json) for user_json in hdf5_file['meta_data']['twitter']['user_objects'][i:i+batch_size]]
            descriptions = [user['description'] for user in users]
            names = [user['name'] for user in users]

            images = process_images(images)
            descriptions = process_descriptions(descriptions)
            names = process_names(names)

            # print(images)
            yield images, descriptions, names

def process_images(images):
    # resizing and normalization of pixel values for images
    images = tf.image.resize(images, [64, 64])
    images = tf.cast(images, tf.float32) / 255
    return images

def process_descriptions(descriptions):
    # create vocab for descriptions and replace words with corresponding ids (maybe UNK some words)
    # REPLACE WITH REAL PREPROCESSING
    translator = googletrans.Translator()
    padded_descriptions = []
    for desc in descriptions:
        # TRANSLATE HERE - ignore this placeholder stuff
        lengths = translator.translate(desc).text.lower().split()
        if len(lengths) < WINDOW_SIZE:
            lengths.extend(['0' for _ in range(WINDOW_SIZE - len(lengths))])
        else:
            lengths = lengths[:WINDOW_SIZE]
        padded_descriptions.append(lengths)
    print(len(padded_descriptions))
        #Can't convert Python sequence with mixed types to Tensor
    return tf.convert_to_tensor(padded_descriptions)

def process_names(names):
    # convert company names to ASCII
    padded_names = []
    for name in names:
        ascii = list([ord(char) / 255 if char.isascii() else 0 for char in name])
        if len(ascii) < WINDOW_SIZE:
            ascii.extend([0 for _ in range(WINDOW_SIZE - len(ascii))])
        else:
            ascii = ascii[:WINDOW_SIZE]
        padded_names.append(ascii)
    return tf.convert_to_tensor(padded_names)

gen = make_input_generator('LLD-logo.hdf5', 128,1)
images, descriptions, names = next(gen)
#print(images)
print(descriptions)
#print(names)

# plt.imshow(images[0])
# plt.savefig('img.png')
