import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import json


def data_generator(data_file_path, batch_size):
    hdf5_file = h5py.File(data_file_path, 'r')

def get_data(filepath):
    # return a generator that can be iterated through to get subsequent batches
    # NHWC transpose, resizing, and normalization of pixel values for images
    # create vocab for descriptions and replace words with corresponding ids (maybe UNK some words)
    # convert company names to ASCII

    # metadata = unpickle(metadata_file_path)
    hdf5_file = h5py.File(filepath, 'r')
    # NCHW -> NHWC
    # print(hdf5_file['meta_data']['names'][:5])
    # print(hdf5_file['meta_data']['twitter']['ids'][:5])
    # users = hdf5_file['meta_data']['twitter']['user_objects'][:5]
    # print([json.loads(user)['description'] for user in users])
    # print([json.loads(user)['name'] for user in users])
    print(len(hdf5_file['data']))
    # images = tf.transpose(hdf5_file['data'][:5], [0, 2, 3, 1])
    # descriptions = hdf5_file['meta_data/user_object']['user_object'][:5]
    # print(descriptions)
    # # descriptions = tf.convert_to_tensor([vars(user["user_object"])["description"] for user in metadata.values()])[:5]
    # # names = tf.convert_to_tensor([vars(user["user_object"])["name"] for user in metadata.values()])[:5]
    # return images, descriptions #, names

get_data('LLD-logo.hdf5')
# print(images.shape)
# print(descriptions.shape)
# print(names.shape)

# # code adapted from: https://data.vision.ee.ethz.ch/sagea/lld/

# def load_icon_data(data_path, pattern='LLD*.pkl', single_file=None):
#     """Loads icon data from pickle files in directory specified in data_path
#         pattern:        file name pattern to search for ['LLD_favicon_data*.pkl']
#         single_file:    If not None, only the file with the specified number (modulo total number of files) 
#                         is loaded [None]
#         Returns the numpy arrays of shape (num_icons, 32, 32, 3) of dtype uint8

# 	Example use with PIL:
# 	icons = load_icon_data(data_path)
# 	img = PIL.Image.fromarray(icons[0])
# 	img.show()"""
#     files = glob(os.path.join(data_path, pattern))
#     files.sort()
#     if single_file is None:
#         with open(files[0], 'rb') as f:
#             # used latin1 encoding to resolve unpickling error
#             # https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
#             icons = cPickle.load(f, encoding='latin1')
#         if len(files) > 1:
#             for file in files[1:]:
#                 with open(file, 'rb') as f:
#                     icons_loaded = cPickle.load(f, encoding='latin1')
#                 icons = np.concatenate((icons, icons_loaded))
#     else:
#         with open(files[single_file % len(files)], 'rb') as f:
#             icons = cPickle.load(f, encoding='latin1')
#     return icons

# def save_icon_data(icons, data_path, package_size=100000):
#     if not os.path.exists(data_path):
#         os.makedirs(data_path)
#     num_packages = int(math.ceil(len(icons) / package_size))
#     num_len = len(str(num_packages))
#     for p in range(num_packages):
#         with open(os.path.join(data_path, 'LLD-icon_data_' + str(p).zfill(num_len) + '.pkl'), 'wb') as f:
#             cPickle.dump(icons[p*package_size:(p+1)*package_size], f, protocol=cPickle.HIGHEST_PROTOCOL)

# icons = load_icon_data('LLD-icon', single_file=1)
# for i in range(20):
#     plt.imshow(icons[i])
#     plt.show()
# # print(np.shape(icons))
