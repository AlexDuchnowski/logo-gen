import numpy as np
import h5py

# simple way to load the complete dataset
i = 0
# open hdf5 file
hdf5_file = h5py.File('LLD-logo.hdf5', 'r')
# get original logo dimensions
shape = hdf5_file['shapes'][i]
# remove zero padding by only indexing the original shape
image = hdf5_file['data'][i, :, :shape[1], :shape[2]]
# get corresponding label (if needed)
image_label = hdf5_file['labels/resnet/rc_64'][i]


# more advanced method: creating a data generator which re-shuffles the data for each epoch (useful for training models)

def make_generator(file_path, batch_size, label_name=None):
    hdf5_file = h5py.File(file_path, 'r')
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 400, 400), dtype='int32')
        labels = np.zeros(batch_size, dtype='int32')
        indices = range(len(hdf5_file['data']))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(indices)
        epoch_count[0] += 1
        for n, i in enumerate(indices):
            shape = hdf5_file['shapes'][i]
            # remove zero padding by using proper array indexing
            images[n % batch_size] = hdf5_file['data'][i][:, :shape[1], :shape[2]]
            if label_name is not None:
                labels[n % batch_size] = hdf5_file[label_name][i]
            if n > 0 and n % batch_size == 0:
                yield (images, labels)
    return get_epoch

data_generator = make_generator(file_path='twitter_dataset_clean.hdf5', batch_size=64, label_name='labels/resnet/rc_64')
image_batch = data_generator.next()
single_image = image_batch[0]