import os
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5
_NUM_IMAGES = {
    'train': 45000,
    'validation': 5000,
    'test': 10000,
}

SHUFFLE_BUFFER = _NUM_IMAGES['train']
SHAPE = [_HEIGHT, _WIDTH, _NUM_CHANNELS]


def get_dataset(is_training, data_dir):
    """Returns a dataset object"""
    filenames = get_filenames(is_training, data_dir)
    return tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)


def get_filenames(is_training, data_dir):
    """Returns a list of filenames."""
    maybe_download_and_extract(data_dir)

    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    if is_training:
        return [
            os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in range(1, _NUM_DATA_FILES + 1)
        ]
    else:
        return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record, _mode, dtype):
    """Parse CIFAR-10 image and label from a raw record."""
    # Convert bytes to a vector of uint8 that is record_bytes long.
    record_vector = tf.io.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot.
    label = tf.cast(record_vector[0], tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                             [_NUM_CHANNELS, _HEIGHT, _WIDTH])

    # Convert from [depth, height, width] to [height, width, depth], and cast
    # as float32.
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    # normalise images to range 0-1
    image = image/255.0

    image = tf.cast(image, dtype)

    return image, image


def maybe_download_and_extract(data_dir):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, 100.0 * count * block_size / total_size))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(data_dir, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(data_dir)
