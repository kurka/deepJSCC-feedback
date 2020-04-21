import os
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf

_HEIGHT = 512
_WIDTH = 768
_NUM_CHANNELS = 3
_NUM_IMAGES = {
    'train': 24,
    'validation': 24,
    'test': 24,
}

SHUFFLE_BUFFER = _NUM_IMAGES['train']
SHAPE = [_HEIGHT, _WIDTH, _NUM_CHANNELS]


def get_dataset(is_training, data_dir):
    """Returns a dataset object"""
    maybe_download_and_extract(data_dir)

    file_pattern = os.path.join(data_dir, "kodim*.png")
    filename_dataset = tf.data.Dataset.list_files(file_pattern)
    return filename_dataset.map(lambda x: tf.image.decode_png(tf.read_file(x)))


def parse_record(raw_record, _mode, dtype):
    """Parse CIFAR-10 image and label from a raw record."""
    image = tf.reshape(raw_record, [_HEIGHT, _WIDTH, _NUM_CHANNELS])
    # normalise images to range 0-1
    image = tf.cast(image, dtype)
    image = tf.divide(image, 255.0)


    return image, image


def preprocess_image(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, _HEIGHT + 8, _WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image


def maybe_download_and_extract(data_dir):
    """Download and extract the tarball from Alex's website."""
    if os.path.exists(data_dir):
        return
    else:
        os.makedirs(data_dir)

        filepath = data_dir

        url = "http://www.cs.albany.edu/~xypan/research/img/Kodak/kodim{}.png"
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filepath, 100.0 * count * block_size / total_size))
            sys.stdout.flush()

        for i in range(25):
            print(url.format(i+1))
            filepath, _ = urllib.request.urlretrieve(url.format(i+1), filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filepath, statinfo.st_size, 'bytes.')
