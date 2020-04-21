import os
import tensorflow as tf
import data.imagenet_preprocessing as imgnet_preprocessing

_DEFAULT_IMAGE_SIZE = 128#224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

# total 'train' images: 1281167
_NUM_IMAGES = {
    'train': 1231167,
    'validation': 49920,  # I thought it was 50k, but its just finding 49k
    'test': 50000,
}

_NUM_TRAIN_FILES = 1024  # number of tfrecords files
SHUFFLE_BUFFER = 10000
SHAPE = [_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS]


def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, 'train', 'train-%05d-of-01024' % i)
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [
            os.path.join(data_dir, 'validation', 'validation-%05d-of-00128' % i)
            for i in range(128)]


def _parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a
    dataset containing serialized Example protocol buffers. Each Example proto
    contains the following fields (values are included as examples):

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox


def parse_record(raw_record, _mode, dtype):
    """Parses a record containing a training example of an image.

    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).

    Args:
      raw_record: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
      is_training: A boolean denoting whether the input is for training.
      dtype: data type to use for images/features.

    Returns:
      Tuple with processed image tensor and one-hot-encoded label tensor.
    """
    image_buffer, label, bbox = _parse_example_proto(raw_record)

    image = imgnet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        num_channels=_NUM_CHANNELS,
        is_training=False)  # as we are not classifying, do minimal processing
    image = tf.cast(image, dtype)

    return image, image


def get_dataset(is_training, data_dir):
    """Returns a dataset object

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
    Returns:
      A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)


    # if is_training:
    #     # Shuffle the input files
    #     dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    # Convert to individual records.
    # cycle_length = 20 means 20 files will be read and deserialized in
    # parallel. This number is low enough to not cause too much contention on
    # small systems but high enough to provide the benefits of parallelization.
    # You may want to increase this number if you have a large number of CPU
    # cores.
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=20))
    return dataset
