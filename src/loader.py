import os
import configparser

import tensorflow as tf

CONFIG_FILE = './config.ini'

CONFIG = read_config()
GEN_CONFIG = CONFIG['GENERAL']
CIFAR10_CONFIG = CONFIG['CIFAR_10']


def _get_cifar10_files(data_dir):
    """
    Returns:
        a list containing paths to each of the cifar-10 files.
    """
    return [ os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6) ]

def load_single_cifar10(apply_distortions=True):
    data_dir = CIFAR10_CONFIG['CIFAR_10_DATA_DIR']
    fnames = _get_cifar10_files(data_dir)
    file_queue = tf.train.string_input_producer(fnames)
    
    height = int(CIFAR10_CONFIG['IMG_HEIGHT'])
    width = int(CIFAR10_CONFIG['IMG_WIDTH'])
    depth = int(CIFAR10_CONFIG['IMG_DEPTH'])
    num_label_bytes = int(CIFAR10_CONFIG['LABEL_BYTES'])

    num_image_bytes = height * width * depth
    num_record_bytes = num_image_bytes + num_label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=num_record_bytes)
    key, value = reader.read(file_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.strided_slice(record_bytes, [0], [num_label_bytes]), tf.float32)

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [num_label_bytes], [num_label_bytes + num_image_bytes]),
                             [depth, height, width])
    float32image = tf.transpose(depth_major, [1, 2, 0])

    if apply_distortions:
        # Apply a bunch of random distortions to the image for training. 
        # Randomly crop a section of the image.
        float32image = tf.random_crop(float32image, [height, width, 3])

        # Randomly flip the image horizontally.
        float32image = tf.image.random_flip_left_right(float32image)
        
        # Subtract off the mean and divide by the variance of the pixels.
        float32image = tf.image.per_image_standardization(float32image)
    
    return ImageRecord(height=height, width=width, depth=depth, float32image=float32image, label=label, key=key)

def load_batch_cifar10(apply_distortions=True):
    single_cifar = load_single_cifar_10(apply_distortions)
    batch_size = GEN_CONFIG['BATCH_SIZE']
    min_fraction_images_in_queue = 0.4
    min_queue_examples = int(GEN_CONFIG['EPOCH_SIZE_TRAIN'] * min_fraction_images_in_queue)

     print('Filling queue with %d CIFAR images before starting to train. This will take a few minutes.'
           % min_queue_examples)
    
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size
                                                      num_threads=16,
                                                      capacity=min_queue_examples + 3 * batch_size,
                                                      min_after_dequeue=min_queue_examples)

    return image_batch, tf.reshape(label_batch, [batch_size])

def read_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config


class ImageRecord:

    def __init__(self, height, width=None, depth=3, float32image=None, label=None, key=None):
        self.height = height
        self.width = width if width else height # use height as width if image is square
        self.depth = depth
        self.float32image = float32image
        self.label = label
        self.key = key
    
