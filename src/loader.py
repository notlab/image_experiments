import os
import configparser

import tensorflow as tf

CONFIG_FILE = './config.ini'

class ImageRecord:

    def __init__(self, height, width=None, depth=3, uint8image=None, label=None, key=None):
        self.height = height
        self.width = width if width else height # use height as width if image is square
        self.depth = depth
        self.uint8image = uint8image
        self.label = label
        self.key = key
    

def read_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config

def _get_cifar_10_files(data_dir):
    """
    Returns:
        a list containing paths to each of the cifar-10 files.
    """
    return [ os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6) ]

def load_cifar_10():
    config = read_config()['CIFAR_10']

    data_dir = config['CIFAR_10_DATA_DIR']
    fnames = _get_cifar_10_files(data_dir)
    file_queue = tf.train.string_input_producer(fnames)
    
    height = int(config['IMG_HEIGHT'])
    width = int(config['IMG_WIDTH'])
    depth = int(config['IMG_DEPTH'])
    num_label_bytes = int(config['LABEL_BYTES'])

    num_image_bytes = height * width * depth
    num_record_bytes = num_image_bytes + num_label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=num_record_bytes)
    key, value = reader.read(file_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.strided_slice(record_bytes, [0], [num_label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [num_label_bytes], [num_label_bytes + num_image_bytes]),
                             [depth, height, width])
    uint8image = tf.transpose(depth_major, [1, 2, 0])
    
    return ImageRecord(height=height, width=width, depth=depth, uint8image=uint8image, label=label, key=key)
