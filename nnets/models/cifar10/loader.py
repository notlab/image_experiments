import os

import tensorflow as tf

from models.common import ImageRecord, ROOT_DIR, CONFIG

GEN_CONFIG = CONFIG['GENERAL']
CIFAR10_CONFIG = CONFIG['CIFAR_10']

def _get_train_files(data_dir):
    '''
    Returns:
        a list containing paths to each of the cifar-10 training data files.
    '''
    filenames = [ os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6) ]
    for f in filenames:
        print(data_dir)
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    return filenames

def _get_eval_files(data_dir):
    '''
    Returns:
        a list containing paths to each of the cifar-10 training data files.
    '''
    filenames = [ os.path.join(data_dir, 'test_batch.bin') ]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    return filenames

def load_cifar10(file_queue, apply_distortions=True):
    height = int(CIFAR10_CONFIG['IMG_HEIGHT'])
    width = int(CIFAR10_CONFIG['IMG_WIDTH'])
    depth = int(CIFAR10_CONFIG['IMG_DEPTH'])
    num_label_bytes = int(CIFAR10_CONFIG['LABEL_BYTES'])

    num_image_bytes = height * width * depth
    num_record_bytes = num_image_bytes + num_label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=num_record_bytes)
    key, value = reader.read(file_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.strided_slice(record_bytes, [0], [num_label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [num_label_bytes], [num_label_bytes + num_image_bytes]),
                             [depth, height, width])
    float32image = tf.transpose(depth_major, [1, 2, 0])

    # Subtract off the mean and divide by the variance of the pixels.
    float32image = tf.image.per_image_standardization(float32image)

    if apply_distortions:
        # Apply a bunch of random distortions to the image for training. 
        # Randomly crop a section of the image.
        float32image = tf.random_crop(float32image, [height, width, 3])

        # Randomly flip the image horizontally.
        float32image = tf.image.random_flip_left_right(float32image)

    float32image.set_shape([height, width, depth])
    label.set_shape([1])
    
    return ImageRecord(height=height, width=width, depth=depth, float32image=float32image, label=label, key=key)

def load_train_batch(apply_distortions=True):
    data_dir = os.path.join(ROOT_DIR, CIFAR10_CONFIG['DATA_DIR'])
    fnames = _get_train_files(data_dir)

    for f in fnames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
  
    file_queue = tf.train.string_input_producer(fnames)
    cifar = load_cifar10(file_queue, apply_distortions)
    batch_size = int(GEN_CONFIG['BATCH_SIZE'])
    min_fraction_images_in_queue = 0.4
    min_queue_examples = int(int(GEN_CONFIG['EPOCH_SIZE_TRAIN']) * min_fraction_images_in_queue)

    print('Filling queue with %d CIFAR images before starting to train. This will take a few minutes.'
          % min_queue_examples)
    
    image_batch, label_batch = tf.train.shuffle_batch([cifar.float32image, cifar.label],
                                                      batch_size=batch_size,
                                                      num_threads=16,
                                                      capacity=min_queue_examples + 3 * batch_size,
                                                      min_after_dequeue=min_queue_examples)

    return image_batch, tf.reshape(label_batch, [batch_size])

def load_eval_batch(apply_distortions=False):
    '''
    Load the cifar10 test data set.
    '''
    data_dir = os.path.join(ROOT_DIR, CIFAR10_CONFIG['DATA_DIR'])
    fnames = _get_eval_files(data_dir)

    for f in fnames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    file_queue = tf.train.string_input_producer(fnames)
    cifar = load_cifar10(file_queue, apply_distortions)
    batch_size = int(GEN_CONFIG['BATCH_SIZE'])
    min_fraction_images_in_queue = 0.4
    min_queue_examples = int(int(GEN_CONFIG['EPOCH_SIZE_TRAIN']) * min_fraction_images_in_queue)

    image_batch, label_batch = tf.train.batch([cifar.float32image, cifar.label],
                                              batch_size=batch_size,
                                              num_threads=16,
                                              capacity=min_queue_examples + 3 * batch_size)

    return image_batch, tf.reshape(label_batch, [batch_size])
