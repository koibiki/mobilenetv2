import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize, imsave


def preprocess(image):
    # subtract mean
    mean = np.array([123.68, 116.779, 103.939])
    image = image - mean
    # scale to 1
    img = image * 0.017
    # return value should be float!
    return img


# tfrecord example features
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_tfrecord_dataset(example_proto):
    input_shape = [224, 224, 3]
    features = {'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/label': tf.FixedLenFeature([], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.decode_raw(parsed_features['image/encoded'], tf.uint8)
    h, w, c = input_shape
    image = tf.reshape(image, [h, w, c])

    label = parsed_features['image/label']
    # preprocess
    # subtract mean valu
    # rgb_mean=np.array([123.68, 116.779, 103.939])
    # img = tf.subtract(img, rgb_mean)
    # red, green, blue = tf.split(3, 3, img)
    # img = tf.concat(3, [
    #     tf.subtract(red , bgr_mean[2]),
    #     tf.subtract(green , bgr_mean[1]),
    #     tf.subtract(blue , bgr_mean[0]),
    # ])
    # center_crop
    # img = tf.image.resize_images(img, [256, 256])
    # j = int(round((256 - 224) / 2.))
    # i = int(round((256 - 224) / 2.))
    # img = img[j:j+224, i:i+224, :]

    # scale to 1
    image = tf.cast(image, tf.float32) * 1.0 / 255.
    return image, label,


# read tf_record
def read_tfrecord_seg_dataset(example_proto):
    input_shape = [256, 256, 3]
    features = {'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/label': tf.FixedLenFeature([256 * 256], tf.int64)}

    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.decode_raw(parsed_features['image/encoded'], tf.uint8)
    h, w, c = input_shape
    image = tf.reshape(image, [h, w, c])

    label = parsed_features['image/label']
    image = tf.cast(image, tf.float32) * 1.0 / 255.
    return image, label
