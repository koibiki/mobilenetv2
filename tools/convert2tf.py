import os
from PIL import Image
import tensorflow as tf
from utils import *
from tqdm import *
import cv2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'imagenet', 'The name of the dataset to convert.')
tf.app.flags.DEFINE_string('img_dir', './data/imagenet', 'The directory store images.')
tf.app.flags.DEFINE_string('train_datas', 'train.txt', 'The images and their labels')
tf.app.flags.DEFINE_string('output_dir', './tfrecords', 'Output directory where to store TFRecords files.')


def convert2example(line):
    img_path, label = line.strip().split()

    image = cv2.imread(img_path)

    img = cv2.resize(image, (224, 224))

    image_data = img.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/label': int64_feature(int(label))
    }))

    return example


def create_tf_record(datas, tf_name):
    num_samples = 0
    with tf.python_io.TFRecordWriter(tf_name) as tfWriter:
        with open(datas, 'r') as f:
            readlines = f.readlines()
            for i in tqdm(range(len(readlines))):
                example = convert2example(readlines[i])
                tfWriter.write(example.SerializeToString())
                num_samples += 1


def main(_):
    create_tf_record('train.txt', '../tf_data/train.tfrecord')


if __name__ == '__main__':
    tf.app.run()
