import glob

from config import *
from net.mobilenet_v2 import MobileNetV2
from net.mobilenet_unet import Unet

import math
from utils import *

unet = Unet(2, restore=True)

glob_pattern = os.path.join('./tf_data', 'train.tfrecord')
train_tfrecords_list = glob.glob(glob_pattern)

glob_pattern = os.path.join('./tf_data', 'test.tfrecord')
test_tfrecords_list = glob.glob(glob_pattern)

train_filename_queue = tf.train.string_input_producer(train_tfrecords_list, num_epochs=None)
test_filename_queue = tf.train.string_input_producer(test_tfrecords_list, num_epochs=None)

train_img_batch, train_label_batch = get_batch_seg(train_filename_queue, cfg.TRAIN.BATCH_SIZE)
test_img_batch, test_label_batch = get_batch_seg(test_filename_queue, cfg.TEST.BATCH_SIZE, shuffle=False)

train_sample_count = sum([sum(1 for _ in tf.python_io.tf_record_iterator(item)) for item in train_tfrecords_list])

train_num_iter = int(math.ceil(train_sample_count / cfg.TRAIN.BATCH_SIZE))

test_sample_count = sum([sum(1 for _ in tf.python_io.tf_record_iterator(item)) for item in test_tfrecords_list])

test_num_iter = int(math.ceil(test_sample_count / cfg.TRAIN.BATCH_SIZE))

unet.train(train_img_batch, train_label_batch, train_num_iter, test_img_batch, test_label_batch, test_num_iter)
