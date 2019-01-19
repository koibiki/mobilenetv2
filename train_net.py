import glob

from config import *
from net.mobilenet_v2 import MobileNetV2
import math
from utils import *

glob_pattern = os.path.join('./tf_data', '*.tfrecord')
tfrecords_list = glob.glob(glob_pattern)
filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=None)

batch_size = cfg.TRAIN.BATCH_SIZE
img_batch, label_batch = get_batch(filename_queue, batch_size)

sample_count = sum([sum(1 for _ in tf.python_io.tf_record_iterator(item)) for item in tfrecords_list])

num_iterations = int(math.ceil(sample_count / cfg.TRAIN.BATCH_SIZE))

mobile_net_v2 = MobileNetV2(2)

mobile_net_v2.train(img_batch, label_batch, num_iterations)
