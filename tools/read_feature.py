import glob

from config import *
from utils import *

glob_pattern = os.path.join('../tf_data', '*.tfrecord')
tfrecords_list = glob.glob(glob_pattern)
filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=None)

batch_size = 2
img_batch, label_batch = get_batch_seg(filename_queue, batch_size)
print("-----")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print("sss")
    imgs, labels = sess.run([img_batch, label_batch])
    print(np.sum(labels))
