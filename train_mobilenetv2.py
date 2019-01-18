import glob
import os
import time

from config import *
from mb import inference
from mobilenet_v2 import mobilenetv2
from model import *
from utils import *


# from mobilenet_v2 import *


def load(sess, saver, checkpoint_dir):
    import re
    print("[*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("[*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print("[*] Failed to find a checkpoint")
        return False, 0


def main():
    height = 224
    width = 224

    with tf.Session() as sess:

        # read queue
        glob_pattern = os.path.join('./', '*.tfrecord')
        tfrecords_list = glob.glob(glob_pattern)
        filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=None)
        batch_size = 32
        img_batch, label_batch = get_batch(filename_queue, batch_size)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        inputs = tf.placeholder(tf.float32, [batch_size, height, width, 3], name='input')
        y = tf.placeholder(tf.int32, batch_size, name='y')

        logits, pred = mobilenetv2(inputs, 2)

        # logits, end_points = mobilenet_v2(inputs, num_classes=2, dropout_keep_prob=0.5, is_training=True)

        # logits, end_points = inference(inputs, num_classes=2, keep_probability=0.5, phase_train=True)
        # pred = end_points['Predictions']

        # loss
        loss_ = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
        # L2 regularization
        # l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = loss_  # + l2_loss

        # evaluate model, for classification
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.cast(y, tf.int64))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # learning rate decay
        # global_step = tf.placeholder(dtype=tf.float32, shape=())
        # lr = tf.train.exponential_decay(0.01, global_step=global_step, decay_steps=3000,
        #                                 decay_rate=0.95)
        # # optimizer
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     # tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9, momentum=0.9)
        #     train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        starter_learning_rate = 0.001
        lr = tf.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.9, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = tf.train.GradientDescentOptimizer(learning_rate=lr) \
                .minimize(loss=total_loss, global_step=global_step)

        # summary
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('accuracy', acc)
        tf.summary.scalar('learning_rate', lr)
        summary_op = tf.summary.merge_all()

        # summary writer
        writer = tf.summary.FileWriter('logs', sess.graph)

        sess.run(tf.global_variables_initializer())

        # saver for save/restore model
        saver = tf.train.Saver(max_to_keep=3)
        # load pretrained model
        step = 0
        if not args.renew:
            print('[*] Try to load trained model...')
            could_load, step = load(sess, saver, 'checkpoints')

        max_steps = int(1000000 / 8 * 50035)

        all_acc = 0

        filter_step = step

        print('START TRAINING...')
        for _step in range(step + 1, max_steps + 1):
            start_time = time.time()
            train_data, labels = sess.run([img_batch, label_batch])

            # train logs and write summary
            _summ, _, _lr, _loss, _pred, _acc, _logits = sess.run(
                [summary_op, train_op, lr, total_loss, pred, acc, logits],
                feed_dict={inputs: train_data, y: labels})

            all_acc += _acc
            if _step % 10 == 0:
                # pred_acc = [_pred[0][labels[i]] for i in range(len(labels))]
                # print(pred_acc)
                writer.add_summary(_summ, _step)

                print('global_step:{0}, time:{1:.3f}, lr:{2:.8f}, acc:{3:.6f}, loss:{4:.6f}'.format
                      (_step, time.time() - start_time, _lr, all_acc / (_step - filter_step), _loss))

            if _step > 0 and _step % 3000 == 0:
                filter_step = _step
                all_acc = 0

            # save model
            if _step > 0 and _step % 500 == 0:
                save_path = saver.save(sess, os.path.join('checkpoints', 'mobilenet'), global_step=_step)
                print('Current model saved in ' + save_path)

                tf.train.write_graph(sess.graph_def, 'checkpoints', 'mobilenet' + '.pb')
                save_path = saver.save(sess, os.path.join('checkpoints', 'mobilenet'), global_step=max_steps)
        print('Final model saved in ' + save_path)
        sess.close()
        print('FINISHED TRAINING.')


if __name__ == '__main__':
    main()
