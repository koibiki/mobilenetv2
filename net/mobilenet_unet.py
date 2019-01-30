from net.base_net import BaseNet
from net.ops import *
from config import cfg
import os
import os.path as osp
import time
import numpy as np
from tqdm import *
import math


class Unet(BaseNet):

    def __init__(self, num_class, is_train=True, restore=False, weight_path=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self._batch_size = cfg.TRAIN.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        self._input_shape = cfg.TRAIN.INPUT_SHAPE if is_train else cfg.TEST.INPUT_SHAPE
        self.weight_path = weight_path
        self._input = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name='input')
        self._y = tf.placeholder(dtype=tf.int32, shape=(self._batch_size, self._input_shape[1] * self._input_shape[2]),
                                 name='y')
        self.sess = self.build_sess()
        with self.sess.as_default():
            self.out = self.build_unet(self._input, num_class, is_train)
            self.global_step = tf.Variable(0, trainable=False)
            self.saver, self.model_save_path = self.build_saver()
            if not is_train or restore:
                self.load_model(self.sess, self.saver, weight_path)

    def build_unet(self, _inputs, num_class, is_train):
        with tf.variable_scope('mobilenetv2'):
            conv1 = conv2d_block(_inputs, 16, 3, 1, is_train=is_train, name='conv1_1')

            conv2 = conv2d_block(conv1, 16, 3, 2, is_train=is_train, name='conv2_1')

            res2 = inverted_res_block(conv2, 16, (3, 3), t=1, s=2, is_train=is_train, n=2, name='res2')
            res3 = inverted_res_block(res2, 24, (3, 3), t=6, s=2, is_train=is_train, n=2, name='res3')
            res4 = inverted_res_block(res3, 32, (3, 3), t=6, s=2, is_train=is_train, n=3, name='res4')
            res5 = inverted_res_block(res4, 64, (3, 3), t=6, s=2, is_train=is_train, n=4, name='res5')
            res6 = inverted_res_block(res5, 96, (3, 3), t=6, s=2, is_train=is_train, n=3, name='res7')

            dec9 = decode(res6, res5, 96, (3, 3), t=6, s=1, is_train=is_train, n=2, name="dec9")
            dec10 = decode(dec9, res4, 64, (3, 3), t=6, s=1, is_train=is_train, n=3, name="dec10")
            dec11 = decode(dec10, res3, 32, (3, 3), t=6, s=1, is_train=is_train, n=2, name="dec11")
            dec12 = decode(dec11, res2, 24, (3, 3), t=6, s=1, is_train=is_train, n=1, name="dec12")
            dec13 = decode(dec12, conv2, 16, (3, 3), t=6, s=1, is_train=is_train, n=1, name="dec13")
            dec14 = decode(dec13, conv1, 16, (3, 3), t=6, s=1, is_train=is_train, n=1, name="dec14")

            logits = slim.conv2d(dec14, num_class, 1, 1, scope='logits')
            logits = tf.reshape(logits, shape=(self._batch_size, -1, 2))

            pred = slim.softmax(logits, scope='prob')

        out = {"conv1": conv1, "conv2": conv2,
               "res2": res2, "res3": res3, "res4": res4, "res5": res5, "res6": res6,
               "dec9": dec9, "dec10": dec10, "dec11": dec11, "dec12": dec12, "dec13": dec13, "dec14": dec14,
               "logits": logits, "pred": pred}

        return out

    def build_loss(self, out, y):
        cross_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=out["logits"]))

        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        total_loss = cross_loss + l2_loss

        return total_loss, cross_loss, l2_loss

    def build_evaluator(self, out, y):
        correct_pred = tf.equal(tf.argmax(out["pred"], axis=-1), tf.cast(y, tf.int64))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return acc

    def build_summary(self, total_cost, acc, cost, learning_rate):
        os.makedirs(cfg.PATH.TBOARD_SAVE_DIR, exist_ok=True)
        tf.summary.scalar(name='Cost', tensor=tf.reduce_sum(cost))
        tf.summary.scalar(name='Acc', tensor=tf.reduce_sum(acc))
        tf.summary.scalar(name='Total_Cost', tensor=tf.reduce_sum(total_cost))
        tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
        merge_summary_op = tf.summary.merge_all()
        return merge_summary_op

    def train(self, parse_func, train_tf_record_path, valid_tf_record_path=None):
        with self.sess.as_default():
            total_loss, cross_loss, reg_loss = self.build_loss(self.out, self._y)

            optimizer, lr = self.build_optimizer(total_loss, self.global_step)

            train_epochs = cfg.TRAIN.EPOCHS

            acc_op = self.build_evaluator(self.out, self._y)

            summary_op = self.build_summary(total_loss, acc_op, cross_loss, lr)

            train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            log_dir = osp.join(cfg.PATH.TBOARD_SAVE_DIR, train_start_time)
            os.mkdir(log_dir)

            summary_writer = tf.summary.FileWriter(log_dir)
            summary_writer.add_graph(self.sess.graph)

            self.sess.run(tf.global_variables_initializer())

            train_example, train_iterator = self.build_train_dataset(self.sess, train_tf_record_path, parse_func)
            train_data_amount = self.get_tf_data_amount(train_tf_record_path)
            train_num_iter = int(math.ceil(train_data_amount / cfg.TRAIN.BATCH_SIZE))

            valid_data_amount = 0
            valid_example = None
            valid_iterator = None
            if valid_tf_record_path is not None:
                valid_example, valid_iterator = self.build_valid_dataset(self.sess, valid_tf_record_path, parse_func)
                valid_data_amount = self.get_tf_data_amount(valid_tf_record_path)

            _global_step = 0
            for epoch in range(train_epochs):
                all_acc = 0
                all_t_c = 0
                all_cross_c = 0
                train_pbar = tqdm(range(train_num_iter))
                for _step in train_pbar:
                    train_data, train_label = self.sess.run(train_example)

                    summary, _, t_c, c_c, l1_c, _acc, _lr, pred_prob = self.sess.run(
                        [summary_op, optimizer, total_loss, cross_loss, reg_loss, acc_op, lr, self.out['pred']],
                        feed_dict={self._input: train_data,
                                   self._y: train_label})

                    all_t_c += np.sum(t_c)
                    all_cross_c += np.sum(c_c)
                    all_acc += np.sum(_acc)

                    desc_str = 'Epoch:{:3d}/{:3d} train cost= {:5f} cross_c={:5f} l1_c={:5f} lr={:5f} acc={:6f}'.format(
                        epoch + 1, train_epochs, all_t_c / (_step + 1), all_cross_c / (_step + 1), np.sum(l1_c), _lr,
                        all_acc / (_step + 1))
                    train_pbar.set_description(desc_str)

                    _global_step = epoch * train_num_iter + _step
                    summary_writer.add_summary(summary=summary, global_step=_global_step)

                if valid_tf_record_path is not None:
                    test_pbar = tqdm(range(valid_data_amount))
                    all_test_loss = 0
                    all_test_acc = 0
                    for _step in test_pbar:
                        train_data, train_label = self.sess.run(valid_example)

                        test_c, test_acc = self.sess.run([cross_loss, acc_op],
                                                         feed_dict={self._input: train_data, self._y: train_label})

                        all_test_loss += np.sum(test_c)
                        all_test_acc += np.sum(test_acc)

                        desc_str = 'Epoch: {:4d}/{:4d} valid  cross_c={:9f} acc={:9f}'.format(epoch + 1, train_epochs,
                                                                                              all_test_loss / (_step + 1),
                                                                                              all_test_acc / (_step + 1))
                        test_pbar.set_description(desc_str)

                tf.train.write_graph(self.sess.graph_def, 'checkpoints', 'net_txt.pb', as_text=True)
                self.saver.save(sess=self.sess, save_path=self.model_save_path, global_step=_global_step)

        print('FINISHED TRAINING.')
