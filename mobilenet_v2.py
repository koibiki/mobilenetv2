from ops import *


def mobilenetv2(inputs, num_classes, is_train=True):
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(inputs, 32, 3, 2, is_train=is_train, name='conv1_1')

        net = inverted_res_block(net, 16, (3, 3), t=1, strides=1, is_train=is_train, n=1, name='res2')
        net = inverted_res_block(net, 24, (3, 3), t=6, strides=2, is_train=is_train, n=2, name='res3')
        net = inverted_res_block(net, 32, (3, 3), t=6, strides=2, is_train=is_train, n=3, name='res4')
        net = inverted_res_block(net, 64, (3, 3), t=6, strides=2, is_train=is_train, n=4, name='res5')
        net = inverted_res_block(net, 96, (3, 3), t=6, strides=1, is_train=is_train, n=3, name='res6')
        net = inverted_res_block(net, 160, (3, 3), t=6, strides=2, is_train=is_train, n=3, name='res7')
        net = inverted_res_block(net, 320, (3, 3), t=6, strides=1, is_train=is_train, n=1, name='res8')

        net = conv2d_block(net, 1280, 1, 1, is_train, name='conv9_1')
        net = slim.dropout(net, keep_prob=0.5, scope='dp')
        logits = slim.conv2d(net, num_classes, 7, 7, activation_fn=None, scope='logits')
        logits = global_avg(logits)
        logits = slim.flatten(logits)

        pred = slim.softmax(logits, scope='prob')
        return logits, pred
