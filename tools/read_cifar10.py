import os.path as osp
import pickle
import cv2
import numpy as np


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


def load_data(train_file):
    d = unpickle(train_file)
    # dict_keys([b'batch_label', b'filenames', b'data', b'coarse_labels', b'fine_labels'])，每个键值前面都有一个b，不同于 python2
    datas = d[b'data']
    labels = d[b'labels']
    length = len(d[b'labels'])

    datas = datas.reshape(length, 3, 32, 32)

    datas = np.transpose(datas, (0, 2, 3, 1))

    return (
        datas,
        labels
    )


if __name__ == '__main__':
    # 解压后的 cifar-100-python 路径
    cifar_python_directory = "../../data/cifar10/cifar-10-batches-py"

    datas, labels = load_data(osp.join(cifar_python_directory, 'data_batch_1'))
    print('Converting...')
