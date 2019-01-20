from utils import *
import cv2
from tqdm import *

from utils import *

root_path = "../../data/"

ones = np.ones((256, 256, 1), dtype=np.float)
mean = np.concatenate((ones * 103.939, ones * 116.779, ones * 123.68), axis=2)


def convert2example(line):
    height = 256
    width = 256
    nClasses = 2

    image_path = root_path + line.strip().split(' ')[0]
    label_path = root_path + line.strip().split(' ')[-1].strip()

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    label_img = cv2.imread(label_path, 1)
    im = np.zeros((height, width, 3), dtype='uint8')
    im[:, :, :] = 128
    lim = np.zeros((height, width, 3), dtype='uint8')

    if img.shape[0] >= img.shape[1]:
        scale = img.shape[0] / height
        new_width = int(img.shape[1] / scale)
        diff = (width - new_width) // 2
        img = cv2.resize(img, (new_width, height))
        label_img = cv2.resize(label_img, (new_width, height))

        im[:, diff:diff + new_width, :] = img
        lim[:, diff:diff + new_width, :] = label_img
    else:
        scale = img.shape[1] / width
        new_height = int(img.shape[0] / scale)
        diff = (height - new_height) // 2
        img = cv2.resize(img, (width, new_height))
        label_img = cv2.resize(label_img, (width, new_height))
        im[diff:diff + new_height, :, :] = img
        lim[diff:diff + new_height, :, :] = label_img
    lim = lim[:, :, 0]
    # im = np.float32(im) / 127.5 - 1
    im = np.float32(im) - mean
    im = np.array(im, dtype=np.float32)
    lim = np.reshape(lim, (width * height))
    lim = np.array(lim, dtype=np.int32)
    image_data = im.tobytes()
    seg_labels = lim
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/label': int64_feature(seg_labels)
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
    #create_tf_record('seg_train.txt', '../tf_data/train.tfrecord')
    create_tf_record('seg_test.txt', '../tf_data/test.tfrecord')


if __name__ == '__main__':
    tf.app.run()
