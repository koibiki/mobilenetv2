import glob

from config import *
from net.mobilenet_v2 import MobileNetV2
from net.mobilenet_unet import Unet

import math
from utils import *

# unet = Unet(2, restore=True)

net = MobileNetV2(2)

train_tf_path = ["./tf_data/train_cat_dog.tfrecord"]

net.train(read_tfrecord_dataset, train_tf_path)
