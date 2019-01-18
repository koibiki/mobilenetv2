import os
import os.path as osp
import random

from tqdm import *

root_path = '../data/train'
label_dict = {'cat': 0, 'dog': 1}



file_list = []
label_list = []
for item in os.listdir(root_path):
    label = item
    for img_name in os.listdir(os.path.join(root_path, item)):
        if 'cat' in img_name or 'dog' in img_name:
            file_list.append(osp.join(osp.join(root_path, item), img_name))
            label_list.append(label_dict[label])

indexes = [i for i in range(len(label_list))]
random.shuffle(indexes)
shuffled_file = [file_list[indexes[i]] for i in range(len(label_list))]
shuffled_label = [label_list[indexes[i]] for i in range(len(label_list))]

with open('train.txt', 'wb') as f:
    for i in tqdm(range(len(shuffled_label))):
        f.write((shuffled_file[i] + ' ' + str(shuffled_label[i])).encode())
        f.write('\n'.encode())
