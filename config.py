from easydict import EasyDict as edict
import os
import os.path as osp

cfg = edict()

# cfg.INPUT_SIZE = (224, 224, 3)
cfg.INPUT_SIZE = (256, 256, 3)
cfg.PATH = edict()
cfg.PATH.ROOT_DIR = os.getcwd()
cfg.PATH.TBOARD_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'logs'))
cfg.PATH.MODEL_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'checkpoints'))
cfg.PATH.TFLITE_MODEL_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'tf_lite_model'))

cfg.TRAIN = edict()
# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.INPUT_SHAPE = (cfg.TRAIN.BATCH_SIZE, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], cfg.INPUT_SIZE[2])
cfg.TRAIN.LEARNING_RATE = 0.001
cfg.TRAIN.LR_DECAY_STEPS = 10000
cfg.TRAIN.LR_DECAY_RATE = 0.9
cfg.TRAIN.EPOCHS = 500
cfg.TRAIN.DISPLAY_STEP = 1
cfg.TRAIN.SAVE_MODEL_STEP = 500
cfg.TRAIN.GPU_MEMORY_FRACTION = 0.5
cfg.TRAIN.TF_ALLOW_GROWTH = True

# TEST
cfg.TEST = edict()
cfg.TEST.BATCH_SIZE = 2
cfg.TEST.INPUT_SHAPE = (cfg.TEST.BATCH_SIZE, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], cfg.INPUT_SIZE[2])
