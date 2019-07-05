import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD

import cfg
from network import East
from losses import quad_loss
from data_generator import gen

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)


east = East()
east_network = east.east_network()
east_network.summary()
east_network.compile(loss=quad_loss, optimizer=SGD(lr=cfg.lr,
                                                   clipvalue=cfg.clipvalue,
                                                   decay=cfg.decay))
if cfg.load_weights and os.path.exists(cfg.pretrained_model):
    print('Load pretrained 512*512 model')
    east_network.load_weights(cfg.pretrained_model)

east_network.fit_generator(generator=gen(),
                           steps_per_epoch=cfg.steps_per_epoch,
                           epochs=cfg.epoch_num,
                           validation_data=gen(is_val=True),
                           validation_steps=cfg.validation_steps,
                           verbose=1,
                           initial_epoch=cfg.initial_epoch,
                           callbacks=[
                               EarlyStopping(patience=cfg.patience, verbose=1),
                               ModelCheckpoint(filepath=cfg.model_weights_path,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               verbose=1)])
east_network.save(cfg.saved_model_file_path)
east_network.save_weights(cfg.saved_model_weights_file_path)
