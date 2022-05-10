#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import argparse

import numpy as np
import tensorflow as tf

from network.config import Config
from network.suction_graspnet import SuctionGraspNet, build_suction_pointnet_graph
from data_generator import DataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class TestDataGenerator(tf.keras.utils.Sequence):
    """Datset Generator for the network test purpose
    """
    def __init__(self, batch_size=3):
        self.batch_size = batch_size

        # input point cloud
        self.input_pc = np.random.rand(batch_size, 20000, 3)
        self.input_pc = tf.constant(self.input_pc, dtype=tf.float32)

        # grasp score
        self.grasp_score = self.input_pc[:, :20000, :]
        self.grasp_score = np.where(self.grasp_score[:, :, 2] > 0.5, 1, 0)
        self.grasp_score = np.expand_dims(self.grasp_score, axis=-1)
        self.grasp_score = tf.constant(self.grasp_score, dtype=tf.int32)
        self.grasp_score = tf.squeeze(self.grasp_score, axis=-1)

        # grasp approach
        self.grasp_approach = np.random.rand(
            batch_size, 20000, 3).astype(np.float32)
        self.grasp_approach = tf.math.l2_normalize(self.grasp_approach, axis=-1)

    def __len__(self):
        return 100

    def __getitem__(self, index):
        return self.input_pc, (self.grasp_score, self.grasp_approach)


if __name__ == '__main__':
    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # allow memory growth

    ###################
    # Parse Arguments #
    ###################
    # arg parser
    paser = argparse.ArgumentParser()
    paser.add_argument('--saved-model-dir', type=str, help='Put trained model directory.')
    paser.add_argument('--init-epoch', type=int, default=-1, help='Initial epoch number. If -1, start from the latest epoch.')
    args = paser.parse_args()
    saved_model_dir = args.saved_model_dir
    init_epoch = args.init_epoch

    # get model directory
    if not os.path.isdir(saved_model_dir):
        raise ValueError('Model directory does not exist: {}'.format(saved_model_dir))
    log_dir = os.path.join(saved_model_dir, "logs")

    # load train config from the saved model directory
    config = Config(os.path.join(saved_model_dir, 'config.yml'))
    train_config = config.load()

    #################
    # Load Datatset #
    #################
    data_dir = os.path.join(BASE_DIR, "data")
    train_dataset = DataGenerator(data_dir, 3, splits="train", threshold=0.05, search_radius=0.003)
    test_dataset = DataGenerator(data_dir, 3, splits="test", threshold=0.05, search_radius=0.003)

    ##############
    # Load Model #
    ##############
    # build model
    inputs, outputs = build_suction_pointnet_graph(train_config)
    model = SuctionGraspNet(inputs, outputs)

    # compile model
    lr_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=train_config["LR"],
        decay_steps=1,
        decay_rate=train_config["DECAY"])
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr_scheduler))

    # load weights
    weight_filelist = glob.glob(os.path.join(saved_model_dir, "weights/*.h5"))
    weight_filelist.sort()
    epoch_list = [int(os.path.basename(weight_file).split('-')[0]) for weight_file in weight_filelist]
    epoch_list = np.array(epoch_list)
    if init_epoch == -1:
        weight_file = weight_filelist[-1]
        init_epoch = epoch_list[-1]
    else:
        idx = np.where(epoch_list == init_epoch)[0][0]
        weight_file = weight_filelist[idx]
        init_epoch = epoch_list[idx]
    model.load_weights(weight_file)
    print("Loaded weights from {}".format(weight_file))

    ####################
    # Prepare Training #
    ####################
    # callbacks
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(saved_model_dir, "weights", "{epoch:03d}-{val_total_loss:.3f}.h5"),
        save_weights_only=False,
        monitor="val_total_loss",
        mode="min",
        save_best_only=train_config["SAVE_BEST_ONLY"])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=0,
                                                          profile_batch=0,
                                                          write_graph=True,
                                                          write_images=False)

    # train model
    model.fit(
        train_dataset,
        epochs=train_config["EPOCH"],
        callbacks=[save_callback, tensorboard_callback],
        validation_data=test_dataset,
        validation_freq=1,
        max_queue_size=100,
        initial_epoch=init_epoch)
