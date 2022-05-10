#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import datetime

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
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # allow memory growth

    ######################
    # Set Save Directory #
    ######################
    # set model dirs
    cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join("checkpoints", cur_date)
    log_dir = os.path.join(model_dir, "logs")

    # create model directory
    try:
        os.makedirs(os.path.join(model_dir, 'weights'))
        print("Directory: ", model_dir, ". Created")
    except FileExistsError:
        print("Directory: ", model_dir, ". Already exists")

    # load and save config
    config = Config('network/config.yml')
    train_config = config.load()
    config.save(train_config, os.path.join(model_dir, 'config.yml'))

    #################
    # Load Datatset #
    #################
    data_dir = os.path.join(BASE_DIR, "data")

    train_dataset = DataGenerator(
        data_dir, 3, splits="train", threshold=0.05, search_radius=0.003)
    test_dataset = DataGenerator(data_dir, 3, splits="test", threshold=0.05,
                                 search_radius=0.003)

    ####################
    # Prepare Training #
    ####################
    # build model
    inputs, outputs = build_suction_pointnet_graph(train_config)

    model = SuctionGraspNet(inputs, outputs)

    # complie model
    lr_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=train_config["LR"],
        decay_steps=1,
        decay_rate=train_config["DECAY"])
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr_scheduler))

    # callbacks
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        # filepath=os.path.join(model_dir),
        filepath=os.path.join(model_dir, "weights", "{epoch:03d}-{val_total_loss:.3f}.h5"),
        save_weights_only=True,
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
        )
