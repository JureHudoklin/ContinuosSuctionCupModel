#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import argparse
import datetime

import numpy as np
import tensorflow as tf

from network.config import Config
from network.suction_graspnet import SuctionGraspNet, build_suction_pointnet_graph
from data_generator import DataGenerator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # allow memory growth

    ###################
    # Parse Arguments #
    ###################
    # arg parser
    paser = argparse.ArgumentParser()
    paser.add_argument('--saved-model-dir', type=str, help='Put trained model directory.', default = None)
    paser.add_argument('--init-epoch', type=int, default=-1, help='Initial epoch number. If -1, start from the latest epoch.')
    args = paser.parse_args()
    saved_model_dir = args.saved_model_dir
    init_epoch = args.init_epoch

    # get model directory
    if saved_model_dir is None:
        print("--- No model directory provided. Training from scratch ---")
        # set model dirs
        cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join("checkpoints", cur_date)
        log_dir = os.path.join(model_dir, "logs")

        os.makedirs(os.path.join(model_dir, 'weights'))
        print("Directory: ", model_dir, ". Created")
        
        # load and save config
        config = Config('network/config.yml')
        train_config = config.load()
        config.save(train_config, os.path.join(model_dir, 'config.yml'))


    elif os.path.isdir(saved_model_dir):
        print(f"--- Continuing training from: {saved_model_dir} ---")
        log_dir = os.path.join(saved_model_dir, "logs")
        model_dir = saved_model_dir

        # load train config from the saved model directory
        config = Config(os.path.join(saved_model_dir, 'config.yml'))
        train_config = config.load()
    else:
        raise ValueError('Model directory does not exist: {}'.format(saved_model_dir))

    
    #################
    # Load Datatset #
    #################
    data_dir = os.path.join(ROOT_DIR, "data")
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

    if saved_model_dir is not None:
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
        filepath=os.path.join(model_dir, "weights", "{epoch:03d}-{val_total_loss:.3f}.h5"),
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