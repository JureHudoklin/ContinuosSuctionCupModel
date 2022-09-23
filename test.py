#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import argparse
import random

import numpy as np
import tensorflow as tf

from network.config import Config
from network.suction_graspnet import SuctionGraspNet, build_suction_pointnet_graph
from data_generator import DataGenerator
from util.network_utils import visualize_network_input, visualize_network_output

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
    paser.add_argument('--saved-model-dir', type=str, help='Put trained model directory.')
    paser.add_argument('--init-epoch', type=int, default=-1, help='Put the epoch of the model to be loaded.')
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

    ###########
    # predict #
    ###########
    # evaluate the model
    scene_number = random.randint(0, len(test_dataset))
    pc_tensor, (gt_scores_tensor,
                gt_approach_tensor) = test_dataset[scene_number]
    pc_contact, pred_scores_tensor, pred_approach_tensor = model(pc_tensor)

    my_scene = visualize_network_input(pc_tensor, (gt_scores_tensor,gt_approach_tensor), return_scene=True)
    visualize_network_output(pc_contact, pred_scores_tensor, pred_approach_tensor, my_scene)
