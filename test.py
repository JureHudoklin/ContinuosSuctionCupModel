#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import argparse

import numpy as np
import tensorflow as tf
import SuctionModel.suction_cup_imaging as sci
from scene_render.render_utils import network_out_tf

from network.config import Config
from network.suction_graspnet import * #SuctionGraspNet, build_suction_pointnet_graph
from scene_render.data_generator import DataGenerator
from scene_render.create_table_top_scene import TableScene

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: 1. Visualize scene it too heavy, consider to use open3d.
# TODO: 2. Visualzie the pred approaches with score.
# TODO: 3. Compare with ground truth approaches.
# TODO: (MINOR) 4. Dataset Generator --> batch_size = 1 --> shape (1, 20000, 3)


def format_output(gt_pc, gt_scores, gt_approach,
                  pred_pc, pred_scores, pred_approach,
                  threshold=0.5):
    # Match Input and output
    indeces_all = tf.squeeze(knn_point(1, gt_pc, pred_pc)[1])
    indeces_all = tf.ensure_shape(
        indeces_all, (None, 2048), name=None
    )
    output_formated = {"indeces_all":indeces_all}
    
    # Match output points and original PC points
    output_formated["gt_scores_resized"] = tf.gather(
        gt_scores, indeces_all, axis=1, batch_dims=1)
    output_formated["gt_approach_resized"] = tf.gather(
        gt_approach, indeces_all, axis=1, batch_dims=1)
    output_formated["gt_pc_resized"] = tf.gather(
        gt_pc, indeces_all, axis=1, batch_dims=1)



    # Pull out positive predictions
    pred_positives_mask = tf.math.greater(pred_scores, threshold)
    gt_positives_mask = gt_scores

    # Predicted approaches that have their length scaled by their score
    output_formated["pred_approach_length_scaled"] = pred_approach * \
        tf.expand_dims(pred_scores, axis=2)
    output_formated["pred_approach_length_scaled"] = output_formated["pred_approach_length_scaled"][0]

    # FIlter out the predictions based on threshold
    output_formated["pred_positive_grasp_points"] = tf.boolean_mask(
        pred_pc[0],  pred_positives_mask[0])
        
    output_formated["gt_positive_grasp_points"] = tf.boolean_mask(
        gt_pc[0], gt_positives_mask[0])  

    output_formated["pred_positive_app_vectors"] = tf.boolean_mask(
        pred_approach_tensor[0], pred_positives_mask[0])
    output_formated["gt_positive_app_vectors"] = tf.boolean_mask(
        gt_approach_tensor[0], gt_positives_mask[0])

    output_formated["pred_pc"] = pred_pc[0]
    output_formated["gt_pc"] = gt_pc[0]
    return output_formated


def visualize_output(output_formated, scene, splits, data_dir):
    my_scene = sci.SuctionCupScene()

    scenes_3d_dir = os.path.join(os.path.join(data_dir, "scenes_3d"), splits)
    # Load existing scene
    table_scene = TableScene(splits, data_dir=data_dir)
    table_scene.load_existing_scene(scene, scenes_3d_dir)
    tri_scene = table_scene.as_trimesh_scene()
    # Display the ground truth approach vectors
    # Display the predicted approach vectors

    for i in range(len(output_formated["pred_approach_length_scaled"])):
        point_1 = output_formated["pred_pc"][i, :]
        direction = output_formated["pred_approach_length_scaled"][i, :]
        color = tf.linalg.norm(direction)
        point_2 = point_1 + direction/20
        my_scene.plot_vector(point_1, point_2, radius_cyl=0.001, color=[color*255, 0, 255, 255], arrow=False)
    my_scene.plot_point_cloud(output_formated["gt_pc"], color=[0, 255, 0, 50])
    my_scene.display(tri_scene)
    return


def best_grasp(pred_pc, pred_scores, pred_approach):
    # Get the max score
    max_grasp_i = tf.argmax(pred_scores, axis = 1)
    print(max_grasp_i)
    location = tf.gather(pred_pc, max_grasp_i, axis=1, batch_dims=1)
    approach = tf.gather(pred_approach, max_grasp_i, axis=1, batch_dims=1)
    return location, approach


def visualize_best_grasp(scene, splits, location, approach, data_dir):
    my_scene = sci.SuctionCupScene()

    scenes_3d_dir = os.path.join(os.path.join(data_dir, "scenes_3d"), splits)
    # Load existing scene
    table_scene = TableScene(splits, data_dir=data_dir)
    table_scene.load_existing_scene(scene, scenes_3d_dir)
    tri_scene=table_scene.as_trimesh_scene()
    # Visualize the generated grasp
    my_scene.plot_vector(location, location+(approach)/10, radius_cyl=0.001)

    my_scene.display(tri_scene)
    return


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

    ###########
    # predict #
    ###########
    # evaluate the model
    scene_number = 3
    pc_tensor, (gt_scores_tensor,
                gt_approach_tensor) = test_dataset[scene_number]
    pc_contact, pred_scores_tensor, pred_approach_tensor = model(pc_tensor)

    # Convert back to world frame
    pc_contact = pc_contact + test_dataset.lb_mean
    pc_tensor = pc_tensor + test_dataset.lb_mean
    pc_contact, pred_approach_tensor = network_out_tf(
        pc_contact, pred_approach_tensor, test_dataset.lb_cam_inverse, inverse=True)
    pc_tensor, gt_approach_tensor = network_out_tf(
        pc_tensor, gt_approach_tensor, test_dataset.lb_cam_inverse, inverse=True)

    #test_dataset.display_gt_pred("000000", pred_pc = pc_contact[0], pred_approach=pred_approach_tensor[0])

    # Best Grasp
    location, approach = best_grasp(
        pc_contact, pred_scores_tensor, pred_approach_tensor)
    # visualize_best_grasp(
    #     "000000", "train", location[0], approach[0], data_dir)

    output_formated = format_output(
        pc_tensor, gt_scores_tensor, gt_approach_tensor, pc_contact, pred_scores_tensor, pred_approach_tensor,
        threshold=0.05)
    visualize_output(output_formated, f"{scene_number:06d}", "test", data_dir)

