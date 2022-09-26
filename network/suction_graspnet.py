#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from .pointnet2_tensorflow2.pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG, Pointnet_FP
from .pointnet2_tensorflow2.pnet2_layers.cpp_modules import select_top_k



# add config as arg later
def build_suction_pointnet_graph(config):
    """
    Build computational graph for Suction-GraspNet

    --------------
    Args:
        config (`Config`) : `Config` instance.
    --------------
    Returns:
        input (`tf.Tensor`) : Input point cloud (B, N, 3).
        outputs (`Tuple`) : (pc_contacts, grasp_socre, grasp_approach).
            pc_contacts (`tf.Tensor`) : Contact point cloud (B, M, 3)
            grasp_score (`tf.Tensor`) : Confidence of the grasps (B, M).
            grasp_approach (`tf.Tensor`) : Approach vector of the grasps (B, M, 3).
    """
    # Input layer
    # (B, 20000, 3)
    input_pc = tf.keras.Input(
        shape=(config["RAW_NUM_POINTS"], 3),
        name='input_point_cloud')


    # Set Abstraction layers
    # (B, 2048, 3), (B, 2048, 320)
    sa_xyz_0, sa_points_0 = Pointnet_SA_MSG(
        npoint=config["SA_NPOINT_0"],
        radius_list=config["SA_RADIUS_LIST_0"],
        nsample_list=config["SA_NSAMPLE_LIST_0"],
        mlp=config["SA_MLP_LIST_0"],
        use_xyz=True,
        activation=tf.nn.relu,
        bn=False)(input_pc, None)
    # (B, 512, 3), (B, 512, 640)
    sa_xyz_1, sa_points_1 = Pointnet_SA_MSG(
        npoint=config["SA_NPOINT_1"],
        radius_list=config["SA_RADIUS_LIST_1"],
        nsample_list=config["SA_NSAMPLE_LIST_1"],
        mlp=config["SA_MLP_LIST_1"],
        use_xyz=True,
        activation=tf.nn.relu,
        bn=False)(sa_xyz_0, sa_points_0)
    # (B, 128, 3), (B, 128, 640)
    sa_xyz_2, sa_points_2 = Pointnet_SA_MSG(
        npoint=config["SA_NPOINT_2"],
        radius_list=config["SA_RADIUS_LIST_2"],
        nsample_list=config["SA_NSAMPLE_LIST_2"],
        mlp=config["SA_MLP_LIST_2"],
        use_xyz=True,
        activation=tf.nn.relu,
        bn=False)(sa_xyz_1, sa_points_1)

    # Global feature layer
    # (B, 1, 3), (1024...?)
    sa_xyz_3, sa_points_3 = Pointnet_SA(
        npoint=None,
        radius=None,
        nsample=None,
        mlp=config["SA_MLP_GROUP_ALL"],
        group_all=True,
        knn=False,
        use_xyz=True,
        activation=tf.nn.relu,
        bn=False)(sa_xyz_2, sa_points_2)

    # Feature propagation layers.
    # (B, 128, 256)
    fp_points_2 = Pointnet_FP(
        mlp=config["FP_MLP_0"],
        activation=tf.nn.relu,
        bn=False)(sa_xyz_2, sa_xyz_3, sa_points_2, sa_points_3)
    # (B, 512, 128)
    fp_points_1 = Pointnet_FP(
        mlp=config["FP_MLP_1"],
        activation=tf.nn.relu,
        bn=False)(sa_xyz_1, sa_xyz_2, sa_points_1, fp_points_2)
    # (B, 2048, 128)
    fp_points_0 = Pointnet_FP(
        mlp=config["FP_MLP_2"],
        activation=tf.nn.relu,
        bn=False)(sa_xyz_0, sa_xyz_1, sa_points_0, fp_points_1)

    # Output from the pointnet++
    # (B, 2048, 3)
    # (B, 2048, 1024)
    output_pc = sa_xyz_0
    output_feature = fp_points_0

    # grasp_score
    # (B, 2048, 1)
    grasp_score = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='valid')(output_feature)
    grasp_score = tf.keras.layers.LeakyReLU()(grasp_score)
    grasp_score = tf.keras.layers.Dropout(rate=0.5)(grasp_score)
    grasp_score = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='valid')(grasp_score)
    grasp_score = tf.keras.activations.sigmoid(grasp_score)
    grasp_score = tf.squeeze(grasp_score, axis=-1)

    # grasp_approach
    # (B, 2048, 3)
    grasp_approach = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='valid')(output_feature)
    grasp_approach = tf.keras.layers.LeakyReLU()(grasp_approach)
    grasp_approach = tf.keras.layers.Dropout(rate=0.5)(grasp_approach)
    grasp_approach = tf.keras.layers.Conv1D(filters=3, kernel_size=1, strides=1, padding='valid')(grasp_approach)
    grasp_approach = tf.math.l2_normalize(grasp_approach, axis=-1)


    return input_pc, (output_pc, grasp_score, grasp_approach)


@tf.function
def knn_point(k, xyz1, xyz2):
    #idx = tf.constanct(tf.int32, shape=(None,2048))

    # This did not work for me so changed
    b = tf.shape(xyz1)[0]
    n = tf.shape(xyz1)[1]
    c = tf.shape(xyz1)[2]
    m = tf.shape(xyz2)[1]


    xyz1 = tf.tile(tf.reshape(xyz1, (b, 1, n, c)), [1, m, 1, 1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b, m, 1, c)), [1, 1, n, 1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)

    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0, 0, 0], [-1, -1, k])
    val = tf.slice(out, [0, 0, 0], [-1, -1, k])

    return val, idx


@tf.function
def score_loss_fn(gt_scores, pred_scores, max_k=512):
    """
    Calculate score loss given ground truth and predicted scores. 
    The inputs must be of dimension [batch_size, num_points].
    --------------
    Args:
        gt_scores (tf.Tensor) : Ground truth scores. (B, N)
        pred_scores (tf.Tensor) : Predicted scores. (B, N)
    --------------
    Returns:
        loss (tf.Tensor) : Binary crossentropy
    """
    # Expand dimensions
    gt_scores = tf.expand_dims(gt_scores, axis=-1)
    pred_scores = tf.expand_dims(pred_scores, axis=-1)
    mask = tf.where(gt_scores != -1, 1, 0)

    # Calculate elementvise binary cross entropy loss
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss = bce(gt_scores, pred_scores, sample_weight=mask)
    # Get the indeces for the sorted losses
    sort_i = tf.argsort(loss, axis=1, direction='DESCENDING')

    # Use the indeces to sort input data
    gt_scores = tf.squeeze(gt_scores)
    gt_scores = tf.gather(gt_scores, sort_i, batch_dims=-1)

    pred_scores = tf.squeeze(pred_scores)
    pred_scores = tf.gather(pred_scores, sort_i, batch_dims=-1)

    # Calculate the loss for top k points
    loss = tf.gather(loss, sort_i, batch_dims=-1)
    loss = tf.reduce_mean(loss[:, :max_k])
    return loss


@tf.function
def approach_loss_fn(gt_approach, pred_approach):
    """
    Calculate the loss for the approach vectors.
    The inputs must be of dimension (B, M, 3).
    Where m are only the predicted points where the ground truth for those points are True !!!!
    --------------
    Args:
        gt_approach (tf.Tensor) : Ground truth approach vectors. (B, M, 3)
        pred_approach (tf.Tensor) : Predicted approach vectors. (B, M, 3)
    --------------
    Returns:
        loss (tf.Tensor): CosineSimilarity
    """
    loss = tf.reduce_mean(
        tf.keras.losses.cosine_similarity(gt_approach, pred_approach)+1)
    return loss


@tf.function
def loss_fn(gt_scores, pred_scores, gt_approach, pred_approach, max_k=256):
    """
    Given formatted ground truth boxes and network output, calculate score and approach loss.
    --------------
    Args:
        gt_scores (tf.Tensor) : Ground truth scores. (B, N)
        pred_scores (tf.Tensor) : Predicted scores. (B, N)
        gt_approach (tf.Tensor) : Ground truth approach vectors. (B, N, 3)
        pred_approach (tf.Tensor) : Predicted approach vectors. (B, N, 3)
    Keyword Args:
        max_k (int) : Amount of points to use for the score loss.
    --------------
    Returns:
        l_score (tf.Tensor) : Score loss value
        l_approach (tf.Tensor) : Approach loss value
    """

    # Calculate score loss
    l_score = score_loss_fn(gt_scores, pred_scores, 512)
    # Filter only grasps that should be positive
    mask = tf.where(gt_scores == 1, True, False)
    gt_approach = tf.boolean_mask(gt_approach, mask)
    pred_approach = tf.boolean_mask(pred_approach, mask)

    # Calculate approach loss
    l_approach = approach_loss_fn(gt_approach, pred_approach)

    return l_score, l_approach


class SuctionGraspNet(tf.keras.models.Model):
    def __init__(self, inputs, outputs):
        super(SuctionGraspNet, self).__init__(inputs, outputs)

    def compile(self, optimizer='adam', run_eagerly=None):
        super(SuctionGraspNet, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)

        # define trackers
        self.grasp_score_acc_tracker = tf.keras.metrics.BinaryAccuracy(name='grasp_sc_acc')
        self.grasp_score_precision_tracker = tf.keras.metrics.Precision(thresholds=0.5, name='grasp_sc_pcs')
        self.grasp_score_loss_tracker = tf.keras.metrics.Mean(name='grasp_sc_loss')
        self.grasp_app_mae_tracker = tf.keras.metrics.MeanAbsoluteError(name='grasp_app_mae')
        self.grasp_app_loss_trakcer = tf.keras.metrics.Mean(name='grasp_app_loss')
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = [
            self.grasp_score_acc_tracker,
            self.grasp_score_precision_tracker,
            self.grasp_score_loss_tracker,
            self.grasp_app_mae_tracker,
            self.grasp_app_loss_trakcer,
            self.total_loss_tracker]
        return metrics

    def train_step(self, data):
        # unpack data
        input_pc, (score_target, approach_target) = data

        # get gradient
        with tf.GradientTape() as tape:
            # get network forward output
            output_pc, score_output, approach_output = self(input_pc, training=True)

            # fromat ground truth boxes
            indeces_all = tf.squeeze(knn_point(1, input_pc, output_pc)[1])
            indeces_all = tf.ensure_shape(
                indeces_all, (None, 2048), name=None
            )
            # Match output points and original PC points
            score_target = tf.gather(
                score_target, indeces_all, axis=1, batch_dims=1)
            approach_target = tf.gather(
                approach_target, indeces_all, axis=1, batch_dims=1)

            # get loss
            score_loss, approach_loss = loss_fn(score_target, score_output,
                                                approach_target, approach_output)
            total_loss = score_loss + approach_loss

        # udate gradient
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        mask_scores = tf.where(score_target != -1, 1, 0)
        mask_scores_exp = tf.expand_dims(mask_scores, -1)
        
        # update loss and metric trackers
        self.grasp_score_acc_tracker.update_state(
            score_target, score_output, sample_weight=mask_scores_exp)
        self.grasp_score_precision_tracker.update_state(
            score_target, score_output, sample_weight=mask_scores_exp)
        self.grasp_score_loss_tracker.update_state(score_loss)
        self.grasp_app_mae_tracker.update_state(approach_target, approach_output)
        self.grasp_app_loss_trakcer.update_state(approach_loss)
        self.total_loss_tracker.update_state(total_loss)

        # pack return
        ret = {
            'score_acc': self.grasp_score_acc_tracker.result(),
            'score_prec': self.grasp_score_precision_tracker.result(),
            'score_loss': self.grasp_score_loss_tracker.result(),
            'app_mae': self.grasp_app_mae_tracker.result(),
            'app_loss': self.grasp_app_loss_trakcer.result(),
            'total_loss': self.total_loss_tracker.result()}
        return ret

    def test_step(self, data):
        # unpack data
        input_pc, (score_target, approach_target) = data

        # get netwokr output
        output_pc, score_output, approach_output = self(input_pc, training=False)

        # fromat ground truth boxes
        indeces_all = tf.squeeze(knn_point(1, input_pc, output_pc)[1])
        indeces_all = tf.ensure_shape(
            indeces_all, (None, 2048), name=None
        )
        # Match output points and original PC points
        score_target = tf.gather(
            score_target, indeces_all, axis=1, batch_dims=1)
        approach_target = tf.gather(
            approach_target, indeces_all, axis=1, batch_dims=1)

        # get loss
        score_loss, approach_loss = loss_fn(score_target, score_output,
                                            approach_target, approach_output)
        total_loss = score_loss + approach_loss

        mask_scores = tf.where(score_target != -1, 1, 0)
        mask_scores = tf.expand_dims(mask_scores, -1)
        # update loss and metric trackers
        self.grasp_score_acc_tracker.update_state(score_target, score_output, sample_weight=mask_scores)
        self.grasp_score_precision_tracker.update_state(
            score_target, score_output, sample_weight=mask_scores)
        self.grasp_score_loss_tracker.update_state(score_loss)
        self.grasp_app_mae_tracker.update_state(approach_target, approach_output)
        self.grasp_app_loss_trakcer.update_state(approach_loss)
        self.total_loss_tracker.update_state(total_loss)

        # pack return
        ret = {
            'score_acc': self.grasp_score_acc_tracker.result(),
            'score_prec': self.grasp_score_precision_tracker.result(),
            'score_loss': self.grasp_score_loss_tracker.result(),
            'app_mae': self.grasp_app_mae_tracker.result(),
            'app_loss': self.grasp_app_loss_trakcer.result(),
            'total_loss': self.total_loss_tracker.result()}
        return ret
