import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from trimeshVisualize import Scene

def visualize_network_input(pc, gt):
    my_scene = Scene()
    
    gt_scores, gt_app = gt
    pc, gt_scores, gt_app = pc[0], gt_scores[0], gt_app[0]
    
    if isinstance(pc, tf.Tensor):
        pc = pc.numpy()
    if isinstance(gt_app, tf.Tensor):
        gt_app = gt_app.numpy()
    if isinstance(gt_scores, tf.Tensor):
        gt_scores = gt_scores.numpy()
    
    my_scene.plot_point_cloud(pc, color=[0, 255, 0, 255])
    for i in range(gt_app.shape[0]):
        my_scene.plot_vector()
    my_scene.plot_grasp(gt_app, gt_scores, color=[0, 0, 255, 255])
   
    my_scene.show()