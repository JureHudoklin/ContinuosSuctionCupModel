import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from trimeshVisualize import Scene

def visualize_network_input(pc, gt, scene = None, return_scene = False):
    """ Given a point cloud and a ground truth grasp, visualize the point cloud and the grasp.
    If a batch of data is provided only the first batch element will get shown

    Arguments:
    ------------
    pc = (N, 3) np.array
    gt = (N, 3) np.array
    """
    
    gt_scores, gt_app = gt

    if isinstance(pc, tf.Tensor):
        pc = pc.numpy()
    if isinstance(gt_app, tf.Tensor):
        gt_app = gt_app.numpy()
    if isinstance(gt_scores, tf.Tensor):
        gt_scores = gt_scores.numpy()

    if pc.ndim == 3:
        print("Showing only the first batch element")
        pc, gt_scores, gt_app = pc[0], gt_scores[0], gt_app[0]

    grasp_tf = grasps_to_tf(pc, gt_app)
    if scene is None
        my_scene = Scene()
    my_scene.plot_point_cloud(pc, color=[0, 255, 0, 255])
    my_scene.plot_grasp(grasp_tf, gt_scores, color=[100, 255, 0, 255])
    if return_scene:
        return my_scene
    else:
        my_scene.display()
        return None

def visualize_network_output(pc, score, app, scene = None, return_scene = False):
    """ Given a point cloud and a ground truth grasp, visualize the point cloud and the grasp.
    If a batch of data is provided only the first batch element will get shown

    Arguments:
    ------------
    pc = ({B}, N, 3) np.array
    app = ({B}, N, 3) np.array
    score = ({B}, N) np.array
    """

    if isinstance(pc, tf.Tensor):
        pc = pc.numpy()
    if isinstance(app, tf.Tensor):
        app = app.numpy()
    if isinstance(score, tf.Tensor):
        score = score.numpy()

    if pc.ndim == 3:
        print("Showing only the first batch element")
        pc, app, score = pc[0], app[0], score[0]

    grasp_tf = grasps_to_tf(pc, app)
   
    if scene is None
        my_scene = Scene()
    my_scene.plot_point_cloud(pc, color=[0, 0, 255, 255])
    my_scene.plot_grasp(grasp_tf, score, color=[100, 0, 255, 255])
    if return_scene:
        return my_scene
    else:
        my_scene.display()
        return None

def grasps_to_tf(pc, approach):
    """Transforms a combination of a point and approach vector to a 4x4 tf matrix
    Arguments:
    -----------
    pc = (N, 3) np.array
    approach = (N, 3) np.array

    Returns:
    -----------
    (N, 4, 4) np.array
    """

    z = approach
    x = np.cross(np.array([0, 1, 0]), z)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    grasp_tf = np.zeros((pc.shape[0], 4, 4))
    rot = np.stack([x, y, z], axis=-1)
    grasp_tf[:, :3, :3] = rot
    grasp_tf[:, :3, 3] = pc
    grasp_tf[:, 3, 3] = 1
    
    return grasp_tf
    
def best_grasp(pred_pc, pred_scores, pred_approach):
    """
    Given the network outputs, return the best grasps
    """
    # Get the max score
    max_grasp_i = tf.argmax(pred_scores, axis = 1)
    location = tf.gather(pred_pc, max_grasp_i, axis=1, batch_dims=1) # (B, 1, 3)
    approach = tf.gather(pred_approach, max_grasp_i, axis=1, batch_dims=1) # (B, 1, 3)
    score = tf.gather(pred_scores, max_grasp_i, axis=1, batch_dims=1) # (B, 1)
    return location, approach, score