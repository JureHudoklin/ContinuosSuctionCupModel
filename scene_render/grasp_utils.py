import numpy as np
import os
import pickle
import trimesh



def transform_grasp(grasp_dict, tf, scale_to_meters = True, add_transform_score = True):

    new_grasp_tf = []
    new_grasp_scores = []
    
    for grasp_tf, grasp_score in zip(grasp_dict["tf"], grasp_dict["scores"]):
        # Scale grasp to m from mm
        if scale_to_meters:
            grasp_tf = scale_grasp(grasp_tf)
        # Calculate new tf for the grasp
        new_tf = np.dot(tf, grasp_tf)
        # Calculate new grasp score
        if add_transform_score:
            temp_points = trimesh.transform_points([np.array([0, 0, 0]), np.array([0, 0, 1.])], new_tf)
            approach_vector = temp_points[1]-temp_points[0]
            new_score = grasp_score * (0.5 + approach_vector[2]*0.5)

        new_grasp_tf.append(new_tf)
        new_grasp_scores.append(new_score)

    grasp_dict["tf"] = new_grasp_tf
    if add_transform_score:
        grasp_dict["scores"] = new_grasp_scores

    return grasp_dict

def scale_grasp(grasp_tf, scale=0.001):
    grasp_tf[0:3, 3]=grasp_tf[0:3, 3] * scale
    return grasp_tf

def transform_point_array(point_arr, tf):
    points = np.copy(point_arr)
    temp = np.ones((points.shape[1]+1, points.shape[0]))
    temp[0:3,:] = points.T
    temp = np.dot(tf, temp)
    return temp[0:3, :].T
