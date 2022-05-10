import multiprocessing
import traceback
import os
import sys
import glob
import random

import numpy as np
import pickle
import trimesh
from multiprocessing import Pool

# Shortcut to multiprocessing's logger
def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)


class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise ValueError()

        # It was fine, give a normal answer
        return result


def run_multiprocessing(func, i, n_processors):
    """
    Wrapper for multiprocessing. Logs exceptions in case of bugs.
    """
    with Pool(processes=n_processors) as pool:
        return pool.map(LogExceptions(func), i)
       


def load_grasp(obj_name, grasp_root_dir, filetype=".pkl"):
    """
    Load a grasp for the given object.
    --------------
    Args:
        - obj_name (str): name of the object to load the grasp for
        - grasp_root_dir (str): Root directory where the grasps are located
    Kwargs.
        - filetype (str, .pkl) : The filetype of the grasp
    --------------
    Returns: 
        grasp (dict) : the dictionary containing grasp data for the given object
    """
    grasp_path_list = glob.glob(os.path.join(
        grasp_root_dir, obj_name)+filetype)
    if len(grasp_path_list) > 0:
        grasp_data = open_grasp(grasp_root_dir, obj_name)
        return grasp_data
    else:
        return None


def get_data_paths(data_root_path):
    meshes = glob.glob(os.path.join(data_root_path, "meshes/*"))
    grasps = glob.glob(os.path.join(data_root_path, "grasps/*"))
    meshes_paths = {}
    grasps_paths = {}
    for i in range(2):
        mesh_cat = os.path.basename(meshes[i])
        meshes_paths[mesh_cat] = meshes[i]
        grasp_cat = os.path.basename(grasps[i])
        grasps_paths[grasp_cat] = grasps[i]
    return meshes_paths, grasps_paths


def open_grasp(root_path, obj_name):
    grasp_file = open(os.path.join(root_path, obj_name) + ".pkl", "rb")
    output = pickle.load(grasp_file)
    return output


def load_mesh(filename, mesh_root_dir, scale=0.001):
    """
    Load a mesh given a name and root directory. Can apply scaling if desired.
    --------------
    Args:
        filename (str): name of the object to load
        mesh_root_dir (str) : path to the directory where obj mesh is located
        scale (float, optional): If specified, use this as scale instead of value 1. Defaults to 1.
    --------------
    Returns: 
        trimesh.Trimesh: Mesh of the loaded object.
    """
    obj_mesh = trimesh.load(os.path.join(mesh_root_dir, filename)+".obj")
    obj_mesh.units = ("millimeters")
    if scale != None:
        obj_mesh = obj_mesh.apply_scale(scale)

    return obj_mesh


def load_scene_3d(filename, data_root_dir):
    """
    Load an already existing 3d scene.
    --------------
    Args:
        filename (str): name of the scene to load (withoud file ending).
        data_root_dir (str): Absolute path to where the scene is located
    --------------
    Returns: 
        scene_grasps {list} : A list of grasps where each grasps is a dict containing "tf" and "score".
        object_names {list} : List of object names in the scene
        obj_transforms {list} : Transformation matrices for placing of individual objects
        obj_grasp_idcs {list} : List of ints indicating to which object some grasps belong to.
    """

    # Load the data
    scene_path = os.path.join(data_root_dir, filename)+".npz"
    with np.load(scene_path, allow_pickle=True) as scene_data:

        # Extrude data
        scene_grasps_tf = scene_data["scene_grasps_tf"]
        scene_grasps_scores = scene_data["scene_grasps_scores"]
        object_names = scene_data["object_names"]
        obj_transforms = scene_data["obj_transforms"]
        obj_grasp_idcs = scene_data["obj_grasp_idcs"]

        return scene_grasps_tf, scene_grasps_scores, object_names, obj_transforms, obj_grasp_idcs
