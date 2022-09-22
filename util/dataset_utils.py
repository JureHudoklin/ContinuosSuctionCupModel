
import os
import glob

import numpy as np
import pickle
import trimesh

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
            - "tf"  : np.array() [num_grasps, 4, 4] 
                4x4 transformation matrices of the object grasps
            - "score" : np.array() [num_grasps]
                The score of each grasp
    """
    grasp_path_list = glob.glob(os.path.join(
        grasp_root_dir, obj_name)+filetype)
    if len(grasp_path_list) > 0:
        grasp_data = open_grasp(grasp_root_dir, obj_name)
        return grasp_data
    else:
        return None


def get_data_paths(data_root_path):
    """Return the path to test and train folder for meshes and grasps.

    Parameters
    ----------
    data_root_path : str
        Path to the root directory of the dataset.

    Returns
    -------
    meshes_paths : dict
        - "train" : Path to the train split folder for meshes.
        - "test" : Path to the test split folder for meshes.
    grasps_paths : dict
        - "train" : Path to the train split folder for grasps.
        - "test" : Path to the test split folder for grasps.
    """
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

def get_eval_obj_names(root_path):
    """Get the names of all evaluation objects in the dataset.

    Parameters
    ----------
    root_path : str
        Path to the root directory of the dataset.

    Returns
    -------
    names_dict : dict
        - "train" : List of objects in the train split.
        - "test" : List of objects in the test split.
    """
    names_dict = {}
    split_paths = glob.glob(os.path.join(root_path, "grasps/*"))

    for split in split_paths:
        obj_category = os.path.basename(split)
        test_train = glob.glob(f"{split}/*")
        names_dict[obj_category] = [os.path.splitext(os.path.basename(obj_p))[
                                    0] for obj_p in test_train]

    return names_dict

def get_meshes_names(root_path):
    """Get the names of all meshes in the dataset.

    Parameters
    ----------
    root_path : str
        Path to the root directory of the dataset.

    Returns
    -------
    names_dict : dict
        - "train" : List of objects in the train split.
        - "test" : List of objects in the test split.
    """
    names_dict = {}
    split_paths = glob.glob(os.path.join(root_path, "meshes/*"))

    for split in split_paths:
        obj_category = os.path.basename(split)
        test_train = glob.glob(f"{split}/*")
        names_dict[obj_category] = [os.path.splitext(os.path.basename(obj_p))[
                                    0] for obj_p in test_train]

    return names_dict

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


def load_scene_data(filename, data_root_dir):
    """
    Load an already existing 3d scene.
    
    Args:
    --------------
    filename (str): name of the scene to load (withoud file ending).
    data_root_dir (str): Absolute path to where the scene is located
        
    Returns: 
    --------------
    scene_grasps_tf : np.array() [num_grasps, 4, 4]
        4x4 transformation matrices of all the grasps in the scene
    scene_grasps_scores : np.array() [num_grasps]
        The score of each grasp in the scene
    object_names : list of str
        The names of the objects in the scene
    obj_transforms : np.array() [num_objects, 4, 4]
        4x4 transformation matrices for placing each scene objects on the "table"
    obj_grasp_idcs : list -- len(list) = num_objects 
        List of ints indicating to which object grasps belong to.
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
