#!/usr/bin/env python3

import os
import sys
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from multiprocessing import freeze_support
from os import close, path, walk

import suction_model.suction_cup_logic as scl
import suction_model.suction_cup_lib as sclib
from trimeshVisualize import Scene

from util.dataset_utils import load_grasp, load_mesh


def evaluate_object_set(root_dir,
                        config_path = None,
                        number_of_points=3000,
                        splits="test",
                        save=True,
                        overwrite=False,
                        display=False,
                        n_processors=8):

    """
    Evaluate all objects in the dataset.
    ---------
    Args:
        - root_dir {str} : The root directory of the dataset.
            Must Contain:
                - meshes  -->  meshes/<splits>
    Kwargs:
        - number_of_points {int} : The number of points to evaluate on each mesh.
        - config_path {str} : Path to the model config file.
        - splits {str} : The name of subdirectory in meshes/ to evaluate. (Used for train/test split)
        - save {bool} : Whether to save the results.
            Results are saved as --> grasps/<splits>/<obj_name>.pkl
        - overwrite {bool} : Whether to overwrite existing results.
        - display {bool} : Whether to display the results.
        - n_processors {int} : Multiprocessing: number of processors.
    """

    start_time = time.time()

    path_data = root_dir
    path_meshes = os.path.join(path_data, f"meshes/{splits}")
    # Check if path_meshes exists
    if not os.path.exists(path_meshes):
        # Create the directory
        os.mkdir(path_meshes)
        
    path_grasps = os.path.join(path_data, f"grasps/{splits}")
    # Check if path_grasps exists
    if not os.path.exists(path_grasps):
        # Create the directory
        os.mkdir(path_grasps)

    # Check for preexisting grasps
    filenames = next(walk(path_meshes),(None, None, []))[2]  # [] if no file
    generated_grasps = next(walk(path_grasps), (None, None, []))[2]
    generated_grasps_names = [os.path.splitext(os.path.basename(obj_p))[
        0] for obj_p in generated_grasps]

    step = 0
    invalid_objects = []

    for step, file_loc in enumerate(filenames):

        # Get the object name
        obj_name = file_loc.split(".",20)[0]
        print("---------------------------") 
        print(f"Computing object {obj_name}  number:", step)

        if (obj_name in generated_grasps_names) and not overwrite:
            print(f"Object {obj_name} already evaluated: CONTINUING")
            continue

        # Set the path and evaluate the object
        file_loc = os.path.join(path_meshes, obj_name) + ".obj"
        obj_model = sclib.ModelData(file_loc, config_path, units=("millimeters", "millimeters"), subdivide=True)

        evaluation_object = scl.EvaluateMC(
            obj_model, n_processors=n_processors, number_of_points=number_of_points)
        try:
            results = evaluation_object.evaluate_model(display=display)
        except:
            print("Object evaluation failed: CONTINUING")
            invalid_objects.append(obj_name)
            continue

        # Save the result
        if save:
            a_file = open(os.path.join(os.path.join(path_data, f"grasps/{splits}"), obj_name) + ".pkl", "w+b")
            pickle.dump(results, a_file)
            a_file.close()

    print("Elapsed Time:" + str(time.time() - start_time))
    print("Objects that could not be evaluated:", invalid_objects)


def evaluate_object_one(obj_name,
                        root_dir,
                        config_path = None,
                        number_of_points = 3000,
                        display=True,
                        splits="test",
                        save=False):

    """
    Evalute N points on a single object
    ---------
    Args:
        - obj_name {str} : The name of the object to evaluate.
        - root_dir {str} : The path to root_dir of data.
    Kwargs:
        - config_path {str} : The path to the model config file.
        - number_of_points {int} : The number of points to evaluate on the object.
        - display {bool} : Whether to display the results.
        - splits {str} : The name of subdirectory in meshes/ to evaluate. (Used for train/test split)
        - save_path {str} : The path to save the results.
            if None, results are not saved.
    ----------
    Returns:
        - results {dict} : The results of the evaluation.
    """

    start_time = time.time()

    print(f"-----Evaluating {obj_name}----")
    file_loc = os.path.join(root_dir, f"meshes/{splits}/{obj_name}.obj")
    obj_model = sclib.ModelData(file_loc, config_path, units=("millimeters", "millimeters"), subdivide=True)

    evaluation_object = scl.EvaluateMC(
        obj_model, n_processors=8, number_of_points=number_of_points, multiprocessing=True)

    results = evaluation_object.evaluate_model(display=display)

    # Save the result
    if save is True:
        a_file = open(os.path.join(os.path.join(
            root_dir, f"grasps/{splits}"), obj_name) + ".pkl", "w+b")
        pickle.dump(results, a_file)
        a_file.close()

    print("Elapsed Time:" + str(time.time() - start_time))
    print("-------------------------------")

    return results

def open_saved_grasp(obj_name, root_dir, splits="test", display = False):
    """
    Open an already evaluated object.
    ---------
    Args:
        - obj_name {str} : The name of the object to evaluate.
        - obj_path {str} : The path to the object to evaluate.
    Kwargs:
        - splits {str} : The name of subdirectory where the mesh/grasp is saved.
        - display {bool} : Whether to display the results.
    ----------
    Returns:
        - results {dict} : The results of the evaluation.
    """

    path_meshes = os.path.join(root_dir, f"meshes/{splits}")
    path_grasps = os.path.join(root_dir, f"grasps/{splits}")

    grasp = load_grasp(obj_name, path_grasps)

    if display:
        my_scene = Scene()
        my_scene.plot_mesh(load_mesh(obj_name, path_meshes, scale = 1))
        my_scene.plot_grasp(grasp["tf"], grasp["scores"])
        my_scene.display()

    return grasp


def test_contact_point(file_loc, test_point=np.array([0, 0, 0]), display_contact=True):
    """
    A function to which performs a full analysis of a single contact point.
    ---------
    Args:
        - file_loc {str} : Absolute path .obj mesh which we want to evaluate
    """

    obj_model = sclib.ModelData(file_loc, subdivide = True)
    
    #Initiate contact
    suction_contact = sclib.SuctionContact(test_point)
    suction_contact.form_seal(obj_model)

    # Create seal and evaluate it
    seal_success = suction_contact.evaluate_contact(-suction_contact.average_normal, obj_model, debug_display=True)
    print("Successfully created seal with object:", seal_success)

    # Test the maximum forces that the seal can withstand
    results = scl.suction_force_limits(file_loc, test_point)
    plt.plot(results[:, 0]*1000, results[:, 1])
    plt.xlabel("Vacuum [%]")
    plt.ylabel("Force [N]")
    plt.show()

    # Get Seal Score
    suction_contact, seal_score, a_v = scl.contact_test_seal(test_point, obj_model, noise_samples=0)
    print("Seal Score:", seal_score)

    # Get Force Score
    force_score = scl.contact_test_forces(suction_contact, obj_model, vac_level=0.07)
    print("Force Score:", force_score)

    # Plot the contact
    if display_contact:      
        my_scene = Scene()
        my_scene.plot_coordinate_system()
        my_scene.plot_mesh(obj_model.mesh)
        my_scene.plot_point_cloud(suction_contact.perimiter, radius=2)
        my_scene.plot_point(suction_contact.p_0, radius=2)
        my_scene.display()



if __name__ == "__main__":
    #freeze_support()
    file_loc = "/home/jure/programming/SuctionCupModel/data/meshes/train/A01_0.obj"
    model_config = "/home/jure/programming/ContinuosSuctionCupModel/suction_model/configs/sm_30_config.yml"

    data_root_dir = "/home/jure/programming/SuctionCupModel/data"

    if True:
        # Open an already evaluated object
        grasp = open_saved_grasp("K21_0", root_dir=data_root_dir, splits="train", display=True)
        print(grasp["tf"].shape)

    if False:
        # Test one point on a particular object
        file_loc = "/home/jure/programming/SuctionCupModel/data/meshes/train/A01_0.obj"
        test_contact_point(file_loc, test_point=np.array([10, 0, 30]), display_contact=True)

    if False:
        # Evaluate a particular object #W02_0
        evaluate_object_one("H05_2",
                            config_path=model_config,
                            root_dir=data_root_dir,
                            number_of_points=1000,
                            display=True,
                            splits="train",
                            )

    if False:
        # Evaluate a whole set of objects
        evaluate_object_set(root_dir = data_root_dir, config_path = model_config,  display = True, splits="train", save = False)
