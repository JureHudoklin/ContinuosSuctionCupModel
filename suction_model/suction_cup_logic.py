#!/usr/bin/env python3

import os
import sys
import trimesh
import time
import numpy as np
import scipy.spatial as ss

from scipy.spatial import cKDTree
from trimeshVisualize import Scene


import suction_cup_lib as sclib
import suction_cup_functions as scf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import utils

# BELOW ARE THE FUNCTIONS FOR EVALUATIONG THE WHOLE MODEL
class EvaluateMC():
    def __init__(self,
                 obj_model,
                 n_processors=8,
                 number_of_points=500,
                 point_sample_radius=2.5,
                 multiprocessing=True,
                 neighboor_average=5,
                 noise_samples=10,
                 noise_cone_angle=0.2):
        """
        Class used to evaluate grasps on an object
        --------------
        - Args:
            - obj_model {ModelData} 
                Object that contains the mesh and the parameters for the "Continious Suction Cup Model"
        - KWargs:
            - n_processors {int} : (default=8)
                Number of threads we will use.
            - number_of_points {int} : (default=500)
                Number of points we will sample on the object.
            - point_sample_radius {float} : (default=2.5)
                How close can the sampled points be to eachother.
            - multiprocessing {bool} : (default=True)
                Whether we will use multiprocessing or not.
            - neighboor_average {int} : (default=5)
                Average the score of neighboring N points
            - noise_samples {int} : (default=10)
                When evaluating the seal sample N approach vectors and evaluate binnary score for each sample.
            - noise_cone_angle {float} : (default=0.2)
                Angle of the cone used to sample the approach vectors.

        """
        # Store set up parameters for
        self.obj_model = obj_model
        self.n_processors = n_processors
        self._number_of_points = number_of_points
        self._point_sample_radius = point_sample_radius
        self._multiprocessing = multiprocessing
        self._neighboor_average = neighboor_average
        self._noise_samples = noise_samples
        self._noise_cone_angle = noise_cone_angle

    def evaluate_one_point_MP(self, inp_vec, evaluate_for_force=True):
        """
        Evaluate one point on the object
        --------------
        inp_vec : (np.array(3,), int)
            Vector containing the contact point and face ID.
        evaluate_for_force : Bool (default = True)
            Perform force evaluation or just seal evaluation.
        --------------
        evaluation : [np.array(n,3), np.array(n,3), bool, float]
            list containing (p_0, p_0_n, contact_score, force_score)
        """
        con_p = inp_vec[0]  # sample_point
        con_n = self.obj_model.mesh.face_normals[inp_vec[1]]

        suction_contact, contact_score, a_v = contact_test_seal(
            con_p, self.obj_model, noise_samples=self._noise_samples, noise_cone_angle=self._noise_cone_angle)

        if contact_score == -1:
            return [suction_contact.p_0, suction_contact.p_0_n, a_v, 0., 0.]
            
        if contact_score > 0:
            if evaluate_for_force:
                force_score = contact_test_forces(
                    suction_contact, self.obj_model)
            else:
                force_score = 1
            return [suction_contact.p_0, suction_contact.p_0_n, a_v, contact_score,  force_score]
        else:
            return [suction_contact.p_0, suction_contact.p_0_n, a_v, contact_score,  0.]

    def evaluate_model(self, display = False):
        start = time.time()

        # Sample points:
        samples, face_id = trimesh.sample.sample_surface_even(
            self.obj_model.mesh, self._number_of_points, self._point_sample_radius)

        if self._multiprocessing:
            out = utils.run_multiprocessing(self.evaluate_one_point_MP,
                                            zip(samples, face_id), self.n_processors)
        else:
            out = []
            for i, test_sample in enumerate(zip(samples, face_id)):
                out.append(self.evaluate_one_point_MP(test_sample))
                if i % 100 == 0:
                    print("Sameple: ", i)

        # Output performance
        print("Input length: {}".format(len(samples)))
        print("Output length: {}".format(len(out)))
        print("Multiprocessing time: {}mins".format((time.time()-start)/60))
        print("Multiprocessing time: {}secs".format((time.time()-start)))

        out_formated, out_formated_old = self._format_output(out)

        if display:
            print("Displaying scene")
            my_scene = Scene()
            my_scene.plot_point_multiple(samples, radius=1)
            my_scene.plot_mesh(self.obj_model.mesh)
            my_scene.plot_grasp(out_formated["tf"], out_formated["scores"])
            my_scene.display()

        return out_formated

    def _format_output(self, out):
        """
        Format the output from the evaluation
        --------------
        - out {list}
            -Output must have the following format:
            -[p_0, p_0_n, a_v, contact_success,  force_score]
        -----------
        - Returns:
            - formated_grasps {dict}
                - {"tf": [np.array(4, 4), ...], "scores": [float, ...]}
                - Output formated and converted into grasp tf's with scores
            - formated_output {dict}
                - {"p_0": [np.array(3,), ...],
                   "p_0_n": [np.array(3,), ...],
                   "a_v": [np.array(3,), ...],
                   "score_seal": [bool, ...],
                   "score_force": [float, ...],
                   "score_total": [float, ...]
                   }
                - Less formated output. More usefull for debugging
        """
        # Covert to numpy array of objects
        out = np.array(out, dtype=object)
        # Prepare a dict where we will store the formated output:
        final_output = {}
        ##########
        # STEP 1 #
        ##########

        # Remove all entries where contact_success is -1 (could not calculate something)
        out_formated = np.copy(out[out[:, 3] != -1])
        # Split out the points
        final_output["p_0"] = np.array(out_formated[:, 0])
        # Split out the normals
        final_output["p_0_n"] = np.array(out_formated[:, 1])
        # Split out the approach vector
        final_output["a_v"] = np.array(out_formated[:, 2])
        # Split out the seal score
        final_output["score_seal"] = np.array(out_formated[:, 3])
        # Split out the force score
        if out_formated[np.argmax(out_formated[:, 4]), 4] != 0:
            final_output["score_force"] = np.array(
                out_formated[:, 4]) / out_formated[np.argmax(out_formated[:, 4]),4]
        else:
            final_output["score_force"] = np.zeros(out_formated[:, 4].shape)

        ##########
        # STEP 2 #
        ##########

        point_total_score = final_output["score_force"] * \
            final_output["score_seal"]
        final_output["score_total"] = point_total_score


        if self._neighboor_average > 1:
            kdtree = cKDTree(list(final_output["p_0"]))
            dist, points = kdtree.query(
                list(final_output["p_0"]), self._neighboor_average)
            point_total_score = np.where(point_total_score[np.array(points)[:,0]] > 0, np.average(
                point_total_score[np.array(points)], axis=1), 0)
                
        formated_grasps = {"tf":[], "scores":[]}
        
        for i, pnt_score in enumerate(point_total_score):
            if pnt_score > 0:
                if type(final_output["p_0"][i]) == None:
                    continue 
                grasp_tf = trimesh.geometry.align_vectors(
                    np.array([0, 0, -1]), final_output["a_v"][i])
                grasp_tf[0:3, 3] = final_output["p_0"][i]
                formated_grasps["tf"].append(grasp_tf)
                formated_grasps["scores"].append(pnt_score)

        formated_grasps["tf"] = np.array(formated_grasps["tf"])
        formated_grasps["scores"] = np.array(formated_grasps["scores"])
        return formated_grasps, final_output


def suction_force_limits(file_loc, con_p, force_direction = None, vac_min = 0.020, vac_max = 0.065, increment = 0.005, n = 30):

    obj_model = sclib.ModelData(file_loc)
    # Initiate contact
    suction_contact = sclib.SuctionContact(con_p)

    # Form seal
    suction_contact.form_seal(obj_model)
    if suction_contact.success == False:
        raise Exception("Failed to get the perimeter")

    a_v = -suction_contact.average_normal

    if force_direction is None:
        force_direction = a_v
    
    contact_success = suction_contact.evaluate_contact(a_v, obj_model)
    vacuums = np.arange(vac_min, vac_max, increment)

    results = []

    for vacuum_level in vacuums:
        contact_success = True
        force = 0
        while contact_success:
            contact_success = suction_contact.evaluate_forces(suction_contact.p_0, a_v*force, np.array([0,0,0]), vacuum_level, obj_model, a_v, in_current_configuration = False, simulate_object_rotation = True)

            force += 0.2

        results.append((vacuum_level, force))

    return np.array(results)

def contact_test_forces(suction_contact, obj_model, vac_level=0.07, **kwargs):
    """
    Test the given suction contact for resistance to external forces. The score is the volume of "wrench space"
    --------------
    - Args:
        - suction_contact {SuctionContact}
            - A suction contact that has a seal already formed
        - obj_model {ModelData()}
            - Data about our cup, gripper, ...
    - Kwargs:
        - vac_level {float} : Default=0.07 kPa
            - The vacuum level for which to test the contact
        - kwargs : {"a_v":np.array(3,)}
            - "a_v" specfic approach vector for a contact. If none is given -average_normal is used.
    --------------
    - Returns:
        - convex_hull.volume {float}
            - The volume of the convex hull
    """
    cog = obj_model.mesh.center_mass

    force_location = cog
    external_moment = np.array([0, 0, 0])

    if "a_v" in kwargs:
        a_v = kwargs["a_v"]
    else:
        a_v = -suction_contact.average_normal

    force_0 = (cog-suction_contact.p_0) / \
        np.linalg.norm(cog-suction_contact.p_0)
    force_directions = scf.create_half_sphere()
    R_mat = trimesh.geometry.align_vectors(np.array([0, 0, 1]), a_v)
    force_directions = np.dot(R_mat[0:3, 0:3], force_directions.T).T
    results = []

    for force in force_directions:
        i = 18
        success_prev = suction_contact.evaluate_forces(
            force_location, force*i, external_moment, vac_level, obj_model, a_v)
        success = success_prev

        while success == success_prev:
            if success:
                i += 2
            else:
                i -= 2

            if (i > 80):
                print(i, "KAY SE DOGAJA")

                return 0.
            elif (i < 0):
                return 0.
            success_prev = success
            success = suction_contact.evaluate_forces(
                force_location, force*i, external_moment, vac_level, obj_model, a_v)
 
        results.append(i*force)

    convex_hull = ss.ConvexHull(np.array(results))

    return convex_hull.volume


def contact_test_seal(con_p, obj_model, a_v=None, noise_samples=5, noise_cone_angle=0.1):
    """
    Tests whether the given contact point can form a seal or not.
    --------------
    - Arguments:
        - n {int}
            - Amount of points we want / 2
        - con_p {np.array(3,)}
            - Point on the model we want to contact
        - obj_model {ModelData}
            - Object containing model parameters   
    - Kwargs:
        - a_v {np.array(3,)} : Default = None
            - The approach vector for the contact. If none is given -average_normal is used.
        - noise_sample {int} : Default = 5 
            - For how many different vectors with added noise to test the point. If 0 no noise is added
        - noise_cone_angle {float} : Default = 0.1
            - The max angle between the normal and the noise vector.
    --------------
    - Return:
        - suction_contact : SuctionContact()
            Returned suction contact object
        - contact_success : float
            The seal score for the given contact: -1 in case of failure; [0,1] valid seal score.
        - a_v {np.array(3,)}
            - The approach vector for which the contact was tested
    """
    
    # Initiate contact
    suction_contact = sclib.SuctionContact(con_p)
    # Form seal
    suction_contact.form_seal(obj_model)
    if suction_contact.success == False:
        return suction_contact, -1, None
    else:
        mean_point = np.mean(suction_contact.perimiter, axis = 0)
        delta_dist = np.abs(np.linalg.norm(mean_point - suction_contact.p_0))
        if delta_dist > 13:
            return suction_contact, 0, -suction_contact.average_normal

    # Add noise if desired
    if noise_samples == 0:
        if a_v is None:
            a_v = -suction_contact.average_normal
        suction_contact, contact_success = _test_for_seal(
            suction_contact, a_v, obj_model)
        if contact_success:
            return suction_contact, 1., a_v
        else:
            return suction_contact, 0., a_v
    else:
        if a_v is None:
            a_v = scf.vector_with_noise(-suction_contact.average_normal, noise_cone_angle)

        success_count = 0
        for i in range(noise_samples):
            a_v_noise = scf.vector_with_noise(a_v, noise_cone_angle)
            suction_contact, contact_success = _test_for_seal(
                suction_contact, a_v_noise, obj_model)
            if contact_success:
                success_count += 1

        return suction_contact, success_count/noise_samples, a_v

def _test_for_seal(suction_contact, a_v, obj_model):
    # Analyze the seal and return the results
    if suction_contact.success == True:
        contact_success = suction_contact.evaluate_contact(a_v, obj_model)
        return suction_contact, contact_success
    else:
        return suction_contact, False
