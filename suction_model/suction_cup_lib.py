#!/usr/bin/env python3

from ast import Mod
from logging import raiseExceptions
import numpy as np
import trimesh
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import potpourri3d as pp3d
import yaml
import os
import inspect

from scipy.interpolate import splprep, splev
from scipy import interpolate
from scipy import optimize
from scipy.integrate import trapz
from numpy import gradient

import suction_cup_functions as scf
from trimeshVisualize import Scene

# Params
# --------------------------------------------------
# 40mm cup: r=17.6, d_0=60,k_bm = 11*10**-3, k_def = 2.9892, coef_friction = 0.5, lip_area = 450, per_smoothing = 0, k_n smoothing = 1,moment_multiplicator = 4,


class ModelData():

    r = 17.6
    d_0 = 60
    max_deformation = 10
    k_bm = 11*10**-3  # Coefficient for curvature
    k_def = 2.9892 # Coefficient for deformation pressure
    coef_friction = 0.5
    lip_area = 450
    perimiter_smoothing = 1
    per_points = 60

    config_dict = {
        "r": r,
        "d_0": d_0,
        "max_deformation": max_deformation,
        "k_bm": k_bm,
        "k_def": k_def,
        "coef_friction": coef_friction,
        "lip_area": lip_area,
        "perimiter_smoothing": perimiter_smoothing,
        "per_points": per_points
    }

    def __init__(self, mesh_location, load_path = None, *args, **kwargs):
        self.mesh = trimesh.load(mesh_location, force='mesh')
        if "units" in kwargs:   # Convert mesh to correct units
            self.mesh.units = kwargs["units"][0]
            self.mesh.convert_units(kwargs["units"][1])
        if "subdivide" in kwargs:   # Subdivide the mesh
            if kwargs["subdivide"] == True:
                self.mesh = self.mesh.subdivide()
        
        if load_path is not None:
            self.load_config(load_path)

        self.intersection_mesh = trimesh.ray.ray_triangle.RayMeshIntersector(
            self.mesh)
        self.proximity_mesh = trimesh.proximity.ProximityQuery(self.mesh)

        self.samples, self.faces = trimesh.sample.sample_surface(
            self.mesh, 30000)
        self.heatmap = False

    def save_config(self, path, name):

        with open(os.path.join(path, name) + ".yml", "w") as yaml_file:
            yaml.dump(self.config_dict, yaml_file, default_flow_style=False)
        print("Saved config:")
        print(self.config_dict)

    def load_config(self, path):
        with open(path, "r") as yaml_file:
            config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
            for attr in list(config_dict.keys()):
                if not attr.startswith("__"):
                    setattr(ModelData, attr, config_dict[attr])
  
        print("Loaded config:")
        print(config_dict)

    def create_heat_map(self):
        self.pp3d_solver = pp3d.PointCloudHeatSolver(self.samples)
        self.tangent_frames = self.pp3d_solver.get_tangent_frames()
        self.heatmap = True

    def model_transform(self, tf):
        self.mesh = self.mesh.apply_transform(tf)
        self.intersection_mesh = trimesh.ray.ray_triangle.RayMeshIntersector(
            self.mesh)
        self.proximity_mesh = trimesh.proximity.ProximityQuery(self.mesh)


class SuctionContact():
    def __init__(self, con_p):
        """
        Suction contact class. It can perform analyze forces and seal for a contact point on a step model.
        --------------
        n : int
            - Amount of points we want use to approximate the perimeter.
        con_p : np.array(3,)
            - Point on the model we want to contact
        Returns
        --------------
        """
        self.con_p = con_p

    def form_seal(self, model):
        if model.heatmap == False:
            model.create_heat_map()

        # try:
        per_success = self.get_perimiter(model)
        # except:
        #per_success = False
        if per_success:
            self.success = True
            self.calculate_along_path()
            self.tangent, self.radial = self.calculate_tangents(model)
            self.calculate_average_normal()
        else:
            self.success = False

    def get_perimiter(self, model, max_deviation=2, min_prox=1):
        """
        Returns a set of points whose geodesic distance from self.con_p is equal to model.r
        --------------
        model : ModelData
        max_deviation : float
            In what range (r, r+max_deviation) we search for perimiter points
        min_prox : float
            A filter parameter for searching the points. We discard the found points that are inside the radius = min_prox
        Returns
        --------------
        suceess : bool
            True if the perimiter was successfully found and false otherwise
        """

        # Extract faces and samples
        faces = model.faces
        samples = model.samples

        # Get the p_0 and p_0_n by searching for the closest point on the mesh
        self.p_0_i = scf.closest_node(self.con_p, samples, 1)
        self.p_0 = samples[self.p_0_i]
        cloasest, distance, triangle_id = model.proximity_mesh.on_surface([
                                                                          self.p_0])
        self.p_0_n = model.mesh.face_normals[triangle_id[0]]

        # Calculate distance from p_0 to all other points
        dist = model.pp3d_solver.compute_distance(self.p_0_i)

        # Set min and max perimiter where to search for points
        perimeter_min = model.r - max_deviation
        perimeter_max = model.r + max_deviation

        # Mask out the points of interest
        smaller = np.where(dist < perimeter_max)
        bigger = np.where(dist > perimeter_min)
        intersection = np.intersect1d(smaller, bigger)

        points_filtered = np.array(samples[intersection])
        if len(points_filtered) == 0:
            return False
            # raise Exception(
            #     "No points found in the interest area. Probably an invalid object.")

        # Fry to move the points to be at right distance.
        # To do that move them along the tangent at their location
        v_per_p0 = self.p_0-points_filtered
        # Get normals at edge points
        normals_at_per = model.tangent_frames[2][intersection]
        # Projection of v_per_p0 to the tangent plane
        temp = (v_per_p0*normals_at_per).sum(1)
        v_tangent = v_per_p0-temp[:, np.newaxis]*normals_at_per
        v_tangent = scf.unit_array_of_vectors(v_tangent)
        # How far are the points from the desired location
        points_deviation = dist[intersection]-model.r
        v_tangent = v_tangent*points_deviation[:, np.newaxis]
        # Move the points
        points_filtered = points_filtered + v_tangent
        # Move the points to the surface
        points_filtered, distance, faces_filtered = model.proximity_mesh.on_surface(
            points_filtered)
        # Filter out the points that are of no interest
        points_filtered, mask = trimesh.points.remove_close(
            points_filtered, radius=min_prox)

        # Order the points
        # METHOD 1: use Traveling Salesman Problem. NOT ALWAYS CORRECT
        order_of_points, dist_eucl = trimesh.points.tsp(
            points_filtered, start=0)
        #order_of_points = np.append(order_of_points, order_of_points[0])
        # METHOD 2: WITH POLAR COORDINATES
        # First calculate the average normal of all points
        normals_at_per = np.array(model.mesh.face_normals[faces_filtered])
        average_normal_approx = scf.unit_vector(
            np.average(normals_at_per, axis=0))
        self.average_normal_approx = average_normal_approx
        #order_of_points = scf.radial_sort(points_filtered, self.p_0, average_normal_approx)
        #order_of_points = np.append(order_of_points, order_of_points[0])

        # Get the filtered points and faces
        self.perimiter = np.array(points_filtered[order_of_points])
        if len(self.perimiter) < 5:
            return False

        self.normal = self._get_normals(
            self.perimiter, model)
        self.calculate_along_path()
        self.tangent, self.radial = self.calculate_tangents(model)

        for _ in range(4):
            test_dir = (
                self.radial*scf.unit_array_of_vectors(self.p_0-self.perimiter)).sum(1)
            if not (test_dir > 0).all():
                if len(self.perimiter) < 5:
                    return False
                majority = np.average(test_dir)
                if majority < 0:
                    self.perimiter = np.flip(self.perimiter, 0)
                    self.normal = self._get_normals(
                        self.perimiter, model)
                    self.calculate_along_path()
                    self.tangent, self.radial = self.calculate_tangents(
                        model)
                else:
                    # Select the point to delete
                    tangent_projections = np.dot(
                        self.tangent, np.roll(self.tangent, -1, axis=0).T)
                    tangent_projections = np.diagonal(tangent_projections)
                    min_1, min_2 = np.argpartition(test_dir, 2)[:2]

                    if min_1 > min_2:
                        min_1, min_2 = min_2, min_1

                    self.perimiter = np.delete(self.perimiter, min_1, axis=0)
                    self.normal = np.delete(self.normal, min_1, axis=0)
                    self.calculate_along_path()
                    self.tangent, self.radial = self.calculate_tangents(model)

                    # Rerun tsp
                    order_of_points, dist_eucl = trimesh.points.tsp(self.perimiter, start=0)
                    self.perimiter = np.array(self.perimiter[order_of_points])

            else:
                break
        else:
            #print("Can not obtain a valid perimiter for the point.")
            return False


        # Interpolate Perimiter
        self.perimiter = np.append(
            self.perimiter, self.perimiter[np.newaxis, 0, :], axis=0)

        self.perimiter_unfiltered = self.perimiter
        if len(self.perimiter_unfiltered) < 5:
            return False
        per_func = self.interpolate_perimiter(self.perimiter, model.perimiter_smoothing)
        u = np.linspace(0, 1, model.per_points)
        perimiter_pnts = per_func(u)

        # Get the normals
        self.normal = self._get_normals(
            perimiter_pnts, model)
        self.perimiter = np.array(perimiter_pnts)
        self.calculate_along_path()
        self.tangent, self.radial = self.calculate_tangents(model)

        return True

    def _get_normals(self, points, model, use_barycentric=True):
        """
        We calculate the normals to the surface for the given points and mesh(model).
        --------------
        points : np.array(n,3)
            Points we need the normals of
        model : ModelData
        use_barycentric : bool
            If true we will use the barycentric coordinates of the point on the triangle to approximate the normal.
            If false the mesh triangle normal is used.
        Returns
        --------------
        None : None
        """
        cloasest, distance, triangle_id = model.proximity_mesh.on_surface(
            points)

        if use_barycentric:
            # Normal at final point
            bary = trimesh.triangles.points_to_barycentric(
                model.mesh.triangles[triangle_id], cloasest, method='cramer')
            bary = np.array(bary)
            face_points = model.mesh.faces[triangle_id]
            normals = np.asarray(model.mesh.vertex_normals[face_points])

            normal = (bary[:, :, np.newaxis]*normals).sum(1)
            normal = scf.unit_array_of_vectors(normal)

        else:
            normal = model.mesh.face_normals[triangle_id]
        return np.array(normal)

    def calculate_along_path(self):
        """
        Cahches the u and du functions around the perimeter of contact
        --------------
        locations : (n, 3) float
        Ordered locations of contact points
        Returns
        --------------
        """
        locations = self.perimiter
        distance = np.roll(locations, -1, axis=0) - locations
        distance = np.linalg.norm(distance, axis=1)
        distance_sum = np.sum(distance)
        self.du = distance/distance_sum
        self.du_cumulative = np.cumsum(np.append(0, self.du[:-1]))

    def calculate_tangents(self, model):
        """
        From locations of points and normal to the surface tangent to the path and normal to the path are calculated
        --------------
        locations : (n, 3) float
        Ordered array of contact locations
        normal : (n, 3) float
        An array of normals to the contact locations
        Returns
        --------------
        tangent : (n, 3) float
        An array of tangets to the path along contact locations
        radial : (n, 3) float
        A cross product of normal and tangent
        """
        # Calculate along path must be run before this function to cache u and du

        locations = self.perimiter
        normal = self.normal
        tangent = gradient(locations, self.du_cumulative, axis=0, edge_order=2)
        norm = np.linalg.norm(tangent, axis=1)

        tangent[:, 0] = np.divide(
            tangent[:, 0], norm)
        tangent[:, 1] = np.divide(
            tangent[:, 1], norm)
        tangent[:, 2] = np.divide(
            tangent[:, 2], norm)

        radial = np.cross(normal, tangent)

        return tangent, radial

    def interpolate_perimiter(self, perimiter_pnts, smoothing):
        x = perimiter_pnts[:, 0]
        y = perimiter_pnts[:, 1]
        z = perimiter_pnts[:, 2]

        tck_per, u = splprep([x, y, z], s=smoothing, per=True, k=3)

        def tck_per_func(u):
            new_p = splev(u, tck_per)
            return np.transpose(new_p)

        return tck_per_func

    def interpolate_normal(self, normal_pnts, smoothing):
        n_x = normal_pnts[:, 0]
        n_y = normal_pnts[:, 1]
        n_z = normal_pnts[:, 2]

        tck_nor, u = splprep([n_x, n_y, n_z], s=smoothing, per=True)

        def tck_nor_func(u):
            new_n = splev(u, tck_nor)
            new_n = new_n/np.linalg.norm(new_n, axis=0, keepdims=True)
            return np.transpose(new_n)

        return tck_nor_func

    def interpolate_tangent(self, tangent_pnts, smoothing):
        t_x = tangent_pnts[:, 0]
        t_y = tangent_pnts[:, 1]
        t_z = tangent_pnts[:, 2]

        tck_x = scipy.interpolate.splrep(
            self.du_cumulative, t_x, s=smoothing, per=True)
        tck_y = scipy.interpolate.splrep(
            self.du_cumulative, t_y, s=smoothing, per=True)
        tck_z = scipy.interpolate.splrep(
            self.du_cumulative, t_z, s=smoothing, per=True)

        def tck_nor_func(u):
            new_n_x = splev(u, tck_x)
            new_n_y = splev(u, tck_y)
            new_n_z = splev(u, tck_z)
            new_n = np.array([new_n_x, new_n_y, new_n_z]).T
            temp = np.linalg.norm(new_n, axis=1)
            new_n = new_n/np.linalg.norm(new_n, axis=1, keepdims=True)
            return new_n

        return tck_nor_func

    def calculate_average_normal(self):
        average_normal_x = trapz(
            self.normal[:, 0], self.du_cumulative)
        average_normal_y = trapz(
            self.normal[:, 1], self.du_cumulative)
        average_normal_z = trapz(
            self.normal[:, 2], self.du_cumulative)
        average_normal = np.array(
            [average_normal_x, average_normal_y, average_normal_z])
        average_normal = average_normal/np.linalg.norm(average_normal)

        self.average_normal = average_normal

    def find_apex(self, a_v, model):

        # Find the "lowest perimeter point"
        # p rojection of point to approach normal
        dist = np.dot(self.perimiter, np.transpose(a_v))
        # index of a point that is furthest away
        max_dist_i = int(dist.argmin())

        # We must find root of this function. That we can get poisition of apex

        def find_a(t, approach, point_mi):
            dist = np.linalg.norm(
                self.p_0 - approach*t - point_mi) - (model.d_0-model.max_deformation)
            return dist

        t_0 = optimize.root(find_a, model.d_0, args=(
            a_v, self.perimiter[max_dist_i, :]))

        a = -a_v*t_0.x + self.p_0  # Position of apex

        return a

    def _calculate_deformation(self, a_v, model, per_points):
        # Deformation Using projection on Approach vector
        dist = np.dot(per_points, np.transpose(a_v))
        max_dist_i = int(dist.argmin())
        distance = dist-dist[max_dist_i]-model.max_deformation
        return distance

    def _calculate_deformation_vectors(self, a_v):
        #def_vectors = unit_array_of_vectors(self.apex - self.perimiter)
        #def_vectors = self.normal
        def_vectors = -np.tile(a_v, (np.shape(self.normal)[0], 1))
        return def_vectors

    def evaluate_contact(self, a_v, model, debug_display=False):
        """
        Analyzes if the the suction contact makes a seal for the given approach vector.
        --------------
        a_v: (3, )  np.array
        Direction how the "robot" approaches the contact point. Usually same as -con_n
        obj_model : ModelData class
            class containing the information of the mesh and suction cup properties.

        Returns
        --------------
        Seal success : bool
            True if seal formed and false otherwise.
        """

        apex = self.find_apex(a_v, model)
        self.apex = apex
        # Deformation
        distance = self._calculate_deformation(a_v, model, self.perimiter)
        # Normal Curvature ------------------------------
        dx = 1/np.shape(self.tangent)[0]
        # Filter the input tangent and normal
        tangent_fit = scf.unit_array_of_vectors(
            scf.fourier_fit_3d(self.du_cumulative, self.tangent, 5))
        normal_fit = scf.unit_array_of_vectors(
            scf.fourier_fit_3d(self.du_cumulative, self.normal, 5))

        # Calculate the curvature
        ddg = np.gradient(tangent_fit, dx, axis=0, edge_order=2)
        k_n_rough = (ddg * normal_fit).sum(1)  # Curvature
        # Fit it using a spline to get a nicer periodic function
        try:
            tck = scipy.interpolate.splrep(
                self.du_cumulative, k_n_rough, s=1, per=False, k=3)

        except:
            return False
        k_n = splev(self.du_cumulative, tck)

        # Pressure because of normal curvature
        self.p_bm = np.gradient(np.gradient(model.k_bm * k_n, dx,
                                            edge_order=2), dx, edge_order=2)

        # Presure because of deformation
        def_vectors = self._calculate_deformation_vectors(a_v)

        p_d = (def_vectors.T * distance * model.k_def).T
        # Amount of pressure in direction of normal to surface
        p_d_n = (p_d * normal_fit).sum(1)

        # Analyzing the all pressures
        self.p_all = p_d_n - self.p_bm

        p_max_i = np.argmax(self.p_all)

        # ------- PLOT FOR EASIER DETERMINING OF PARAMETERS ----------
        if debug_display:
            plt.subplot(3, 1, 1)
            plt.plot(self.du_cumulative, self.normal, "r", label="raw [x,y,z]")
            plt.plot(self.du_cumulative, normal_fit,
                     "g", label="fitted [x,y,z]")
            plt.title("Fitting normal using FFT")
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(k_n_rough, "r", label="k_n before smoothing")
            plt.plot(k_n, "g", label="k_n after smoothing")
            plt.title("Smoothing the Curvature")
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(p_d_n, label="normal deformation pressure")
            plt.plot(self.p_bm, label="bending pressure")
            plt.plot(self.p_all, label="all together")
            plt.legend()
            plt.title("Pressure distribution")
            plt.show()
        # -------------------------------------------------------------

        if self.p_all[p_max_i] > 0:
            # Seal was not formed
            return False
        else:
            # Seal was formed
            return True

    def evaluate_forces(self, f_pos, f_dir, m_vec, vacuum, model, a_v, in_current_configuration=False, simulate_object_rotation=True):
        """
        Returns a set of points and normals that are at a distance of r along the surfaceof the mesh
        --------------
        f_pos : position where the force acts on the object. Used to calculate the forces moment: (x,y,z) in mm.
        f_dir : Size of the force in N. (3,) array
        m_vec : numpy array (3,)
        vacuum : pressure difference between inside and outside in N/mm^2 (MPa)
        model : Object model that contains suction cup parameters
        in_current_configuration : Bool
            If True we will take the current rotation as the zero position. That means we the moment applied in this position will be considered as our zero point.
            If False we wont care about the current rotation of the object. The rotation of the object will be adjusted so the moment equals to external applied moment.
        simulate_object_rotation : Bool
            If True the external force will be rotated along with the object.
        --------------
        contact_succes : Returns true if the contact can sustain the force acting on it. False otherwise.
        """
        # Base Coordinate System ----------------------------
        # Set up CS - This are axis of the contact CS written in global coordinates
        dir_t = self.perimiter[0, :] - self.p_0
        z_ax = self.average_normal
        y_ax = np.cross(z_ax, dir_t)
        y_ax = y_ax/np.linalg.norm(y_ax)
        x_ax = np.cross(y_ax, z_ax)

        # Copy of normals, tangents and radial vectors-------
        normal_cp = np.copy(self.normal)
        tangent_cp = np.copy(self.tangent)
        radial_cp = np.copy(self.radial)
        perimiter_cp = np.copy(self.perimiter)

        # Transform force to contact------------------------
        r_m = f_pos - self.p_0  # mm
        m = np.cross(r_m, f_dir) + m_vec  # N*mm

        # Vacuum force ---------------------------------
        proj_points = trimesh.points.project_to_plane(self.perimiter,
                                                      plane_normal=self.average_normal, plane_origin=[
                                                          0, 0, 0],
                                                      transform=None, return_transform=False, return_planar=True)
        # Area
        area = scf.poly_area(proj_points[:, 0], proj_points[:, 1])

        # Compensate for moment ------------------------
        # First move perimiter points to the origin
        T = scf.translation(-self.p_0)
        perimiter_transpozed = np.ones((4, np.shape(self.perimiter)[0]))
        perimiter_transpozed[0:3, :] = perimiter_cp.T
        perimiter_transpozed = np.dot(T, perimiter_transpozed)

        dist = np.dot(perimiter_transpozed[0:3, :].T, np.transpose(a_v))
        offset = dist[int(dist.argmin())]
        distance = dist-offset-model.max_deformation

        # Presure because of deformation
        def_vectors = self._calculate_deformation_vectors(a_v)
        p_d = (def_vectors.T * distance * model.k_def).T

        # Projecting the deformation pressure
        def_n = -(p_d*self.normal).sum(1)
        def_t = -(p_d*self.tangent).sum(1)
        def_r = -(p_d*self.radial).sum(1)

        moment_sum = self.moment_calc(
            def_n, def_t, def_r, normal_cp, tangent_cp, radial_cp, np.transpose(perimiter_transpozed[0:3, :]))
        if in_current_configuration == True:
            m += moment_sum

        # -------------TEMP-----------------
        dir_t = self.perimiter[0, :] - self.p_0
        z_ax_2 = -a_v
        y_ax_2 = np.cross(z_ax_2, dir_t)
        y_ax_2 = y_ax_2/np.linalg.norm(y_ax_2)
        x_ax_2 = np.cross(y_ax_2, z_ax_2)

        f_zzz = self.force_calc(def_n+self.p_bm, def_t, def_r,
                                self.normal, self.tangent, self.radial, z_ax_2)
        f_xxx = self.force_calc(def_n+self.p_bm, def_t, def_r,
                                self.normal, self.tangent, self.radial, x_ax_2)
        f_yyy = self.force_calc(def_n+self.p_bm, def_t, def_r,
                                self.normal, self.tangent, self.radial, y_ax_2)

        self.deformation_force_mom = [
            np.array([f_xxx, f_yyy, f_zzz]), moment_sum]

        # Calculate the main moment axis-----------------
        # First we projct the moment to the contact coordinate System
        m_x = x_ax*(m*x_ax).sum()
        m_y = y_ax*(m*y_ax).sum()
        m_z = z_ax*(m*z_ax).sum()
        # Add together  x and y
        m_xy = m_x + m_y
        rot_modificator = 8000
        rot_sum = np.zeros(3)
        for i in range(50):
            # Calculate moment
            moment_sum = self.moment_calc(
                def_n, def_t, def_r, normal_cp, tangent_cp, radial_cp, np.transpose(perimiter_transpozed[0:3, :]))

            # Project Moment (x is not global x but the "x" of contact CS)
            ma_x = x_ax*(moment_sum*x_ax).sum()
            ma_y = y_ax*(moment_sum*y_ax).sum()
            ma_z = z_ax*(moment_sum*z_ax).sum()
            # Compare with what the moment should be
            ma_xy = ma_x+ma_y
            # We are performing rotation around this axis
            if np.linalg.norm(m_xy-ma_xy) < 0.1:
                break
            m_xy_dir = (m_xy-ma_xy)/np.linalg.norm(m_xy-ma_xy)

            if not np.allclose(ma_xy, m_xy, atol=40, rtol=0.001):

                rot_scale = np.sum(np.abs(ma_xy-m_xy))

                rotation = rot_scale/rot_modificator

                if rot_modificator < 8000:
                    rot_modificator += 1000

                # Rotate everything and try again #rot_scale*0.00007
                rot_sum += m_xy_dir*rotation
                R_mat = scf.rot_axis_angle(m_xy_dir, rotation)
                # Rotate CS
                x_ax = np.dot(R_mat, x_ax)
                y_ax = np.dot(R_mat, y_ax)
                z_ax = np.dot(R_mat, z_ax)
                # Rotate Tangents
                normal_cp = np.dot(R_mat, normal_cp.T).T
                tangent_cp = np.dot(R_mat, tangent_cp.T).T
                radial_cp = np.dot(R_mat, radial_cp.T).T
                # Rotate points
                perimiter_transpozed[0:3, :] = np.dot(
                    R_mat, perimiter_transpozed[0:3, :])
                # Rotate the object and recalculate moment
                if simulate_object_rotation == True:
                    r_m = np.dot(R_mat, r_m)
                    m = np.cross(r_m, f_dir) + m_vec  # N*mm
                    # Calculate the main moment axis-----------------
                    # First we projct the moment to the contact coordinate System
                    m_x = x_ax*(m*x_ax).sum()
                    m_y = y_ax*(m*y_ax).sum()
                    m_z = z_ax*(m*z_ax).sum()
                    # Add together  x and y
                    m_xy = m_x + m_y

                # Recalculate distances
                dist = np.dot(
                    perimiter_transpozed[0:3, :].T, np.transpose(a_v))
                distance = dist-offset-model.max_deformation
                # Presure because of deformation
                # HERE  A KOEFFICIENT 4 FOR DEF. ANY OTHER WAY TO SOLVE THIS?????
                p_d = (def_vectors.T * distance * model.k_def).T
                # Projecting the deformation pressure
                def_n = -(p_d*normal_cp).sum(1)
                def_t = -(p_d*tangent_cp).sum(1)
                def_r = -(p_d*radial_cp).sum(1)
            else:
                self.perimiter_transpozed = perimiter_transpozed[0:3, :]
                #print(np.linalg.norm(temp))
                break
           
        if np.linalg.norm(rot_sum) >= 0.9:
            return False


        # In the end we add the moment around z axis to the pressure distribution
        m_z = ma_z-m_z
        # Distances of points to "z" axis that goes trough origin.
        z_ax_stacked = np.tile(z_ax, (np.shape(perimiter_cp)[0], 1))
        leavers_vec = np.transpose(perimiter_transpozed[0:3, :]) - (
            z_ax_stacked.T * (np.transpose(perimiter_transpozed[0:3, :])*z_ax).sum(1)).T
        leavers = np.linalg.norm(leavers_vec, axis=1)
        # We determine the moment each point has to provide
        m_z_spread = np.linalg.norm(m_z)  # *self.du
        # We calculate the force/pressure at each point
        m_z_p = -m_z_spread/leavers
        # We determine the directions in which the pressure acts
        m_z_p_dir = np.cross(z_ax, leavers_vec)
        m_z_p_dir = scf.unit_array_of_vectors(m_z_p_dir)
        # FInal moment around z axis as a vector distribution of pressure
        m_z_p_vec = (m_z_p_dir.T*m_z_p).T
        m_z_p_n = -(m_z_p_vec*normal_cp).sum(1)
        m_z_p_t = -(m_z_p_vec*tangent_cp).sum(1)
        m_z_p_r = -(m_z_p_vec*radial_cp).sum(1)

        # Vacuum force contribution #340
        vac_n = np.tile(model.lip_area*vacuum, np.shape(normal_cp)[0])

        # Next we analyze the plane forces ----------------------------------------
        p_nor = def_n + m_z_p_n + vac_n
        p_tan = def_t + m_z_p_t
        p_rad = def_r + m_z_p_r

        # Actual force in the direction of average_normal
        def_n, def_t, def_r = 0, 0, 0
        self.premik = 0

        scale_coef = 3.5
        for i in range(10):
            # Calculate force
            force = self.force_calc(
                p_nor, p_tan, p_rad, normal_cp, tangent_cp, radial_cp, z_ax)
            force_sum = area*vacuum-force
            # What kind of force is desired in the direction of the average_normal
            force_desired = np.dot(-f_dir, z_ax)

            # We have reached equilibrium force, break the loop
            if np.allclose([force_sum], [force_desired], atol=1) == True:
                break
            else:
                # Transform the points up or down to get cloaser to the desired force
                sign = np.sign(force_sum - force_desired)
                scale = np.abs(force_sum - force_desired)

                self.premik += sign*z_ax_2*scale/3
                scale_coef += 3.5-0.3

                if i == 9:
                    print("WARNING FORCE NOT BALANCING")

                T_mat = scf.translation(sign*z_ax_2*scale/3.5)
                perimiter_transpozed[0:4, :] = np.dot(
                    T_mat, perimiter_transpozed)

                dist = np.dot(
                    perimiter_transpozed[0:3, :].T, np.transpose(a_v))
                distance = dist-offset-model.max_deformation

                def_vectors = self._calculate_deformation_vectors(a_v)
                # Presure because of deformation
                p_d = (def_vectors.T * distance * model.k_def).T
                # Projecting the deformation pressure
                def_n = -(p_d*normal_cp).sum(1)
                def_t = -(p_d*tangent_cp).sum(1)
                def_r = -(p_d*radial_cp).sum(1)
                p_nor = def_n + m_z_p_n + vac_n
                p_tan = m_z_p_t + def_t
                p_rad = m_z_p_r + def_r

        self.perimiter_transpozed = perimiter_transpozed[0:3, :]

        # Lastly add the pressure form the x and y axis forces
        f_x_already = self.force_calc(
            p_nor, p_tan, p_rad, normal_cp, tangent_cp, radial_cp, x_ax)
        f_y_already = self.force_calc(
            p_nor, p_tan, p_rad, normal_cp, tangent_cp, radial_cp, y_ax)
        f_x = (np.dot(-f_dir, x_ax)+f_x_already)*x_ax
        f_y = (np.dot(-f_dir, y_ax)+f_y_already)*y_ax
        f_x_n = (normal_cp*f_x).sum(1)
        f_x_t = (tangent_cp*f_x).sum(1)
        f_x_r = (radial_cp*f_x).sum(1)
        f_y_n = (normal_cp*f_y).sum(1)
        f_y_t = (tangent_cp*f_y).sum(1)
        f_y_r = (radial_cp*f_y).sum(1)

        p_nor += f_x_n + f_y_n
        p_tan += f_x_t + f_y_t
        p_rad += f_x_r + f_y_r

        # We also add the curvature pressure
        premik = np.linalg.norm(self.premik)
        reduction_p_bm = np.abs(premik-model.max_deformation)
        p_nor += self.p_bm

        # We look at friction for as an integral over whole perimiter.
        t1 = trapz(p_nor*model.coef_friction, self.du_cumulative)
        t2 = trapz(np.sqrt(p_tan**2+p_rad**2), self.du_cumulative)

        if p_nor[np.argmin(p_nor)] < 0:
            #print("Failure because of normal force.")
            return False
        if t2 > t1:
            #print("Failure bacause of friction force.")
            return False
        else:
            return True

    def force_calc(self, p_nor, p_tan, p_rad, n, t, r, direction):
        p_nor_v = (n.T * p_nor).T
        p_nor_p = (direction*p_nor_v).sum(1)
        force_n = trapz(p_nor_p, self.du_cumulative)

        p_tan_v = (t.T * p_tan).T
        p_tan_p = (direction*p_tan_v).sum(1)
        force_t = trapz(p_tan_p, self.du_cumulative)

        p_rad_v = (r.T * p_rad).T
        p_rad_p = (direction*p_rad_v).sum(1)
        force_r = trapz(p_rad_p, self.du_cumulative)

        return force_n+force_t+force_r

    def moment_calc(self, p_nor, p_tan, p_rad, n, t, r, perimeter):
        """
        Based on the given perimiter distributed force, force direction and perimiter points centered around 1
         the function calculates the moments generated by the distributed force.
        --------------
        p_nor: (n, ) np.array
            Normal component of the distributed force
        p_tan: (n, ) np.array
            Tangent component of the distributed force
        p_rad: (n, ) np.array
            Radial component of the distributed force
        n: (n, 3) np.array
            Matrix containing the normals to the surface along the perimiter
        t: (n, 3) np.array
            Matrix containing the tangents to the surface along the perimiter
        r: (n, 3) np.array
            Matrix containing the radials to the surface along the perimiter
        perimiter: (n, 3) np.array
            Perimiter points centered around (0,0,0)
        --------------
        moment : (3, ) np.array
            Moment vector calculated given the inputs.
        """

        p_nor_v = (n.T * p_nor).T
        p_tan_v = (t.T * p_tan).T
        p_rad_v = (r.T * p_rad).T

        pressure_sum = p_nor_v+p_tan_v+p_rad_v
        # Calculating the applied moment. We must transform everything to mm
        inside_vector = perimeter  # - np.array(self.p_0)
        moment = np.cross(inside_vector, pressure_sum) * \
            self.du[:, np.newaxis] * 4
        # We get a 3D moment in Nmm, We transform it to N*m
        moment_sum = np.sum(moment, axis=0)

        moment_x = moment_sum[0]
        moment_y = moment_sum[1]
        moment_z = moment_sum[2]

        return np.array([moment_x, moment_y, moment_z])
