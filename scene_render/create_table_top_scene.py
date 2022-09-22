
import os
import trimesh
import trimesh.path
import trimesh.transformations as tra
import numpy as np
import random
import time
import signal
import argparse

from os import walk
# from trimesh.permutate import transform
from multiprocessing import freeze_support

from util.render_utils import transform_grasp
from util.dataset_utils import get_eval_obj_names, get_data_paths, load_scene_data, load_mesh, load_grasp
import util.utils as utils

import trimeshVisualize


class Scene(object):
    """Represents a scene, which is a collection of objects and their poses."""
    def __init__(self) -> None:
        """Create a scene object."""
        self._objects = {}
        self._poses = {}
        self._support_objects = []

        self.collision_manager = trimesh.collision.CollisionManager()

    def add_object(self, obj_id, obj_mesh, pose, support=False):
        """
        Add a named object mesh to the scene.
        --------------
        Args:
            obj_id (str): Name of the object.
            obj_mesh (trimesh.Trimesh): Mesh of the object to be added.
            pose (np.ndarray): Homogenous 4x4 matrix describing the objects pose in scene coordinates.
            support (bool, optional): Indicates whether this object has support surfaces for other objects. Defaults to False.
        --------------
        """
        self._objects[obj_id] = obj_mesh
        self._poses[obj_id] = pose
        if support:
            self._support_objects.append(obj_mesh)

        self.collision_manager.add_object(
            name=obj_id, mesh=obj_mesh, transform=pose)

    def _get_random_stable_pose(self, stable_poses, stable_poses_probs):
        """Return a stable pose according to their likelihood.
        Args:
            stable_poses (list[np.ndarray]): List of stable poses as 4x4 matrices.
            stable_poses_probs (list[float]): List of probabilities.
        Returns:
            np.ndarray: homogeneous 4x4 matrix
        """
        index = np.random.choice(len(stable_poses), p=stable_poses_probs)
        inplane_rot = tra.rotation_matrix(
            angle=np.random.uniform(0, 2.0 * np.pi), direction=[0, 0, 1]
        )
        return inplane_rot.dot(stable_poses[index])

    def _get_support_polygons(self, min_area=0.01, gravity=np.array([0, 0, -1.0]), erosion_distance=0.02):
        """Extract support facets by comparing normals with gravity vector and checking area.
        Args:
            min_area (float, optional): Minimum area of support facets [m^2]. Defaults to 0.01.
            gravity ([np.ndarray], optional): Gravity vector in scene coordinates. Defaults to np.array([0, 0, -1.0]).
            erosion_distance (float, optional): Clearance from support surface edges. Defaults to 0.02.
        Returns:
            list[trimesh.path.polygons.Polygon]: list of support polygons.
            list[np.ndarray]: list of homogenous 4x4 matrices describing the polygon poses in scene coordinates.
        """
        # Check if gravity is a unit vector
        assert np.isclose(np.linalg.norm(gravity), 1.0)

        support_polygons = []
        support_polygons_T = []

        # Add support plane if it is set (although not infinite)
        support_meshes = self._support_objects

        for obj_mesh in support_meshes:
            # get all facets that are aligned with -gravity and bigger than min_area
            support_facet_indices = np.argsort(obj_mesh.facets_area)
            support_facet_indices = [
                idx
                for idx in support_facet_indices
                if np.isclose(obj_mesh.facets_normal[idx].dot(-gravity), 1.0, atol=0.5)
                and obj_mesh.facets_area[idx] > min_area
            ]

            for inds in support_facet_indices:
                index = inds
                normal = obj_mesh.facets_normal[index]
                origin = obj_mesh.facets_origin[index]

                T = trimesh.geometry.plane_transform(origin, normal)
                vertices = trimesh.transform_points(
                    obj_mesh.vertices, T)[:, :2]

                # find boundary edges for the facet
                edges = obj_mesh.edges_sorted.reshape((-1, 6))[
                    obj_mesh.facets[index]
                ].reshape((-1, 2))
                group = trimesh.grouping.group_rows(edges, require_count=1)

                # run the polygon conversion
                polygon = trimesh.path.polygons.edges_to_polygons(
                    edges=edges[group], vertices=vertices
                )

                assert len(polygon) == 1

                # erode to avoid object on edges
                polygon[0] = polygon[0].buffer(-erosion_distance)

                if not polygon[0].is_empty and polygon[0].area > min_area:
                    support_polygons.append(polygon[0])
                    support_polygons_T.append(
                        trimesh.transformations.inverse_matrix(T))

        return support_polygons, support_polygons_T

    def as_trimesh_scene(self, display = False):
        """
        Return trimesh scene representation.
        --------------
        Keyword args:
            display (bool->default is False) : Wheather to display the trimesh scene
        --------------
        Returns:
            trimesh.Scene: Scene representation.
        """
        trimesh_scene = trimesh.scene.Scene()
        for obj_id, obj_mesh in self._objects.items():
            trimesh_scene.add_geometry(
                obj_mesh,
                node_name=obj_id,
                geom_name=obj_id,
                transform=self._poses[obj_id],
            )

        if display:
            trimesh_scene.show()
            
        return trimesh_scene


class TableScene(Scene):
    """
    Holds current table-top scene, samples object poses and checks grasp collisions.
    --------------
    Arguments:
        root_folder {str} -- path to acronym data
        gripper_path {str} -- relative path to gripper collision mesh
    Keyword Arguments:
        lower_table {float} -- lower table to permit slight grasp collisions between table and object/gripper (default: {0.02})
    --------------
    """

    def __init__(self, splits, gripper_path=None, data_dir=None, lower_table=0.02):

        super().__init__()

        self.splits = splits # Train or test data
        print(gripper_path)
        if gripper_path != None:
            tf = trimesh.geometry.align_vectors(np.array([0, 0, 1]), np.array([0, 0, -1]))
            #self.gripper_mesh = trimesh.load(os.path.join(BASE_DIR, gripper_path), force="mesh")
            self.gripper_mesh = trimesh.load(gripper_path, force="mesh")
            self.gripper_mesh.apply_transform(tf)
            self.gripper_mesh = self.gripper_mesh.apply_scale(0.001)
            print("Gripper mesh loaded")
        else:
            tf = trimesh.transformations.translation_matrix(np.array([0,0, 0.6]))
            self.gripper_mesh = trimesh.primitives.Cylinder(radius=0.1, height=1, transform = tf)
            self.gripper_mesh = self.gripper_mesh.apply_scale(0.001)
            print("Gripper mesh loaded 2")

        # Table
        self._table_dims = [1.0, 1.2, 0.6]
        self._table_support = [0.6, 0.6, 0.6]
        self._table_pose = np.eye(4)
        self.table_mesh = trimesh.creation.box(self._table_dims, units = "meters")
        self.table_support = trimesh.creation.box(self._table_support, units = "meters")

        # Obj meshes
        self.data_dir = data_dir
        
        self.obj_dict = get_eval_obj_names(self.data_dir)
        self.meshes_root, self.grasps_root = get_data_paths(self.data_dir)
        self.splits = splits
        #self.data_splits = load_splits(root_folder)
        #self.category_list = list(self.data_splits.keys())
        # self.contact_infos = load_contacts(
        #     root_folder, self.data_splits, splits=self.splits)

        self._lower_table = lower_table

        self._scene_count = 0

    def _filter_colliding_grasps(self, transformed_grasps):
        """
        Filter out colliding grasps
        Arguments:
            transformed_grasps {np.ndarray} -- Nx4x4 grasps
            transformed_contacts {np.ndarray} -- 2Nx3 contact points
        Returns:
            [np.ndarray, np.ndarray] -- Mx4x4 filtered grasps, Mx2x3 filtered contact points
        """
        filtered_grasps = []
        filtered_scores = []
        for grasp_tf, grasp_score in zip(transformed_grasps["tf"], transformed_grasps["scores"]):
            if not self.is_colliding(self.gripper_mesh, grasp_tf):
                filtered_grasps.append(grasp_tf)
                filtered_scores.append(grasp_score)
        return {"tf": np.array(filtered_grasps).reshape(-1, 4, 4), "scores": np.array(filtered_scores)}

    def get_random_object(self):
        """
        Return random scaled but not yet centered object mesh
        --------------
        --------------
        Returns:
            [trimesh.Trimesh, str] -- ShapeNet mesh from a random category, h5 file path
        """
        # Get an object
        
        for i in range(20):
            random_obj_name = random.choice(self.obj_dict[self.splits])
            if random_obj_name in self._objects.keys():
                continue
            else:
                break
        else:
            raise Exception("Not enough unique objects found in dataset!!!")

        # load mesh
        obj_mesh = load_mesh(random_obj_name, self.meshes_root[self.splits])
        # load coresponding grasp
        obj_grasp = load_grasp(random_obj_name, self.grasps_root[self.splits])

        # mesh_mean = np.mean(obj_mesh.vertices, 0, keepdims=True)
        # obj_mesh.vertices -= mesh_mean

        return random_obj_name, obj_mesh, obj_grasp

    def find_object_placement(self, obj_mesh, max_iter):
        """Try to find a non-colliding stable pose on top of any support surface.
        Args:
            obj_mesh (trimesh.Trimesh): Object mesh to be placed.
            max_iter (int): Maximum number of attempts to place to object randomly.
        Raises:
            RuntimeError: In case the support object(s) do not provide any support surfaces.
        Returns:
            bool: Whether a placement pose was found.
            np.ndarray: Homogenous 4x4 matrix describing the object placement pose. Or None if none was found.
        """
        support_polys, support_T = self._get_support_polygons()
        if len(support_polys) == 0:
            raise RuntimeError("No support polygons found!")

        # get stable poses for object
        stable_obj = obj_mesh.copy()
        stable_obj.vertices -= stable_obj.center_mass
        stable_poses, stable_poses_probs = stable_obj.compute_stable_poses(
            threshold=0, sigma=0, n_samples=3
        )
        #stable_poses, stable_poses_probs = obj_mesh.compute_stable_poses(threshold=0, sigma=0, n_samples=5)
        # Sample support index
        support_index = max(enumerate(support_polys),
                            key=lambda x: x[1].area)[0]

        iter = 0
        colliding = True
        while iter < max_iter and colliding:

            # Sample position in plane
            pts = trimesh.path.polygons.sample(
                support_polys[support_index], count=1
            )

            # To avoid collisions with the support surface
            pts3d = np.append(pts, 0)

            # Transform plane coordinates into scene coordinates
            placement_T = np.dot(
                support_T[support_index],
                trimesh.transformations.translation_matrix(pts3d),
            )

            pose = self._get_random_stable_pose(
                stable_poses, stable_poses_probs)

            placement_T = np.dot(
                np.dot(placement_T,
                       pose), tra.translation_matrix(-obj_mesh.center_mass)
            )

            # Check collisions
            colliding = self.is_colliding(obj_mesh, placement_T)

            iter += 1


        return not colliding, placement_T if not colliding else None

    def is_colliding(self, mesh, transform, eps=1e-6):
        """
        Whether given mesh collides with scene
        --------------
        Arguments:
            mesh {trimesh.Trimesh} -- mesh 
            transform {np.ndarray} -- mesh transform
        Keyword Arguments:
            eps {float} -- minimum distance detected as collision (default: {1e-6})
        --------------
        Returns:
            [bool] -- colliding or not
        """
        dist = self.collision_manager.min_distance_single(
            mesh, transform=transform)
        return dist < eps

    def arrange(self, num_obj, max_iter=100, time_out=8):
        """
        Arrange random table top scene with contact grasp annotations
        --------------
        Arguments:
            num_obj {int} -- number of objects
        Keyword Arguments:
            max_iter {int} -- maximum iterations to try placing an object (default: {100})
            time_out {int} -- maximum time to try placing an object (default: {8})
        --------------
        Returns:
            scene_filtered_grasps {list} -- list of valid grasps for the scene. That is grasps which the gripper can reach.
            scene_filtered_scores {list} -- A corresponding list of grasp scores
            object_names {list} -- names of all objects in the scene
            obj_transforms {list} --  transformation matrices for all the scene objects
            obj_grasp_idcs {list} : List of ints indicating to which object some grasps belong to.
        """

        # Add table
        self._table_pose[2, 3] -= self._lower_table
        self.add_object('table', self.table_mesh, self._table_pose)

        self._support_objects.append(self.table_support)

        object_names = []
        obj_transforms = []
        obj_scales = []
        object_grasps = []

        for i in range(num_obj):
            obj_name, obj_mesh, obj_grasp = self.get_random_object()
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(10)
            try:
                success, placement_T = self.find_object_placement(
                    obj_mesh, max_iter)
                
            except Exception as exc:
                print(exc, " after {} seconds!".format(time_out))
                continue
            signal.alarm(0)
            if success:
                self.add_object(obj_name, obj_mesh, placement_T)
                # obj_scales.append(
                #     float(random_grasp_path.split('_')[-1].split('.h5')[0]))
                object_names.append(obj_name)
                obj_transforms.append(placement_T)
                object_grasps.append(obj_grasp)
            else:
                print("Couldn't place object"," after {} iterations!".format(max_iter))
        print('Placed {} objects'.format(len(object_names)))

        scene_filtered_grasps = []
        scene_filtered_scores = []
        obj_grasp_idcs = []
        grasp_count = 0

        for obj_transform, object_grasp in zip(obj_transforms, object_grasps):

            transformed_obj_grasp = transform_grasp(object_grasp, obj_transform)
            # Expand first dimension to list of np arrays
            transformed_obj_grasp["tf"] = list(transformed_obj_grasp["tf"])
            transformed_obj_grasp["scores"] = list(transformed_obj_grasp["scores"])

            filtered_grasps = self._filter_colliding_grasps(transformed_obj_grasp)
            scene_filtered_grasps.append(filtered_grasps["tf"])
            scene_filtered_scores.append(filtered_grasps["scores"])
            grasp_count += len(filtered_grasps["tf"])
            obj_grasp_idcs.append(grasp_count)

        

        scene_filtered_grasps = np.concatenate(scene_filtered_grasps, 0)
        scene_filtered_scores = np.concatenate(scene_filtered_scores, 0)


        self._table_pose[2, 3] += self._lower_table
        self.set_mesh_transform('table', self._table_pose)

        return scene_filtered_grasps, scene_filtered_scores, object_names, obj_transforms, obj_grasp_idcs

    def set_mesh_transform(self, name, transform):
        """
        Set mesh transform for collision manager
        --------------
        Arguments:
            name {str} -- mesh name
            transform {np.ndarray} -- 4x4 homog mesh pose
        --------------
        """
        self.collision_manager.set_transform(name, transform)
        self._poses[name] = transform

    def handler(self, signum, frame):
        raise Exception("Could not place object ")

    def visualize(self, tf, scores):
        """
        It visualises the given scene. If grasps are provided they are also displayed. 
        Be careful may take a long time.
        --------------
        Arguments:
            tf {list} -- a list of grasp transforms
            scores {list} -- a list of grasp scores
        --------------
        """
        scene = self.as_trimesh_scene(display=False)
        my_scene = trimeshVisualize.Scene()

        for i in range(tf.shape[0]):
            my_scene.plot_grasp(tf[i], scores[i], units="meters")
        my_scene.plot_coordinate_system(scale = 0.01)
        my_scene.display(my_scene=scene)
        
    def load_scene(self, scene_name, scenes_root_dir, visualize=False):
        """
        Loads the scene to the object
        --------------
        Arguments:
            scene_name {str} : Name of the scene. (without extension)
            scenes_root_dir {str} : A path to the directory where the scene is located
        --------------
        """
        self.reset()

        self.add_object('table', self.table_mesh, self._table_pose)
        self._support_objects.append(self.table_support)

        scene_grasps_tf, scene_grasps_scores, object_names, obj_transforms, obj_grasp_idcs = load_scene_data(
            scene_name, scenes_root_dir)
        

        for obj, tf in zip(object_names, obj_transforms):
            self.add_object(obj, load_mesh(obj,self.meshes_root[self.splits]), tf)
            
        if visualize:
            self.visualize(scene_grasps_tf, scene_grasps_scores)

        return scene_grasps_tf, scene_grasps_scores, object_names, obj_transforms, obj_grasp_idcs

    def save_scene(self, output_dir, scene_grasps_tf, scene_grasps_scores, object_names, obj_transforms, obj_grasp_idcs):
        """
        Save the scene with its objects and grasps to a directory.
        --------------
        Arguments:
            output_dir {str} : Absolute path to where to save the scene
            scene_grasps_tf {list} : A list of grasps "tf".
            scene_grasps_scores {list} : A corespoding list of scores for individual grasps.
            object_names {list} : List of object names in the scene
            obj_transforms {list} : Transformation matrices for placing of individual objects
            obj_grasp_idcs {list} : List of ints indicating to which object some grasps belong to.
        --------------
        """
        # Make dir if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        scene_info = {}
        scene_info["scene_grasps_tf"] = scene_grasps_tf
        scene_info["scene_grasps_scores"] = scene_grasps_scores
        scene_info["object_names"] = object_names
        scene_info["obj_transforms"] = obj_transforms
        scene_info["obj_grasp_idcs"] = np.array(obj_grasp_idcs)
        output_path = os.path.join(output_dir, f"{self._scene_count:06d}.npz")
        while os.path.exists(output_path):
            self._scene_count += 1
            output_path = os.path.join(output_dir, f"{self._scene_count:06d}.npz")
        np.savez(output_path, **scene_info)

    def reset(self):
        """
        --------------
        Reset, i.e. remove scene objects
        --------------
        """
        for name in self._objects:
            self.collision_manager.remove_object(name)
        self._objects = {}
        self._poses = {}
        self._support_objects = []

class SceneDatasetGenerator():
    def __init__(self,
                 data_dir,
                 gripper_path,
                 splits,
                 min_num_objects = 8,
                 max_num_objects = 13,
                 max_iterations = 100,
                 ) -> None:
        self._data_dir = data_dir
        self._gripper_path = gripper_path
        self._splits = splits
        self._min_num_objects = min_num_objects
        self._max_num_objects = max_num_objects
        self._max_iterations = max_iterations
        self.fails = []
        
        self.save_dir =  os.path.join(os.path.join(self._data_dir, "scenes_3d"), self._splits)
        
    
    def generate_scene(self, scene_idx):
        """Generate a scene with name = "scene_idx"

        Parameters
        ----------
        scene_idx : int
            The index used to name the scene. Object get selected randomly.

        Returns
        -------
        Success : bool
            Whether the scene was successfully generated.
        """
        start_time = time.time()
        print(f"Evaluating scene {scene_idx}")
        
        # Create scene
        table_scene = TableScene(self._splits, self._gripper_path, self._data_dir)
        table_scene.reset()
        num_objects = np.random.randint(self._min_num_objects, self._max_num_objects+1)

        
        try:
            scene_grasps_tf, scene_grasps_scores, object_names, obj_transforms, obj_grasp_idcs = table_scene.arrange(num_objects, self._max_iterations)
            self.save_scene(scene_idx, self.save_dir, scene_grasps_tf, scene_grasps_scores, object_names, obj_transforms, obj_grasp_idcs)
            print(f"Created scene {scene_idx}, time taken {time.time()-start_time}")
            return True
        except:
        
            self.fails.append(scene_idx)
            return False
        
    def generate_N_scenes(self, start_idx, end_idx, n_processors = 1, overwrite=False):
        """Generate N scenes with names from start_idx to end_idx. Can be done in parallel.
        
        Args:
        ----------
        start_idx : int
        end_idx : int
        n_processors : int, optional
            Number of processors to use, by default 1
        overwrite : bool, optional
            Whether to overwrite existing scenes, by default False
        ----------
        """
        
        path_scenes = os.path.join(os.path.join(self._data_dir, "scenes_3d"), self._splits)
        
        existing_scenes = next(walk(path_scenes), (None, None, []))[2]
        existing_scenes = [int(os.path.splitext(os.path.basename(scene))[0]) for scene in existing_scenes]

        start_time = time.time()
        scene_list = list(range(start_idx, end_idx))
        
        if overwrite == False:
            scene_list = [i for i in scene_list if i not in existing_scenes]
        if n_processors == 1:
            for scene in scene_list:
                self.generate_scene(scene)
        else:
            out = utils.run_multiprocessing(self.generate_scene,
                                            scene_list, n_processors)
        print(f"Evaluated {len(scene_list)} scenes, time taken {time.time()-start_time}")
        print(f"Failed to evaluate {self.fails} scenes")
        return True
    
    def save_scene(self, 
                scene_idx,
                save_dir,
                scene_grasps_tf,
                scene_grasps_scores,
                object_names, obj_transforms,
                obj_grasp_idcs):
        """
        Save the scene with its objects and grasps to a directory.
        
        Arguments:
        --------------
        scene_idx {int} : Scene index
        output_dir {str} : Absolute path to where to save the scene
        scene_grasps_tf {list} : A list of grasps "tf".
        scene_grasps_scores {list} : A corresponding list of scores for individual grasps.
        object_names {list} : List of object names in the scene
        obj_transforms {list} : Transformation matrices for placing of individual objects
        obj_grasp_idcs {list} : List of ints indicating to which object some grasps belong to.
            
        Output:
        --------------
        scene_info {dict} : A dictionary containing the scene information
        """
        # Make dir if it does not exist
        if not os.path.exists(save_dir):
            print(f"Output directory does not exist. Creating directory {save_dir}")
            os.makedirs(save_dir)

        scene_info = {}
        scene_info["scene_grasps_tf"] = scene_grasps_tf
        scene_info["scene_grasps_scores"] = scene_grasps_scores
        scene_info["object_names"] = object_names
        scene_info["obj_transforms"] = obj_transforms
        scene_info["obj_grasp_idcs"] = np.array(obj_grasp_idcs)
        output_path = os.path.join(save_dir, f"{scene_idx:06d}.npz")
        while os.path.exists(output_path):
            self._scene_count += 1
            output_path = os.path.join(save_dir, f"{scene_idx:06d}.npz")
        np.savez(output_path, **scene_info)

