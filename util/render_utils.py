from cv2 import transform
import matplotlib.pyplot as plt
import trimesh
import numpy as np
import tensorflow as tf
import copy


def show_image(data, segmap = None):
    """
    Overlay rgb image with segmentation and imshow segment
    Arguments:
        data {np.ndarray} -- color or depth image
        segmap {np.ndarray} -- integer segmap of same size as rgb
    """
    plt.figure()

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.ion()
    plt.show()

    if data is not None:
        plt.imshow(data)
    if segmap is not None:
        print(segmap)
        plt.imshow(segmap)
    plt.draw()
    plt.pause(5)


def inverse_transform(trans):
    """
    Computes the inverse of 4x4 transform.
    
    Arguments:
    --------
    trans {np.ndarray} -- 4x4 transform.
        
       
    Returns:
    --------
    [np.ndarray] -- inverse 4x4 transform
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output


def network_out_tf(pc, app, tf_cam, inverse = False):
    """
    Transforms point cloud and approach vector between frames given a transform.
    ----------
    Arguments:
        pc {tf.tensor} -- BxNx3 point cloud
        app {tf.tensor} -- BxNx3 approach vector
        tf_cam {np.ndarray} -- Bx4x4 transform
    Keyword Arguments:
        inverse {bool} -- if True, inverse transform is applied (default: {False})
    ----------
    Returns:
        [tf.tensor] -- BxNx3 point cloud
        [tf.tensor] -- BxNx3 approach vector
    """
    pc_temp = pc.numpy()
    app_temp = app.numpy()
    transform = np.copy(tf_cam)
    pc_out = np.zeros_like(pc_temp, dtype = np.float32)
    app_out = np.zeros_like(app_temp, dtype=np.float32)
    for j in range(len(pc_temp)):
        if inverse:
            transform[j] = inverse_transform(transform[j])
        temp = np.ones((len(pc_temp[j]), 4))
        temp[:, :3] = pc_temp[j]
        pc_out[j] = np.dot(
            transform[j], temp.T).T[:, 0: 3]
        app_out[j] = np.dot(
            transform[j, 0:3, 0:3], app_temp[j].T).T[:, 0: 3]

    pc_out = tf.convert_to_tensor(pc_out, np.float32)
    app_out = tf.convert_to_tensor(app_out, np.float32)
    return pc_out, app_out

class Object(object):
    """Represents a graspable object."""

    def __init__(self, filename):
        """Constructor.
        :param filename: Mesh to load
        :param scale: Scaling factor
        """
        self.mesh = trimesh.load(filename)
        self.scale = 1.0

        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('object', self.mesh)

    def rescale(self, scale=1.0):
        """Set scale of object mesh.
        :param scale
        """
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def resize(self, size=1.0):
        """Set longest of all three lengths in Cartesian space.
        :param size
        """
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)

    def in_collision_with(self, mesh, transform):
        """Check whether the object is in collision with the provided mesh.
        :param mesh:
        :param transform:
        :return: boolean value
        """
        return self.collision_manager.in_collision_single(mesh, transform=transform)


def transform_grasp(grasp_dict, tf, scale_to_millimeters = True, add_transform_score = True):
    """
    Given a new transformation matrix for the object, transform all the grasps for that object.

    Parameters
    ----------
    grasp_dict : (dict) : the dictionary containing grasp data for the given object
            - "tf"  : np.array() [num_grasps, 4, 4] 
                4x4 transformation matrices of the object grasps
            - "score" : np.array() [num_grasps]
                The score of each grasp
    tf : np.array() [4, 4]
        New transformation matrix for the object
    scale_to_millimeters : bool, optional
        Scales the grasps coordinates from meters to millimeters, by default True
    add_transform_score : bool, optional
        Scales the score of the grasps based on their new orientation.
        (grasp with z-axis pointing down get score of 0, grasp with z-axis pointing up get score of 1), by default True

    Returns
    -------
    grasp_dict : (dict)
        Updated grasp dictionary.
    """

    new_grasp_tf = []
    new_grasp_scores = []
    
    grasp_tf = grasp_dict["tf"] # [num_grasps, 4, 4]
    if np.shape(grasp_tf) == (0,):
        return grasp_dict
    grasp_tf = scale_grasp(grasp_tf)
    # Calculate new tf for the grasp np.dot(tf, grasp_tf) 
    new_grasp_tf = np.matmul(tf, grasp_tf)
    # Calculate new grasp score
    if add_transform_score:
        temp_points = np.zeros((new_grasp_tf.shape[0], 4))
        temp_points[:, -2] = 1
        temp_points = np.einsum("ijk,ik->ij", new_grasp_tf, temp_points)
        approach_vector = temp_points[:, -2]
        new_grasp_scores = grasp_dict["scores"] * (0.5 + 0.5 * approach_vector)

    grasp_dict["tf"] = new_grasp_tf
    if add_transform_score:
        grasp_dict["scores"] = new_grasp_scores

    return grasp_dict

def scale_grasp(grasp_tf, scale=0.001):
    """Scale the xyz position of the grasp.

    Parameters
    ----------
    grasp_tf : np.array() [(N), 4, 4]
    scale : float, optional
        New scale fro grasp, by default 0.001 (Meter to millimeter)

    Returns
    -------
    grasp_tf: np.array() [4, 4]
    """
    if grasp_tf.ndim == 2:
        grasp_tf[:3, 3] = grasp_tf[:3, 3] * scale
    elif grasp_tf.ndim == 3:
        grasp_tf[:, :3, 3] = grasp_tf[:, :3, 3] * scale
    return grasp_tf

def transform_point_array(point_arr, tf):
    points = np.copy(point_arr)
    temp = np.ones((points.shape[1]+1, points.shape[0]))
    temp[0:3,:] = points.T
    temp = np.dot(tf, temp)
    return temp[0:3, :].T
