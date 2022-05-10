from cv2 import transform
import matplotlib.pyplot as plt
import trimesh
import numpy as np
import tensorflow as tf
from copy import deepcopy


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
        trans {np.ndarray} -- 4x4 transform.
    Returns:
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
        pc {tf.tensor} -- BxNX3 point cloud
        app {tf.tensor} -- BxNX3 approach vector
        tf_cam {np.ndarray} -- Bx4x4 transform
    Keyword Arguments:
        inverse {bool} -- if True, inverse transform is applied (default: {False})
    ----------
    Returns:
        [tf.tensor] -- BxNX3 point cloud
        [tf.tensor] -- BxNX3 approach vector

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
