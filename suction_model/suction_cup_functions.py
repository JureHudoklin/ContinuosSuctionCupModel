#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import trimesh
import random

from numpy.random import random_integers
from scipy import fft
from scipy.spatial.distance import cdist


def fourier_fit_3d(x, y, frequency_limit):
    """
    Filters a 3D input function using Fast Fourier Transform. The input x axis is expected to be range of 1.
    --------------
    x: (n, ) np.array
      Points along x_axis
    y : (n, 3)  np.array
      Coresponding points in space to "time stamps" x
    frequency_limit : int
      Cutoff frequency
    --------------
    y_filtered : (n, 3) np.array
      Filtered input y points
    """
    # Extract individual arrays
    y_x = y[:, 0]
    y_y = y[:, 1]
    y_z = y[:, 2]
    # Perform Fourier transform
    fft_o_x = fft.fft(y_x)
    fft_o_y = fft.fft(y_y)
    fft_o_z = fft.fft(y_z)
    # Calculate the frequencies
    freq = fft.fftfreq(len(x), 1/np.shape(x)[0])
    # Create the frequency mask
    freq_mask = np.where(freq > frequency_limit, 0, 1)
    freq_mask = np.where(freq < -frequency_limit, 0, 1)*freq_mask
    # Remove the undesired frequencies
    fft_o_x_c = fft_o_x*freq_mask
    fft_o_y_c = fft_o_y*freq_mask
    fft_o_z_c = fft_o_z*freq_mask
    # Perform the inverse Fourier transform

    ifft_o_x = fft.ifft(fft_o_x_c).real
    ifft_o_y = fft.ifft(fft_o_y_c).real
    ifft_o_z = fft.ifft(fft_o_z_c).real
    # Package everything together into one array
    y_filtered = np.array([ifft_o_x, ifft_o_y, ifft_o_z]).T
    return y_filtered


def fourier_fit_2d(x, y, frequency_limit):
    """
    Filters a 2D input function using Fast Fourier Transform. The input x axis is expected to be range of 1.
    --------------
    x: (n, ) np.array
      Points along x_axis
    y : (n, )  np.array
      Coresponding points in space to "time stamps" x
    frequency_limit : int
      Cutoff frequency
    Returns
    --------------
    y_filtered : (n, 3) np.array
      Filtered input y points
    """
    # Perform Fourier transform
    fft_o = fft.fft(y)
    # Calculate the frequencies
    freq = fft.fftfreq(len(x), 1/np.shape(x)[0])
    # Create the frequency mask
    freq_mask = np.where(freq > frequency_limit, 0, 1)
    freq_mask = np.where(freq < -frequency_limit, 0, 1)*freq_mask
    # Remove the undesired frequencies
    fft_o_c = fft_o*freq_mask
    # Perform the inverse Fourier transform
    ifft_o = fft.ifft(fft_o_c)
    y_filtered = ifft_o
    return y_filtered


def poly_area(x, y):
    """
    Calculates the area from a set of 2D points
    --------------
    x: (n, ) np.array
      Points x coordinates
    y : (n, )  np.array
      Points y coordinates

    Returns
    --------------
    area : float
        The area between the set of points
    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

def unit_vector(vec):
    """
    Return the normalized vector
    --------------
    vec: (3,) np.array
        Vector we want to normalize

    Returns
    --------------
    vec_normalized : (3,) np.array
        Samve vector but length 1.
    """
    return vec/np.linalg.norm(vec)

def unit_array_of_vectors(A_mat):
    """
    Formats an n x 3 matrix so each row is of length 1.
    --------------
    A_mat: (n, 3) np.array
      An n x 3 matrix representing 3D vectors

    Returns
    --------------
    A_mat : (n, 3) np.array
        Same matrix but with rows of normal size (length 1)
    """
    norm = np.linalg.norm(A_mat, axis=1)

    A_mat[:, 0] = np.divide(
        A_mat[:, 0], norm)
    A_mat[:, 1] = np.divide(
        A_mat[:, 1], norm)
    A_mat[:, 2] = np.divide(
        A_mat[:, 2], norm)

    return A_mat


def radial_sort(points,
                origin,
                normal):
    """
    Sorts a set of points radially (by angle) around an
    an axis specified by origin and normal vector.
    Parameters
    --------------
    points : (n, 3) float
      Points in space
    origin : (3,)  float
      Origin to sort around
    normal : (3,)  float
      Vector to sort around
    Returns
    --------------
    ordered : (n) int
      Array of new order
    """

    # create two axis perpendicular to each other and the normal,
    # and project the points onto them
    axis0 = [normal[0], normal[2], -normal[1]]
    axis1 = np.cross(normal, axis0)
    ptVec = points - origin
    pr0 = np.dot(ptVec, axis0)
    pr1 = np.dot(ptVec, axis1)

    # calculate the angles of the points on the axis
    angles = np.arctan2(pr0, pr1)

    # return the points sorted by angle
    return np.argsort(angles)


def closest_node(node, nodes, n):
    #a = cdist([node], nodes).argmin()
    #return np.argpartition(a, n)[n:]
    return cdist([node], nodes).argmin()


def skew(x):
    """
    Creates a skew symmetric matrix form the input vector
    --------------
    x : (3, ) np.array
      vector
    --------------
    Skew Symmetric Matrix : (3, 3) np.array
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def rot_axis_angle(u, phi):
    """
    Calculates a rotation matrix for rotation about axis u and angle phi
    --------------
    u : (3, ) np.array
      Unit rotation axis
    phi : float
      Angle in radians
    --------------
    R_mat : (3, 3) np.array
      Rotation matrix
    """
    R_mat = np.cos(phi)*np.eye(3)+np.sin(phi)*skew(u) + \
        (1-np.cos(phi))*np.outer(u, u)
    return R_mat


def translation(u):
    """
    Calculates a translation matrix for vector u
    --------------
    u : (3, ) np.array
      Translation vector
    --------------
    T : (3, 3) np.array
      Translation matrix
    """
    return np.array([[1, 0, 0, u[0]],
                     [0, 1, 0, u[1]],
                     [0, 0, 1, u[2]],
                     [0, 0, 0, 1]])


def plot_3D(A_mat):
    """
    Takes in a nx3 matrix od 3D points and plots them in a 3D figure
    --------------
    A_mat : (n, 3) np.array
      An n x 3 matrix representing 3D vectors
    --------------
    None : 
        Points get plotted and shown. No return value
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(A_mat[:, 0], A_mat[:, 1], A_mat[:, 2], 'r-')
    plt.show()


class FitClass():
    def __init__(self, y_0, y_0_dot) -> None:
        """
        A class that can be used to fit a 5th or 6th order periodic polynomial. 
        We must set the starting point and derivative and the size of x values should be from 0 to 1.
        --------------
        y_0 : float
            y value of first and last point (f(0) = f(1) = y_0)
        y_0_dot : float
            derivative at function start and end (f'(0) = f'(1) = y_0_dot)
        --------------
        """
        self.a_1 = y_0_dot
        self.a_0 = y_0

    def interpolate_5th_order(self, x, a_0, a_1):
        """
        Based on coefficients a_0, a_1, start point and gradient other coefficients are calculated and the function value at that x is returned
        --------------
        a_0 : float
            Coefficient multiplying the x**5 part
        a_1 : float
            Coefficient multiplying the x**4 part
        --------------
        function_value : float
            Function value f(x | a_0, a_1)
        """
        a_3 = 2*a_0 + a_1 - 3*self.a_1
        a_2 = 2*self.a_1-3*a_0-2*a_1
        return a_0*x**5 + a_1*x**4 + a_2*x**3 + a_3*x**2 + self.a_1*x**1 + self.a_0*x**0

    def interpolate_6th_order(self, x, a_0, a_1, a_2):
        """
        Based on coefficients a_0, a_1, start point and gradient other coefficients are calculated and the function value at that x is returned
        --------------
        a_0 : float
            Coefficient multiplying the x**6 part
        a_1 : float
            Coefficient multiplying the x**5 part
        --------------
        function_value : float
            Function value f(x | a_0, a_1)
        """
        a_3 = -4*a_0 - 3*a_1 - 2*a_2+2*self.a_1
        a_4 = 3*a_0+2*a_1+a_2-3*self.a_1
        return a_0*x**6 + a_1*x**5 + a_2*x**4 + a_3*x**3 + a_4*x**2 + self.a_1*x**1 + self.a_0*x**0


def create_half_sphere(subdivisions=1):
    """
    Creates an array of direction vectors that describe half a sphere.
    --------------
    subdivisions : int
        Defines how "accurately" the sphere is desired. Must be larger than 1 
    --------------
    direction_vectors : np.array(3,)
        Vectors of length 1 describing the half a sphere
    """

    force_sphere = trimesh.primitives.Sphere(
        radius=10, subdivisions=subdivisions)
    force_directions = force_sphere.vertices[force_sphere.vertices[:, 2] > 0]
    direction_vectors = force_directions/10
    return direction_vectors


def point_with_noise(point, noise_radius=20):
    """
    Adds some noise to the input position vector. The output vector will be inside the defined noise cube 2a*2a*2a
    --------------
    a_v : np.array(3,)
        Input unit vector
    noise_radius : float 
        Size of the cube  ("Maximum noise that can be added")
    --------------
    new_point : np.array(3,)
        Original point with noise applied.
    """
    noise = random_integers(-noise_radius, noise_radius, (3,))
    new_point = point+noise
    return new_point


def vector_with_noise(a_v, noise_cone_angle):
    """
    Adds some noise to the input unit vector. The output vector will be inside the defined noise cone from original vector.
    --------------
    a_v : np.array(3,)
        Input unit vector
    noise_cone_angle : float 
        Angle in radians
    --------------
    vector_with_noise : np.array(3,)
        Original vector with noise applied.
    """
    # Create a noise vector in zero Position of length 1
    noise_max = np.tan(noise_cone_angle)
    noise_x = random.uniform(-noise_max, noise_max)
    noise_y = random.uniform(-noise_max, noise_max)
    vector_with_noise = np.array([noise_x, noise_y, 1])
    vector_with_noise = vector_with_noise/np.linalg.norm(vector_with_noise)

    # Calculate then rotation matrix from (0,0,1) to approach vector
    rot_matrix = trimesh.geometry.align_vectors(
        np.array([0, 0, 1]), a_v, return_angle=False)

    # Rotate the vector
    vector_with_noise = np.dot(rot_matrix[0:3, 0:3], vector_with_noise)

    return vector_with_noise
