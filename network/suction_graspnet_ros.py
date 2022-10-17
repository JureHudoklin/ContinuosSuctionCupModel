#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import math
import os
import sys
import time
import numpy as np
import tensorflow
from scipy.spatial import cKDTree
from glob import glob

from suction_graspnet import SuctionGraspNet, build_suction_pointnet_graph
from config import Config
from point_cloud_reader import regularize_pc_point_count

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point, Pose, PoseArray
from cv_bridge import CvBridge

from suction_grasp_estimation.srv import SuctionGraspNetPlanner, SuctionGraspNetPlannerResponse
from suction_grasp_estimation.msg import SuctionGrasp

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.
    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

def to_pointcloud(depth, intrinsics):
    """
    Convert depth map to point cloud
    ----------
    Arguments:
        depth {np.ndarray} -- HxW depth map
        intrinsics {dict} -- camera intrinsics as dict
    ----------
    Returns:
        np.ndarray -- Nx4 homog. point cloud
    """

    fx= intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    mask = np.where(depth > intrinsics["znear"])

    x = mask[1]
    y = mask[0]

    normalized_x = (x.astype(np.float32) - cx)
    normalized_y = (y.astype(np.float32) - cy)

    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = depth[y, x]
    ones = np.ones(world_z.shape[0], dtype=np.float32)

    return np.vstack((world_x, world_y, world_z, ones)).T

def imgmsg_to_cv2(img_msg):
    """Convert ROS Image messages to OpenCV images.
    `cv_bridge.imgmsg_to_cv2` is broken on the Python3.
    `from cv_bridge.boost.cv_bridge_boost import getCvType` does not work.
    Args:
        img_msg (`sensor_msgs/Image`): ROS Image msg
    Raises:
        NotImplementedError: Supported encodings are "8UC3" and "32FC1"
    Returns:
        `numpy.ndarray`: OpenCV image
    """
    # check data type
    
    if img_msg.encoding == '':
        return None
    elif img_msg.encoding == '8UC3':
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == 'bgr8':
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == 'rgb8':
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == '32FC1':
        dtype = np.float32
        n_channels = 1
    elif img_msg.encoding == '64FC1':
        dtype = np.float64
        n_channels = 1
    else:
        raise NotImplementedError(
            'custom imgmsg_to_cv2 does not support {} encoding type'.format(img_msg.encoding))

    # bigendian
    dtype = np.dtype(dtype)
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    if n_channels == 1:
        img = np.ndarray(shape=(img_msg.height, img_msg.width),
                         dtype=dtype, buffer=img_msg.data)
    else:
        img = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                         dtype=dtype, buffer=img_msg.data)

    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        img = img.byteswap().newbyteorder()
    return img


class RvizVisualizer(object):
    def __init__(self):
        self.pc_pub = rospy.Publisher('suction_grasp_planner/pointcloud', PointCloud, queue_size=1)
        self.grasp_pub = rospy.Publisher('suction_grasp_planner/grasps', PoseArray)
                                   
    def __call__(self, suction_grasps, pc):
        frame_id = suction_grasps.header.frame_id
        
        # Visualize the PC
        std_pc = PointCloud()
        std_pc.header.frame_id = frame_id
        std_pc.header.stamp = rospy.Time.now()
        for point in pc:
            std_point = Point()
            std_point.x, std_point.y, std_point.z = point
            std_pc.points.append(std_point)
            
        self.pc_pub.publish(std_pc)
            
        # Visualize the grasps
        std_grasps = PoseArray()
        std_grasps.header.frame_id = frame_id
        std_grasps.header.stamp = rospy.Time.now()
        for grasp in suction_grasps.grasps:
            std_grasp = Pose()
            std_grasp.position = grasp.position
            
            # Get the orientation from the approach vector
            x_axis = np.array([grasp.approach.x, grasp.approach.y, grasp.approach.z]) # Approach transform z axis
            y_axis = np.array([(-x_axis[1]-x_axis[2])/x_axis[0], 1, 1]) 
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            
            tf_matrix_ = np.array([x_axis, y_axis, z_axis])
            tf_matrix = np.zeros((4, 4))
            tf_matrix[:3, :3] = tf_matrix_.T
            tf_matrix[3, 3] = 1
        
            quat = quaternion_from_matrix(tf_matrix)
            std_grasp.orientation.x, std_grasp.orientation.y, std_grasp.orientation.z, std_grasp.orientation.w = quat
            
            std_grasps.poses.append(std_grasp)
            
        self.grasp_pub.publish(std_grasps)
            
class SuctionGraspEstimator():
    def __init__(self, checkpoint_dir):
        """
        Parameters
        ----------
        checkpoint_dir : "str"
            Path to the directory containing the trained model weights and config file.
        """
        self.build_model(checkpoint_dir)

    def build_model(self, checkpoint_dir):
        """ Build the suction grasp net model

        Parameters
        ----------
        checkpoint_dir : "str"
            The path to the file containing the trained model weights and config file.
            Weights of the last checkpoint will be loaded.
        """
        # solve tensorflow memory issue
        physical_devices = tensorflow.config.list_physical_devices('GPU')
        tensorflow.config.experimental.set_memory_growth(
            physical_devices[0], True)  # allow memory growth

        # load train config from the saved model directory
        config = Config(os.path.join(checkpoint_dir, 'config.yml'))
        self.network_config = config.load()

        inputs, outputs = build_suction_pointnet_graph(self.network_config)
        self.model = SuctionGraspNet(inputs, outputs)

        # compile model
        self.model.compile(optimizer=tensorflow.keras.optimizers.Adam(
            learning_rate=0.1))

        # load weights
        weight_filelist = glob(
            os.path.join(checkpoint_dir, "weights/*.h5"))
        weight_filelist.sort()
        weight_file = weight_filelist[-1]
        print("Loaded weights from {}".format(weight_file))
        self.model.load_weights(weight_file)
        
        # set model to inference mode
        self.model.trainable = False

    def extract_point_clouds(self, depth_im,
                             camera_intr,
                             segmap = None,
                             zfar = 2,
                             znear = 0.1):
        """ Extract point clouds from a depth image.
        
        Parameters
        ----------
        depth_im :  np.ndarray (H, W)
        camera_intr : np.ndarray (3, 3)
        segmap : np.ndarray (H, W, dtype = int)  [optional, default: None]
            The segmentation map of the depth image. Pixels marked with 0 will be ignored. 
        zfar : float [optional, default: 2]
            The far clipping plane of the depth image.
        znear : float [optional, default: 0.1]
            The near clipping plane of the depth image.
            
        Returns:  
        ----------
        pc: numpy array of shape (N, 3)
        object_pcs: dictionary of numpy arrays of shape (N, 3)
        """
        intrinsics = {"fy" : camera_intr[0, 0],
                      "fx" : camera_intr[1, 1],
                      "cx" : camera_intr[0, 2],
                      "cy" : camera_intr[1, 2],
                      "znear" : znear,
                      "zfar" : zfar}

        if segmap is None:
            pc = to_pointcloud(depth_im, intrinsics)
            object_pcs = None
        else:
            segmap = segmap.astype(np.uint8)
            object_idx = np.unique(segmap)

            object_pcs = {}
            for idx in object_idx:
                obj_segmap = np.zeros(segmap.shape)
                if idx == 0:
                    continue
                obj_depth_im = np.where(segmap == idx, depth_im, 0)
                obj_pc = to_pointcloud(obj_depth_im, intrinsics)[:, 0:3]
                object_pcs[idx] = obj_pc

            # Convert depth image to point cloud
            pc = to_pointcloud(depth_im, intrinsics)

        return pc[:, 0:3], object_pcs

    def predict_scene_grasps(self, pc):
        # Regularize point cloud
        pc = regularize_pc_point_count(
            pc, self.network_config["RAW_NUM_POINTS"])

        pc_tensor = tensorflow.convert_to_tensor(pc, dtype=tensorflow.float32)
        pc_tensor = tensorflow.expand_dims(pc_tensor, 0)

        pc_mean = tensorflow.reduce_mean(pc_tensor, axis=1, keepdims=True)
        pc_tensor = pc_tensor - pc_mean

        pred_pc_tensor, pred_scores_tensor, pred_approach_tensor = self.model(
            pc_tensor)

        pred_pc_tensor = pred_pc_tensor + pc_mean
        pred_pc = tensorflow.squeeze(pred_pc_tensor).numpy()
        pred_scores = tensorflow.squeeze(pred_scores_tensor).numpy()
        pred_approach = tensorflow.squeeze(pred_approach_tensor).numpy()

        return pred_pc, pred_scores, pred_approach

    def filter_grasps(self, pred_pc, pred_score, pred_approach, object_pcs):
        """
        Filter grasps based on the predicted point cloud.
        ----------
        Arguments:
            pred_pc: numpy array of shape (N, 3)
            object_pcs: dictionary of numpy arrays of shape (N, 3)
        ----------
        Returns:
            filtered_grasps: dictionary of numpy arrays of shape (N, 3)
        """
        # Filter grasps based on the predicted point cloud
        filtered_grasps = {}
        if object_pcs is None:
            rospy.loginfo("Segmentation map is not provided. Grasp filtering is skipped.")
            sort_idx = np.argsort(pred_score)[::-1]
            filtered_grasps[1] = (pred_pc[sort_idx], pred_score[sort_idx], pred_approach[sort_idx])
        else:
            for obj_idx, obj_pc in object_pcs.items():
                my_tree = cKDTree(obj_pc)
                # _, indices = my_tree.query_ball_point(obj_pc, k=1)
                some_list = my_tree.query_ball_point(pred_pc, r=0.003)
                indices = []
                for i, _list in enumerate(some_list):
                    if len(_list) == 0:
                        continue
                    else:
                        indices.append(i)
                indices = np.array(indices)
                filtered_pc, filtered_score, filtered_approach = np.copy(pred_pc[indices]), np.copy(
                    pred_score[indices]), np.copy(pred_approach[indices])
                # Sort by score in descending order
                sort_idx = np.argsort(filtered_score)[::-1]
                filtered_pc, filtered_score, filtered_approach = filtered_pc[sort_idx], filtered_score[sort_idx], filtered_approach[sort_idx] 
                    
                filtered_grasps[obj_idx] = (filtered_pc, filtered_score, filtered_approach)
                
        return filtered_grasps

class GraspPlannerServer(object):
    def __init__(self,
                 checkpoint_dir = None,
                 z_range=[0.2, 1.8],
                 top_k=-1,
                 visualize=False):
        # get parameters
        self.z_range = z_range
        self.top_k = top_k
        
        self.visualize = visualize
        if self.visualize:
            self.grasp_visualizer = RvizVisualizer()

        # Get the model
        self.grasp_estimator = SuctionGraspEstimator(checkpoint_dir)

        # ros cv bridge
        self.cv_bridge = CvBridge()

        # ros service
        rospy.Service("suction_grasp_planner", SuctionGraspNetPlanner,
                      self.plan_grasp_handler)
        rospy.sleep(0.1)
        rospy.loginfo("Started Suction-GraspNet grasp planner with args: \
                      {}".format({"z_range": self.z_range, "top_k": self.top_k, "visualize": self.visualize}))

    def plan_grasp_handler(self, req):
        """
        Given the request plan the grasp
        
        Args:
        ----------
        req : 
            color_im : sensor_msgs/Image [Optional]
            depth_im : sensor_msgs/Image
            segmask : sensor_msgs/Image
            camera_intr : sensor_msgs/CameraInfo
        ----------
        res :
            SuctionGrasp[] grasps
        """
        start_time = time.time()
        req_frame_id = req.header.frame_id
        req_time_stamp = req.header.stamp
        
        # unpack request massage
        color_im, depth_im, segmask, camera_intr = self.read_images(req)
        rospy.loginfo("Read images in {:.3f} seconds.".format(time.time() - start_time))

        # Convert depth image to point clouds
        pc_full, object_pcs = self.grasp_estimator.extract_point_clouds(
            depth_im, camera_intr,
            segmap=segmask,
            zfar=self.z_range[1],
            znear=self.z_range[0])
        rospy.loginfo("Extract point clouds in {:.3f} seconds.".format(time.time() - start_time))

        # Generate grasp
        pred_pc, pred_scores, pred_approach = self.grasp_estimator.predict_scene_grasps(
            pc_full)
        rospy.loginfo("Predict grasps in {:.3f} seconds.".format(time.time() - start_time))

        # Filter per object grasps
        filtered_grasps = self.grasp_estimator.filter_grasps(
            pred_pc, pred_scores, pred_approach, object_pcs)

        # Generate grasp responce msg
        grasp_resp = SuctionGraspNetPlannerResponse()
        grasp_resp.header.frame_id = req_frame_id
        grasp_resp.header.stamp = req_time_stamp
        for obj_idx, filtered_grasps_i in filtered_grasps.items():
            if obj_idx == 0:
                continue
            
            if self.top_k == -1:
                top_k = len(filtered_grasps_i[0])
            else:
                top_k = self.top_k
                
            for i in range(0, top_k):
                if i >= len(filtered_grasps_i[0]):
                    break
                approach = filtered_grasps_i[2]
                contact_point = filtered_grasps_i[0]
                score = filtered_grasps_i[1]

                grasp = SuctionGrasp()
                
                grasp.approach.x, grasp.approach.y, grasp.approach.z = approach[i]
                grasp.position.x, grasp.position.y, grasp.position.z = contact_point[i]
                grasp.score = score[i]
                grasp.obj_id = obj_idx
                
                grasp_resp.grasps.append(grasp)
                
        if self.visualize:
            self.grasp_visualizer(grasp_resp, pc_full)
        rospy.loginfo("Generate grasp response in {:.3f} seconds.".format(time.time() - start_time))
        return grasp_resp

    def read_images(self, req):
        """Reads images from a ROS service request.
        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS ServiceRequest for grasp planner service.
        """
        # Get the raw depth and color images as ROS `Image` objects.
        
        raw_color = req.color_image
        
        raw_depth = req.depth_image
        
        raw_segmask = req.segmask


        # Get the raw camera info as ROS `CameraInfo`.
        raw_camera_info = req.camera_info

        camera_intr = np.array([raw_camera_info.K]).reshape((3, 3))
        try:
            color_im = imgmsg_to_cv2(raw_color)
            depth_im = imgmsg_to_cv2(raw_depth)
            segmask = imgmsg_to_cv2(raw_segmask)
        except NotImplementedError as e:
            rospy.logerr(e)
            
        if depth_im is None:
            err_msg = "Suction Grasp Planner server could not extract a depth image from the ROS message."
            rospy.logerr(err_msg)
            ValueError(err_msg)
            
        return (color_im, depth_im, segmask, camera_intr)


if __name__ == "__main__":
    # init node
    rospy.init_node('suction_grasp_planner')
    rospy.loginfo(
        "Suction GraspNet Planner is launched with Python {}".format(sys.version))

    args = {
        "checkpoint_dir": rospy.get_param('~ckpt_dir'),
        "z_range": np.array([rospy.get_param('~z_min'), rospy.get_param('~z_max')]),
        "top_k": rospy.get_param('~top_k'),
        "visualize": rospy.get_param('~visualize'),
    }

    # start Contact GraspNet Planner service
    GraspPlannerServer(**args)

    rospy.spin()