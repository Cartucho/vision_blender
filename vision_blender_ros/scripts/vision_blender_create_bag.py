#!/usr/bin/env python
import os
import glob
import re
import sys
import json

import rospy
import rospkg
import rosbag
from copy import deepcopy
import tf2_ros
import tf2_msgs.msg
import tf_conversions # TODO: not sure if needed

from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
import geometry_msgs.msg
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np

# node class
class CreateBag:
    # ref: https://github.com/amc-nu/RosImageFolderPublisher/blob/master/src/image_folder_publisher/scripts/image_folder_publisher.py
    # ref: https://answers.ros.org/question/11537/creating-a-bag-file-out-of-a-image-sequence/

    def load_blender_info(self):
        ## get ground truth file extension
        self.gt_file_extension = '.npz'
        rospy.loginfo("[{}] (gt_file_extension) blender ground truth file extension set to `{}`".format(self.app_name, self.gt_file_extension))
        ## get camera info
        camera_info_path = os.path.join(self.data_dir_path, 'camera_info.json')
        if not os.path.isfile(camera_info_path):
            rospy.logfatal("[{}] file not found: {}".format(self.app_name, self.camera_info_path))
            sys.exit(1)
        with open(camera_info_path) as json_file:
            camera_info = json.load(json_file)
            ## img format
            img_format_str = camera_info['img_format']
            self.img_format = None
            if img_format_str == 'JPEG':
                self.img_format == '.jpg'
            elif img_format_str == 'BMP':
                self.img_format = '.bmp'
            elif img_format_str == 'TIFF':
                self.img_format = '.tif'
            elif img_format_str == 'PNG':
                self.img_format = '.png'
            else:
                rospy.logfatal("[{}] image format not supported: {}".format(self.app_name, img_format_str))
                sys.exit(1)
            rospy.loginfo("[{}] (img_format) camera image format `{}`".format(self.app_name, self.img_format))
            ## img resolution
            self.img_res_x = camera_info['img_res_x']
            self.img_res_y = camera_info['img_res_y']
            rospy.loginfo("[{}] (img_res_x, img_res_y) camera image resolution - width:{} height:{}".format(self.app_name, self.img_res_x, self.img_res_y))
            ## camera intrinsic matrix
            cam_mat_intr = camera_info['cam_mat_intr']
            self.cam_fx = cam_mat_intr['f_x']
            self.cam_fy = cam_mat_intr['f_y']
            self.cam_cx = cam_mat_intr['c_x']
            self.cam_cy = cam_mat_intr['c_y']
            rospy.loginfo("[{}] (fx, fy, cx, cy) intrinsic camera parameters - fx:{} fy:{} cx:{} cy:{}".format(self.app_name, self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy))
            ## stereo info
            self.is_stereo = camera_info['is_stereo']
            rospy.loginfo("[{}] (is_stereo) is stereo? {}".format(self.app_name, self.is_stereo))
            if self.is_stereo:
                stereo_info = camera_info['stereo_info']
                self.stereo_img_suffix_left = stereo_info['stereo_left_suffix']
                rospy.loginfo("[{}] (stereo_img_suffix_left) blender stereo left camera suffix set to `{}`".format(self.app_name, self.stereo_img_suffix_left))
                self.stereo_img_suffix_right = stereo_info['stereo_right_suffix']
                rospy.loginfo("[{}] (stereo_img_suffix_right) blender stereo right camera suffix set to `{}`".format(self.app_name, self.stereo_img_suffix_right))
                assert (stereo_info['stereo_mode'] == 'PARALLEL')
                assert (stereo_info['stereo_pivot'] == 'LEFT')
                self.baseline = stereo_info['stereo_interocular_distance']
                rospy.loginfo("[{}] (baseline) blender stereo interocular distance set to {} m".format(self.app_name, self.baseline))


    def create_camera_info_messages(self):
        cam_info_d = [0.0, 0.0, 0.0, 0.0, 0.0]
        cam_info_k = [self.cam_fx, 0.0, self.cam_cx,
                           0.0, self.cam_fy, self.cam_cy,
                           0.0, 0.0, 1.0]
        cam_info_r = [1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0]
        cam_info_p = [self.cam_fx, 0.0, self.cam_cx, 0.0,
                           0.0, self.cam_fy, self.cam_cy, 0.0,
                           0.0, 0.0, 1.0, 0.0]

        self.camera_info_msg_left = CameraInfo()
        self.camera_info_msg_left.height = self.img_res_y
        self.camera_info_msg_left.width = self.img_res_x
        self.camera_info_msg_left.distortion_model = 'plumb_bob'
        self.camera_info_msg_left.D = cam_info_d
        self.camera_info_msg_left.K = cam_info_k
        self.camera_info_msg_left.R = cam_info_r
        self.camera_info_msg_left.P = cam_info_p

        if self.is_stereo:
          cam_info_p_r = [self.cam_fx, 0.0, self.cam_cx, -(self.cam_fx * self.baseline),
                               0.0, self.cam_fy, self.cam_cy, 0.0,
                               0.0, 0.0, 1.0, 0.0]
          self.camera_info_msg_right = deepcopy(self.camera_info_msg_left)
          self.camera_info_msg_right.P = cam_info_p_r


    def __init__(self):
        self.app_name = "vision_blender_create_bag"
        self.cv_bridge = CvBridge()

        # ROSBAG config
        # get publish rate parameter
        self.rate = rospy.get_param('~hz', 10) # 10Hz by default
        rospy.loginfo("[{}] (hz) rosbag publish rate set to {} Hz".format(self.app_name, self.rate))
        self.rate_sec = (1.0 / self.rate) # rate in seconds
        # get path parameter to the directory with ground-truth data to be loaded
        rospack = rospkg.RosPack()
        data_dir_path_default = os.path.join(rospack.get_path('vision_blender_ros'), 'data')
        self.data_dir_path = rospy.get_param('~data_dir_path', data_dir_path_default)
        if not os.path.isdir(self.data_dir_path):
            rospy.logfatal("[{}] (data_dir) invalid data folder path: {}".format(self.app_name, self.data_dir_path))
            sys.exit(1)
        rospy.loginfo("[{}] (data_dir) reading ground-truth data from {}".format(self.app_name, self.data_dir_path))
        # BLENDER config
        self.load_blender_info()
        self.create_camera_info_messages()
        # get name of the reference frame for the camera
        self.frame_id = rospy.get_param('~camera_frame_id', '/camera_link')
        # get name for rosbag
        rosbag_name = rospy.get_param('~rosbag_name', 'vision_blender_gt') # TODO: assert that the name ends in .bag
        self.rosbag_path = os.path.join(self.data_dir_path, rosbag_name) # the rosbag will be written in the data folder
        rospy.loginfo("[{}] (rosbag_path) OUTPUT rosbag path set to {}".format(self.app_name, self.rosbag_path))
        # get rostopic names
        if self.is_stereo:
          ## left camera
          self.topic_name_stereo_left_rgb = rospy.get_param('~topic_name_stereo_left_rgb', 'camera/left/rgb/image_raw')
          rospy.loginfo("[{}] (topic_name_stereo_left_rgb) set to {}".format(self.app_name, self.topic_name_stereo_left_rgb))
          self.topic_name_stereo_left_cam_info = rospy.get_param('~topic_name_stereo_left_cam_info', 'camera/left/rgb/camera_info')
          rospy.loginfo("[{}] (topic_name_stereo_left_cam_info) set to {}".format(self.app_name, self.topic_name_stereo_left_cam_info))
          ## right camera
          self.topic_name_stereo_right_rgb = rospy.get_param('~topic_name_stereo_right_rgb', 'camera/right/rgb/image_raw')
          rospy.loginfo("[{}] (topic_name_stereo_right_rgb) set to {}".format(self.app_name, self.topic_name_stereo_right_rgb))
          self.topic_name_stereo_right_cam_info = rospy.get_param('~topic_name_stereo_right_cam_info', 'camera/right/rgb/camera_info')
          rospy.loginfo("[{}] (topic_name_stereo_right_cam_info) set to {}".format(self.app_name, self.topic_name_stereo_right_cam_info))



    def get_file_paths_without_extension(self, dirpath, extension):
        """ Get a list of file paths without their extension """
        # ref: https://stackoverflow.com/questions/50913866/getting-file-names-without-file-extensions-with-glob
        rpattern = re.compile(r'(.*?)\{}'.format(extension))
        target_dir = os.path.join(dirpath, '*')
        file_list = (rpattern.match(f) for f in glob.glob(target_dir))
        file_list = [match.group(1) for match in file_list if match]
        return file_list


    def natural_sort(self, l):
        """ Natural sort order """
        # ref: https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
        return sorted(l, key = alphanum_key)


    def get_list_data_path(self, dirpath):
        """ Get a list of the paths to the ground-truth data """
        gt_file_list = []
        gt_file_list = self.get_file_paths_without_extension(dirpath, self.gt_file_extension)
        if len(gt_file_list) > 0:
            gt_file_list = self.natural_sort(gt_file_list)
        return gt_file_list


    def increment_time(self, t):
        """ Increment time to publish messages at the desired frame rate """
        t_new = t + rospy.Duration(secs=self.rate_sec)
        return t_new


    def read_image(self, img_path):
        """ Read iamge and check if reading was successful """
        img = cv2.imread(img_path)
        if img is None:
            rospy.logfatal("[{}] error reading image file: {}".format(self.app_name, img_path))
            sys.exit(1)
        return img


    def get_header(self, count, t):
        # create header
        h = Header()
        #h.seq = gt_file
        h.seq = count
        h.stamp = t
        h.frame_id = self.frame_id
        return h


    def get_image_data(self, gt_file, suffix, camera_info_msg, h):
        img_path = "{}{}{}".format(gt_file, suffix, self.img_format)
        img = cv2.imread(img_path)
        # convert to image message
        try:
            img_msg = self.cv_bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            rospy.logfatal("[{}] error converting cv image: {}".format(self.app_name, e.message))
            sys.exit(1)
        img_msg.header = h
        camera_info_msg.header = h
        return img_msg, camera_info_msg


    def get_pose_message(self, cam_pose, h):
        tf = geometry_msgs.msg.TransformStamped()
        tf.header = h
        tf.header.frame_id = 'world'
        tf.child_frame_id = self.frame_id
        tf.transform.translation.x = cam_pose[0,3]
        tf.transform.translation.y = cam_pose[1,3]
        tf.transform.translation.z = cam_pose[2,3]
        rot_mat = cam_pose[:,:3]
        rot_mat, jac = cv2.Rodrigues(rot_mat)
        q = tf_conversions.transformations.quaternion_from_euler(rot_mat[0], rot_mat[1], rot_mat[2])
        tf.transform.rotation.x = q[0]
        tf.transform.rotation.y = q[1]
        tf.transform.rotation.z = q[2]
        tf.transform.rotation.w = q[3]
        tf_msg = tf2_msgs.msg.TFMessage([tf])
        return tf_msg


    def get_gt_data(self, gt_file):
        gt_file_path = '{}{}'.format(gt_file, self.gt_file_extension)
        gt_data = np.load(gt_file_path)
        return gt_data


    def write_rosbag(self, gt_file_list):
        """ create rosbag file and write data to it """
        bag = rosbag.Bag(self.rosbag_path, 'w')
        # get current time
        t = rospy.get_rostime()
        # write data
        try:
            count = 0
            for gt_file in gt_file_list:
                rospy.loginfo("[{}] Reading data {}".format(self.app_name, gt_file))
                gt_data = self.get_gt_data(gt_file)
                # write image data
                if self.is_stereo:
                    # get image data
                    h = self.get_header(count, t)
                    img_msg_left, camera_info_msg_left = self.get_image_data(gt_file, self.stereo_img_suffix_left, self.camera_info_msg_left, h)
                    img_msg_right, camera_info_msg_right = self.get_image_data(gt_file, self.stereo_img_suffix_right, self.camera_info_msg_right, h)
                    # get camera pose
                    tf_msg = self.get_pose_message(gt_data['extr'], h)
                    # write data
                    bag.write(self.topic_name_stereo_left_rgb, img_msg_left, t)
                    bag.write(self.topic_name_stereo_left_cam_info, camera_info_msg_left, t)
                    bag.write(self.topic_name_stereo_right_rgb, img_msg_right, t)
                    bag.write(self.topic_name_stereo_right_cam_info, camera_info_msg_right, t)
                    bag.write("/tf", tf_msg, t)
                else:
                    # Monocular camera
                    pass
                count += 1
                t = self.increment_time(t)
        finally:
            bag.close()


    def run(self):
        # get images path
        gt_file_list = self.get_list_data_path(self.data_dir_path)
        if not gt_file_list:
            rospy.logfatal("[{}] no `{}` file found in folder: {}".format(self.app_name, self.gt_file_extension, self.data_dir_path))
            sys.exit(1)
        self.write_rosbag(gt_file_list)
        rospy.loginfo("[{}] Rosbag ready! You cand find it here: {}".format(self.app_name, self.rosbag_path))


# main function
if __name__ == '__main__':
    # initialize node and name it
    rospy.init_node('vision_blender_create_bag', anonymous=True)
    # try to create the rosbag file
    try:
        create_bag = CreateBag()
        create_bag.run()
    except rospy.ROSInterruptException():
        pass
