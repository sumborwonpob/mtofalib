#!/usr/bin/env python3

# I made this.

import cv2
#from cv2 import *

import rclpy
from rclpy.node import Node
from rclpy import qos

import numpy as np
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from mtofalib.msg import Mtof
bridge = CvBridge()

# ::::::::::::::::: Put your parameters here :::::::::::::::::::::::::
# ::::::::::::::::::::: Camera Params :::::::::::::::::::::
# Original image size
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
# New image size
NEW_WIDTH = 1280 # In case you want to resize the image
NEW_HEIGHT = 720 # In case you want to resize the image
# Camera intrinsics (of the original image size)
# Camera matrix (does not support skew by default. If you need it, apply yourself)
Kmat = np.array([[504.98910672340946, 0.0, 629.8239641205832], [0.0, 506.32538446501604, 386.57607630506993], [0.0, 0.0, 1.0]])
# Distortion matrix: Pinhole [k1 k2 p1 p2 k3], Fisheye [k1 k2 k3 k4]
Dmodel = "fisheye" # pinhole or fisheye
Dmat = np.array([[-0.09530407046832189], [0.06061897620514853], [-0.04570676901823003], [0.015049902686400237]])
# ::::::::::::::::::::: ROS-related Params ::::::::::::::::::::::
IMAGE_TOPIC = "/camera/compressed"
MTOF_TOPIC = "/mtof/data"
# ::::::::::::::::::::: MToF Params :::::::::::::::::::::
T_C_MTOF = np.array([-0.025, 0.0, -0.02]) # MToF's position in the camera's coordinate frame in meters (X right, Y down, Z front)
R_C_MTOF = np.array([2.47845104, 0.22337276, 1.7083604 ]) * 3.1417 / 180  #convert deg to rad
VIS_DEPTH_IMAGE_SIZE = 400 # How big debug depth image in pixels (for debugging and for centroid calculation)
PHI_WH = 45 * 3.1417/180 # FOV of MToF in rad 45 deg = 0.785398 rad
NEED_ROTATE = 1 # 0 = No rotation, 1-3 integer for number of times to rotate CW
# ::::::::::::::::::::: ROS-related Params ::::::::::::::::::::::
IMAGE_COMPRESSED = True
IMAGE_TOPIC = "/camera/compressed"
MTOF_TOPIC = "/mtof/data"
# ::::::::::::::::::::: Visualization :::::::::::::::::::::
# Depth TURBO color-map parameters
VIS_MAX_RANGE = 1.0 # Range in meters which will be in color blue
VIS_MIN_RANGE = 0.1 # Range in meters which will be in Color red

# Input result 
class ShowResult(Node):
    depth_data = None
    zone_res = 0

    Kmat_new = None

    font_scale = 1

    def __init__(self):
        super().__init__('mtofal')
    # --------- Publishers ------------
        # Debug images publishers
        self.debug_pubber = self.create_publisher(CompressedImage, '/mtofal/debug/aruco/compressed', qos.qos_profile_system_default)
        self.depth_pubber = self.create_publisher(CompressedImage, '/mtofal/debug/depth/compressed', qos.qos_profile_system_default)

        self.image_subber = self.create_subscription(CompressedImage, IMAGE_TOPIC, self.image_callback, qos_profile=qos.qos_profile_sensor_data)
        self.tof_subber = self.create_subscription(Mtof, MTOF_TOPIC, self.tof_callback, qos_profile=qos.qos_profile_sensor_data)

        Kmat[0][0] = Kmat[0][0] * (NEW_WIDTH / IMAGE_WIDTH)
        Kmat[0][2] = Kmat[0][2] * (NEW_WIDTH / IMAGE_WIDTH)
        Kmat[1][1] = Kmat[1][1] * (NEW_HEIGHT / IMAGE_HEIGHT)
        Kmat[1][2] = Kmat[1][2] * (NEW_HEIGHT / IMAGE_HEIGHT)
        self.Kmat_new = Kmat

        print(self.Kmat_new)
        print(Dmat)

        # ----------- Calculate undist map ------------------
        self.rectmap1, self.rectmap2 = cv2.fisheye.initUndistortRectifyMap(Kmat, Dmat, None, Kmat, (NEW_WIDTH,NEW_HEIGHT), cv2.CV_32FC1)


    def image_callback(self, msg):
        frame = bridge.compressed_imgmsg_to_cv2(msg)
        frame_debug = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # frame_debug = cv2.fisheye.undistortImage(frame_debug, self.Kmat_new, Dmat, Knew=self.Kmat_new)
        frame_debug = cv2.remap(frame_debug, self.rectmap1, self.rectmap2, cv2.INTER_NEAREST)

        if(self.zone_res != 0):
            # Project mtof depth data to 3D
            angle_per_zone = PHI_WH/self.zone_res
            point3d_list = []
            for i in range(self.zone_res):
                for j in range(self.zone_res):
                    distance = self.depth_data[i][j]
                    # Start from top-left
                    azimuth = ((j * angle_per_zone) - (PHI_WH/2) + (angle_per_zone/2))
                    elevation = -((i * angle_per_zone) - (PHI_WH/2) + (angle_per_zone/2))
                    zone_angle = R.from_euler("xyz", [elevation, azimuth, 0], degrees=False)
                    zone_point3d = zone_angle.apply(np.array([0.0, 0.0, distance]))
                    point3d_list.append(zone_point3d)
                    
            # Project those points to 2D frame using provided translation and calculated rotation
            zone_imgpoint, hole_jacobian = cv2.fisheye.projectPoints(np.array([point3d_list]).reshape(1,64,3), R_C_MTOF, T_C_MTOF, self.Kmat_new, Dmat)
            zone_imgpoint_undist = cv2.fisheye.undistortPoints(np.array(zone_imgpoint).reshape(64,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)

            # Draw them
            for i in range(len(zone_imgpoint_undist)):
                imgpt = zone_imgpoint_undist[i][0]
                dist = self.depth_data[int(i/self.zone_res)][i%self.zone_res]
                color = self.calculateColorMap(dist)
                cv2.circle(frame_debug, (int(imgpt[0]), int(imgpt[1])), int(30), color, int(2*self.font_scale))

        image_message = bridge.cv2_to_compressed_imgmsg(frame_debug)
        image_message.header.stamp = self.get_clock().now().to_msg()
        image_message.header.frame_id = "camera_link"
        self.debug_pubber.publish(image_message)

    def tof_callback(self, msg):
        # Process depth into the array
        process_img_size = VIS_DEPTH_IMAGE_SIZE
        self.zone_res = msg.zone_width

        depth_img_pixels = int(process_img_size/self.zone_res)
        depth_img =  np.zeros((process_img_size, process_img_size, 3), dtype = np.uint8)
        depth_data = np.array(msg.zone_distance_mm)/1000.0
        depth_data = np.reshape(depth_data, (self.zone_res, self.zone_res))

        # Rotate
        for i in range(NEED_ROTATE):
            depth_data = np.rot90(depth_data, k=1)

        # Store 
        self.depth_data = depth_data

# ------------- Function for calculating color map for depth image ---------------
    def calculateColorMap(self, z):
        depth_normalized = 0
        b = 0
        g = 0
        r = 0
        if(z <= VIS_MIN_RANGE):
            depth_normalized = 0
        elif(z >= VIS_MAX_RANGE):
            depth_normalized = 1023
        else:
            depth_normalized = int(((z-VIS_MIN_RANGE)/(VIS_MAX_RANGE-VIS_MIN_RANGE))*1023)

        if(depth_normalized < 256):
            b = 0
            g = depth_normalized
            r = 255
        elif(depth_normalized >= 256 and depth_normalized < 512):
            b = 0
            g = 255
            r = 255 - (depth_normalized-256)
        elif(depth_normalized >= 512 and depth_normalized < 768):
            b = (depth_normalized-512)
            g = 255
            r = 0
        elif(depth_normalized >= 768 and depth_normalized < 1024):
            b = 255
            g = 255 - (depth_normalized-768)
            r = 0
        return [b ,g ,r]
    

def main(args=None):
    rclpy.init(args=args)
    mtof_caliber = ShowResult()
    rclpy.spin(mtof_caliber)
    mtof_caliber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

