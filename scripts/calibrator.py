#!/usr/bin/env python3

# I made this.

import cv2
#from cv2 import *

import rclpy
from rclpy.node import Node
from rclpy import qos

import math
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.cluster.vq import kmeans, vq
from scipy import signal
from scipy.spatial.transform import Rotation as R
from scipy.signal import correlate
from scipy.interpolate import interp1d
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from mtofalib.msg import Mtof
bridge = CvBridge()

print("OpenCV version: ", cv2.__version__)
print("Matplotlib version: ", matplotlib.__version__)

# ::::::::::::::::: Put your parameters here :::::::::::::::::::::::::
# ::::::::::::::::::::: Camera Params :::::::::::::::::::::
# Original image size
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 800
# New image size
NEW_WIDTH = 1280 # In case you want to resize the image
NEW_HEIGHT = 800 # In case you want to resize the image
# Camera intrinsics (of the original image size)
# Camera matrix (does not support skew by default. If you need it, apply yourself)
Kmat = np.array([
    [575.890910, 0.000000, 649.963370],
    [0.000000, 580.746951, 355.605746],
    [0.000000, 0.000000, 1.000000]
])
# Distortion matrix: Pinhole [k1 k2 p1 p2 k3], Fisheye [k1 k2 k3 k4]
Dmodel = "pinhole" # pinhole or fisheye
Dmat = np.array([-0.247104, 0.035979, 0.000866, -0.002558, 0.000000])
# ::::::::::::::::::::: MToF Params :::::::::::::::::::::
T_C_MTOF = np.array([-0.025, 0.0, -0.02]) # Measured MToF's position in the camera's coordinate frame in meters (X right, Y down, Z front)
VIS_DEPTH_IMAGE_SIZE = 400 # How big debug depth image in pixels (for debugging and for centroid calculation)
PHI_WH = 45 * 3.14159265359/180 # FOV of MToF in deg (converrted to rad) VL53L5CX/VL53L8CX = 45 deg
NEED_ROTATE = 3 # 0 = No rotation, 1-3 integer for number of times to rotate CW
# Target is to aim for the depth image to arrange from top-left to bottom-right, same as image pixel arrangement
# ::::::::::::::::::::: ROS-related Params ::::::::::::::::::::::
IMAGE_COMPRESSED = True
IMAGE_TOPIC = "/vo/camera/compressed"
MTOF_TOPIC = "/mtof/data"
# ::::::::::::::::::::: Visualization :::::::::::::::::::::
# Depth TURBO color-map parameters
VIS_MAX_RANGE = 1.0 # Range in meters which will be in color blue
VIS_MIN_RANGE = 0.1 # Range in meters which will be in Color red
# ::::::::::::::::::::: Calib Params ::::::::::::
NUM_DATA_POINTS = 200 # How many data should be collected for the calibration (recommeded more than 100 with sufficient movement)
# ::::::::::::::::::::: Put the dimension of your calibration target ::::::::::::
ARUCO_DICT = cv2.aruco.DICT_APRILTAG_16h5 # Change according to your calib target
# See repo readme for details, unit in meters
object_points = np.array([
    # Marker 11
    [-0.07, -0.01, 0.00], # 0
    [-0.13, -0.01, 0.00], # 1
    [-0.13, -0.07, 0.00], # 2
    [-0.07, -0.07, 0.00], # 3
    # Marker 12
    [-0.07, 0.07, 0.00], # 4
    [-0.13, 0.07, 0.00], # 5
    [-0.13, 0.01, 0.00], # 6
    [-0.07, 0.01, 0.00], # 7
    # Marker 13
    [0.13, -0.01, 0.00], # 8
    [0.07, -0.01, 0.00], # 9
    [0.07, -0.07, 0.00], # 10
    [0.13, -0.07, 0.00], # 11
    # Marker 14
    [0.13, 0.07, 0.00], # 12
    [0.07, 0.07, 0.00], # 13
    [0.07, 0.01, 0.00], # 14
    [0.13, 0.01, 0.00], # 15
    # Origin
    [0.0, 0.0, 0.0], # 16
    # Inner Edges
    [0.06, 0.06, 0.0], # 17 BL
    [-0.06, 0.06, 0.0], # 18 BR
    [-0.06, -0.06, 0.0], # 19 TR
    [0.06, -0.06, 0.0], # 20 TL
    # Outer Edges
    [0.16, 0.12, 0.0], # 21 BL
    [-0.16, 0.12, 0.0], # 22 BR
    [-0.16, -0.115, 0.0], # 23 TR
    [0.16, -0.115, 0.0], # 24 TL
])

class Mtofal(Node):
    # :::::::::::::::::::::: Misc :::::::::::::::::::::::
    depth_data = None
    zone_res = 0

    new_tof_data = False
    getting_data = True
    calibration_done = False

    font_scale = 1

    # :::::::::::::::::::::: Calculation variables ::::::::::::::::::
    depth_ts_array = []
    depth_center_array = []
    image_ts_array = []
    image_center_array = []
    image_xyz_array = []
    image_rpy_array = []
    depth_3d_array = []
    image_pixel_array = []
    hole_time = None
    hole_center = None
    undist_imgPoints = None

    # :::::::::::::::::::::: Calib result :::::::::::::::::::::::::::
    result_rvec = None

    # :::::::::::::::::::::: Camera Params :::::::::::::::::::::::::::::::
    Kmat_new = None
    rectmap1 = None
    rectmap2 = None

    # ::::::::::::::::::::::::::::::::::::::::::: CONSTRUCTOR :::::::::::::::::::::::::::::::::::::::::
    def __init__(self):
        super().__init__('mtofal')
        
        print("Initializing program....")

        # --------- Publishers ------------
        # Debug images publishers
        self.debug_pubber = self.create_publisher(CompressedImage, '/mtofal/debug/aruco/compressed', qos.qos_profile_system_default)
        self.depth_pubber = self.create_publisher(CompressedImage, '/mtofal/debug/depth/compressed', qos.qos_profile_system_default)

        # ------------ Subscribers --------
        if(IMAGE_COMPRESSED):
            self.image_subber = self.create_subscription(CompressedImage, IMAGE_TOPIC, self.image_callback, qos_profile=qos.qos_profile_sensor_data)
        else:
            self.image_subber = self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, qos_profile=qos.qos_profile_sensor_data)

        self.tof_subber = self.create_subscription(Mtof, MTOF_TOPIC, self.tof_callback, qos_profile=qos.qos_profile_sensor_data)

        # Initialize Aruco dict
        self.dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT)

        # ------------ Calculate new Kmat -------------------
        print(Kmat)
        Kmat[0][0] = Kmat[0][0] * (NEW_WIDTH / IMAGE_WIDTH)
        Kmat[0][2] = Kmat[0][2] * (NEW_WIDTH / IMAGE_WIDTH)
        Kmat[1][1] = Kmat[1][1] * (NEW_HEIGHT / IMAGE_HEIGHT)
        Kmat[1][2] = Kmat[1][2] * (NEW_HEIGHT / IMAGE_HEIGHT)
        self.Kmat_new = Kmat

        # ----------- Calculate undist map ------------------
        if(Dmodel != "fisheye" and Dmodel != "pinhole"):
            print("Not supported distortion model: ", Dmodel)
            print("Use either fisheye or pinhole model")
            exit()
        if(Dmodel == "pinhole"):
            self.rectmap1, self.rectmap2 = cv2.initUndistortRectifyMap(Kmat, Dmat, None, Kmat, (NEW_WIDTH,NEW_HEIGHT), cv2.CV_32FC1)
        if(Dmodel == "fisheye"):
            self.rectmap1, self.rectmap2 = cv2.fisheye.initUndistortRectifyMap(Kmat, Dmat, None, Kmat, (NEW_WIDTH,NEW_HEIGHT), cv2.CV_32FC1)

        #-------------- Calculate display font scale -----------
        self.font_scale = NEW_WIDTH/640

        print("")
        # ---------- Dewa, hajime mashou ka? ----------
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("::::::::::::::::::::::::: Mtofalib :::::::::::::::::::::::::")
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

        # ---------- Print Calib Info ------------------
        print("Distortion Model: ", Dmodel)
        print("Camera matrix: ", Kmat)
        print("Distortion matrix: ", Dmat)
        print("")
        #user_input = input("Type something and press enter when ready: ") # Type something, doesn't matter. It will go even if you say no.
        #if(user_input == "something"):
        #    print("Haha. Very funny.")

    # ::::::::::::::::::::::::::::::::::: Image Callback ::::::::::::::::::::::::::::::::::::::
    def image_callback(self, msg):
        # Acquire cv Mat
        if(IMAGE_COMPRESSED):
            frame = bridge.compressed_imgmsg_to_cv2(msg)
        else:
            frame = bridge.imgmsg_to_cv2(msg)
        # resize to new size
        frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))
        # detect Aruco in the image
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, self.dictionary)
        frame_debug = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame_debug = cv2.remap(frame_debug, self.rectmap1, self.rectmap2, cv2.INTER_NEAREST)


        # --------------------------- Calibration Phrase ---------------------------------------------
        if(not self.calibration_done):
            # Get the image points and append object points
            objPoints_now = []
            imgPoints_now = []
            if(ids is not None and self.getting_data):
                for i in range(len(ids)):
                    if(ids[i] == 11):
                        objPtsIdxStart = 0
                    if(ids[i] == 12):
                        objPtsIdxStart = 4
                    if(ids[i] == 13):
                        objPtsIdxStart = 8
                    if(ids[i] == 14):
                        objPtsIdxStart = 12
                    for j in range(4):
                        objPoints_now.append(object_points[objPtsIdxStart+j])
                        imgPoints_now.append([corners[i][0][j][0], corners[i][0][j][1]])
                # Undistort object points
                if(Dmodel == "pinhole"):
                    self.undist_imgPoints = cv2.undistortPoints(np.array(imgPoints_now), self.Kmat_new, Dmat)
                if(Dmodel == "fisheye"):
                    self.undist_imgPoints = cv2.fisheye.undistortPoints(np.array(imgPoints_now).reshape(1, len(imgPoints_now), 2), self.Kmat_new, Dmat)
                # Solve for pose of the camera w.r.t. to board origin (center 0,0 in your object points)
                retval, rvec, tvec, _ = cv2.solvePnPRansac(np.array(objPoints_now), np.array(self.undist_imgPoints), np.eye(3,3), np.zeros((1,5)))
                # Draw in the frame
                cv2.drawFrameAxes(frame_debug, self.Kmat_new, Dmat, rvec, tvec, 0.06)

                # Project the rest of the board that is not inside the image
                if(Dmodel == "pinhole"):
                    self.imgePointsProjected, jacobian_ = cv2.projectPoints(np.array(object_points).reshape(len(object_points), 1, 3), rvec, tvec, self.Kmat_new, Dmat)
                    self.undistortedNormVec = cv2.undistortPoints(self.imgePointsProjected, self.Kmat_new, Dmat)
                    self.imgePointsProjected = cv2.undistortPoints(self.imgePointsProjected, self.Kmat_new, Dmat, None, self.Kmat_new)
                if(Dmodel == "fisheye"):
                    self.imgePointsProjected, jacobian_ = cv2.fisheye.projectPoints(np.array(object_points).reshape(len(object_points), 1, 3), rvec, tvec, self.Kmat_new, Dmat)
                    self.undistortedNormVec = cv2.fisheye.undistortPoints(self.imgePointsProjected, self.Kmat_new, Dmat)
                    self.imgePointsProjected = cv2.fisheye.undistortPoints(self.imgePointsProjected, self.Kmat_new, Dmat, None, self.Kmat_new)

                # Draw debug
                for i in range(self.imgePointsProjected.shape[0]):
                    if(i == 0 or i == 4):
                        marker_type = cv2.MARKER_TRIANGLE_UP
                        cv2.drawMarker(frame_debug, (int(self.imgePointsProjected[i][0][0]), int(self.imgePointsProjected[i][0][1])), (0, 0, 255), marker_type, int(20*self.font_scale))
                    if(i == 8 or i == 12):
                        marker_type = cv2.MARKER_TRIANGLE_DOWN
                        cv2.drawMarker(frame_debug, (int(self.imgePointsProjected[i][0][0]), int(self.imgePointsProjected[i][0][1])), (0, 0, 255), marker_type, int(20*self.font_scale))
                    cv2.drawMarker(frame_debug, (int(self.imgePointsProjected[i][0][0]), int(self.imgePointsProjected[i][0][1])), (0, 255, 0), cv2.MARKER_SQUARE, int(15*self.font_scale))
                cv2.line(frame_debug, (int(Kmat[0][2]), int(Kmat[1][2])), ((int(self.imgePointsProjected[16][0][0]), int(self.imgePointsProjected[16][0][1]))), (255, 0, 255), int(2*self.font_scale))
                cv2.putText(frame_debug, "%.2fm"%(tvec[2]), (int(NEW_WIDTH/2)-50,int(NEW_HEIGHT/2)-10),  cv2. FONT_HERSHEY_PLAIN, 2*self.font_scale, (0, 255, 0), int(3*self.font_scale), cv2.LINE_AA)
                

                # Save data
                if(self.hole_center is not None and not (math.isnan(self.hole_center[0]) or math.isnan(self.hole_center[1])) and self.new_tof_data):
                    # Project center point to the image using initially provided position
                    # Calculate 3D position from MToF
                    mtof_aziele = R.from_euler("xyz",[-self.hole_center[1], self.hole_center[0], 0.0], degrees=False)
                    hole_point3d = mtof_aziele.apply(np.array([0.0, 0.0, tvec[2,0]]))

                    if(Dmodel == "pinhole"):
                        # Project that 3D point into image frame with provided relative position and zero rotation
                        self.hole_imgpoint, hole_jacobian = cv2.projectPoints(np.array([hole_point3d]).reshape(1,1,3), np.array([0.0, 0.0, 0.0]), T_C_MTOF, self.Kmat_new, Dmat)
                        # Convert to normalized 2D vector
                        self.hole_imgpoint_undist = cv2.undistortPoints(np.array(self.hole_imgpoint).reshape(1,1,2), self.Kmat_new, Dmat)[0][0]
                    if(Dmodel == "fisheye"):
                        # Project that 3D point into image frame with provided relative position and zero rotation
                        self.hole_imgpoint, hole_jacobian = cv2.fisheye.projectPoints(np.array([hole_point3d]).reshape(1,1,3), np.array([0.0, 0.0, 0.0]), T_C_MTOF, self.Kmat_new, Dmat)
                        # Convert to normalized 2D vector
                        self.hole_imgpoint_undist = cv2.fisheye.undistortPoints(np.array(self.hole_imgpoint).reshape(1,1,2), self.Kmat_new, Dmat)[0][0]

                    # Convert 2D vector to 3D vector by adding constant z=1 and nromalize the vector
                    image_vec = np.array([self.undistortedNormVec[16][0][0], self.undistortedNormVec[16][0][1], 1])
                    mtof_vec = np.array([self.hole_imgpoint_undist[0], self.hole_imgpoint_undist[1], 1])
                    image_vec = image_vec/np.linalg.norm(image_vec)
                    mtof_vec = mtof_vec/np.linalg.norm(mtof_vec)
                    self.image_xyz_array.append(tvec)
                    self.image_rpy_array.append(rvec)
                    self.image_center_array.append(image_vec)
                    self.depth_center_array.append(mtof_vec)
                    self.depth_3d_array.append(hole_point3d)
                    self.image_pixel_array.append(self.imgePointsProjected[16][0])
                    self.image_ts_array.append(msg.header.stamp.sec + (msg.header.stamp.nanosec*1e-9))
                    self.depth_ts_array.append(self.hole_time)
                    self.new_tof_data = False
                    print("  Data Points collected: ", len(self.image_center_array),"/",NUM_DATA_POINTS)

                    # If there's enough data, initiate calibration
                    if(len(self.image_center_array) >= NUM_DATA_POINTS):
                        self.getting_data = False
                        print("  Enough data taken!")
                        print("Doing calibration.....")
                        self.calibrate_now()
        # -------------------------- Calibration done! Show result phrase ------------------------
        else:
            # Project mtof depth data to 3D
            angle_per_zone = PHI_WH/self.zone_res
            col = 0
            row = 0
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
            
            if(Dmodel == "pinhole"):
                self.zone_imgpoint, hole_jacobian = cv2.projectPoints(np.array([point3d_list]).reshape(1,64,3), self.result_rvec, T_C_MTOF, self.Kmat_new, Dmat)
                self.zone_imgpoint_undist = cv2.undistortPoints(np.array(self.zone_imgpoint).reshape(64,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
            if(Dmodel == "fisheye"):
                # Project those points to 2D frame using provided translation and calculated rotation
                self.zone_imgpoint, hole_jacobian = cv2.fisheye.projectPoints(np.array([point3d_list]).reshape(1,64,3), self.result_rvec, T_C_MTOF, self.Kmat_new, Dmat)
                self.zone_imgpoint_undist = cv2.fisheye.undistortPoints(np.array(self.zone_imgpoint).reshape(64,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)

            # Draw them
            for i in range(len(self.zone_imgpoint_undist)):
                imgpt = self.zone_imgpoint_undist[i][0]
                dist = self.depth_data[int(i/self.zone_res)][i%self.zone_res]
                color = self.calculateColorMap(dist)
                cv2.circle(frame_debug, (int(imgpt[0]), int(imgpt[1])), int(15*self.font_scale), color, int(2*self.font_scale))

        image_message = bridge.cv2_to_compressed_imgmsg(frame_debug)
        image_message.header.stamp = self.get_clock().now().to_msg()
        image_message.header.frame_id = "camera_link"
        self.debug_pubber.publish(image_message)
        #pass

    # :::::::::::::::::::::::::::::::::::: MTOF Callback :::::::::::::::::::::::::::::::::::::::
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

        # Calculate center of mass
        # Clustering between the board and the hole
        centroids, _ = kmeans(depth_data.flatten(), 2)
        segmented, _ = vq(depth_data.flatten(), centroids)
        depth_segmented = np.reshape(segmented, (self.zone_res,self.zone_res))
        # Rearrage the index
        if(centroids[0] < centroids[1]):
            depth_segmented = depth_segmented<1
        # Find the hole center
        self.hole_time = msg.header.stamp.sec+(msg.header.stamp.nanosec*1e-9)
        hole_center = (np.array(center_of_mass(depth_segmented))-(self.zone_res/2))*(PHI_WH/self.zone_res) # Result is in radians
        self.hole_center = np.array([hole_center[1], hole_center[0]]) # Swap row cols

        # Draw depth image
        for i in range(self.zone_res):
            for j in range(self.zone_res):
                xx = i
                yy = j
                b,g,r = self.calculateColorMap(depth_data[xx][yy])
                px = int(yy*depth_img_pixels)
                py = int(xx*depth_img_pixels)
                cv2.rectangle(depth_img, 
                    (px, py), 
                    (px+depth_img_pixels, py+depth_img_pixels),
                    (b,g,r),
                    -1
                )
                cv2.putText(depth_img, str((xx*8)+yy), (px,py+int(depth_img_pixels/2)), 1, 2.0, (0,0,0), 2)
        
        # Draw debug lines during pitch/yaw calib
        if(self.hole_center is not None and not (math.isnan(self.hole_center[0]) or math.isnan(self.hole_center[1]))):
            actual_center = (self.hole_center*self.zone_res/PHI_WH)+(self.zone_res/2)
            cv2.drawMarker(depth_img, (int(actual_center[0]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res)), int(actual_center[1]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res))), (255, 0, 255), cv2.MARKER_CROSS, 20, 5)
            cv2.line(depth_img, (int(VIS_DEPTH_IMAGE_SIZE/2), int(VIS_DEPTH_IMAGE_SIZE/2)), (int(actual_center[0]*VIS_DEPTH_IMAGE_SIZE/self.zone_res), int(actual_center[1]*VIS_DEPTH_IMAGE_SIZE/self.zone_res)), (255, 0, 255), 2)

        #  Make image and publish
        self.new_tof_data = True
        image_message = bridge.cv2_to_compressed_imgmsg(depth_img)
        image_message.header.stamp = self.get_clock().now().to_msg()
        self.depth_pubber.publish(image_message)
    
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
    
    # ----------------- Function for calibration -----------------------------
    def calibrate_now(self):
        
        # ================================================================
        # =================== 1st-pass calibration =======================
        # ================================================================
        print("First pass: initial calibration...")

        # Get timestamp
        depth_timestamp = np.array(self.depth_ts_array)
        image_timestamp = np.array(self.image_ts_array)

        # Calculate common time space and interpolate
        common_time = np.linspace(max(depth_timestamp[0], image_timestamp[0]), min(depth_timestamp[-1], image_timestamp[-1]), NUM_DATA_POINTS*2)
        # board center x
        depth_x_inter = interp1d(depth_timestamp, np.array(self.depth_center_array)[:,0], kind='linear', fill_value='extrapolate')(common_time) 
        img_x_inter = interp1d(image_timestamp, np.array(self.image_center_array)[:,0], kind='linear', fill_value='extrapolate')(common_time) 
        # board center y
        depth_y_inter = interp1d(depth_timestamp, np.array(self.depth_center_array)[:,1], kind='linear', fill_value='extrapolate')(common_time) 
        img_y_inter = interp1d(image_timestamp, np.array(self.image_center_array)[:,1], kind='linear', fill_value='extrapolate')(common_time) 
        # board center z (unit)
        depth_z_inter = interp1d(depth_timestamp, np.array(self.depth_center_array)[:,2], kind='linear', fill_value='extrapolate')(common_time) 
        img_z_inter = interp1d(image_timestamp, np.array(self.image_center_array)[:,2], kind='linear', fill_value='extrapolate')(common_time) 
        # 3D board pos of AR marker
        pos3d_x_inter = interp1d(image_timestamp, np.array(self.image_xyz_array)[:,0].T, kind='linear', fill_value='extrapolate')(common_time) 
        pos3d_y_inter = interp1d(image_timestamp, np.array(self.image_xyz_array)[:,1].T, kind='linear', fill_value='extrapolate')(common_time) 
        pos3d_z_inter = interp1d(image_timestamp, np.array(self.image_xyz_array)[:,2].T, kind='linear', fill_value='extrapolate')(common_time) 
        # 3D board rot of AR marker
        pos3d_rx_inter = interp1d(image_timestamp, np.array(self.image_rpy_array)[:,0].T, kind='linear', fill_value='extrapolate')(common_time) 
        pos3d_ry_inter = interp1d(image_timestamp, np.array(self.image_rpy_array)[:,1].T, kind='linear', fill_value='extrapolate')(common_time) 
        pos3d_rz_inter = interp1d(image_timestamp, np.array(self.image_rpy_array)[:,2].T, kind='linear', fill_value='extrapolate')(common_time) 
        # 2D board center image point
        meas_img_x_inter = interp1d(image_timestamp, np.array(self.image_pixel_array)[:,0].T, kind='linear', fill_value='extrapolate')(common_time) 
        meas_img_y_inter = interp1d(image_timestamp, np.array(self.image_pixel_array)[:,1].T, kind='linear', fill_value='extrapolate')(common_time) 
        # 3D board center mtof point
        meas_mtof_x_inter = interp1d(image_timestamp, np.array(self.depth_3d_array)[:,0].T, kind='linear', fill_value='extrapolate')(common_time) 
        meas_mtof_y_inter = interp1d(image_timestamp, np.array(self.depth_3d_array)[:,1].T, kind='linear', fill_value='extrapolate')(common_time) 
        meas_mtof_z_inter = interp1d(image_timestamp, np.array(self.depth_3d_array)[:,2].T, kind='linear', fill_value='extrapolate')(common_time) 

        # Apply filter to the data
        b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.
        depth_x_filt = signal.filtfilt(b, a, depth_x_inter, method="gust")
        img_x_filt = signal.filtfilt(b, a, img_x_inter, method="gust")
        depth_y_filt = signal.filtfilt(b, a, depth_y_inter, method="gust")
        img_y_filt = signal.filtfilt(b, a, img_y_inter, method="gust")
        depth_z_filt = signal.filtfilt(b, a, depth_z_inter, method="gust")
        img_z_filt = signal.filtfilt(b, a, img_z_inter, method="gust")

        # Compose vector
        depth_vec_list = np.array([depth_x_filt, depth_y_filt, depth_z_filt]).T
        image_vec_list = np.array([img_x_filt, img_y_filt, img_z_filt]).T

        # Calculate SVD
        H = depth_vec_list.T @ image_vec_list
        U, _, Vt = np.linalg.svd(H)
        RotMat = Vt.T @ U.T
        RotR = R.from_matrix(RotMat)

        self.result_rvec = RotR.as_euler("xyz", degrees=False)
        print("        Pre-calib result", RotR.as_euler("xyz", degrees=True), "deg")
        print("    Done!")

        # ================================================================
        # ===================== Outlier filtering ========================
        # ================================================================
        print("Outlier detection and filtering...")

        # Get 3D position at each measurement
        pos3d_vec_list = np.array([pos3d_x_inter, pos3d_y_inter, pos3d_z_inter]).T
        rot3d_vec_list = np.array([pos3d_rx_inter, pos3d_ry_inter, pos3d_rz_inter]).T

        # Project position of outer edges to image plane
        # Find 4 outer edges of the calib board to objpoints
        objpts_norm = np.linalg.norm(object_points, axis=1)
        sorted_id = np.argsort(objpts_norm)
        outer_edges = [object_points[sorted_id[-1]], object_points[sorted_id[-2]], object_points[sorted_id[-3]], object_points[sorted_id[-4]]]

        # Find bounding box for MToF sensor in the image correlate to board distance measured from AR marker
        # Get Z-axis only from marker estimation
        bdbox_vec_list = pos3d_vec_list
        # bdbox_vec_list[:, 0, 0] = 0.0
        # bdbox_vec_list[:, 0, 1] = 0.0
        # Compose object points of the 4 edges of the MToF
        half_angle = PHI_WH/2
        mtof_edges_rad = np.array([
            # yaw, pitch
            [-half_angle, half_angle], # top left
            [-half_angle, -half_angle], # bottom left
            [half_angle, -half_angle], # bottom right
            [half_angle, half_angle], # top right
        ])
        mtof_bd_imgpts_list = []
        i = 0
        for depth3d in bdbox_vec_list:
            bd_corner_list = []
            # Get bounding box 3D vector
            for mtof_edge in mtof_edges_rad:
                zone_angle = R.from_euler("xyz", [mtof_edge[0], mtof_edge[1], 0], degrees=False)
                zone_points3d = zone_angle.apply(np.array([0.0,0.0,depth3d[0][2]]))
                bd_corner_list.append(zone_points3d)

            # Project them to image
            if(Dmodel == "pinhole"):
                bd_imgpoint, hole_jacobian = cv2.projectPoints(np.array([bd_corner_list]).reshape(1,4,3), self.result_rvec, T_C_MTOF, self.Kmat_new, Dmat)
                bd_imgpoint_undist = cv2.undistortPoints(np.array(bd_imgpoint).reshape(4,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
                mtof_bd_imgpts_list.append(bd_imgpoint_undist)
            if(Dmodel == "fisheye"):
                # Project those points to 2D frame using provided translation and calculated rotation
                bd_imgpoint, hole_jacobian = cv2.fisheye.projectPoints(np.array([bd_corner_list]).reshape(1,4,3), self.result_rvec, T_C_MTOF, self.Kmat_new, Dmat)
                bd_imgpoint_undist = cv2.fisheye.undistortPoints(np.array(bd_imgpoint).reshape(4,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
                mtof_bd_imgpts_list.append(bd_imgpoint_undist)
        
        # Project outer edges
        valid_data_idx = []
        for i in range(len(rot3d_vec_list)):
            this_tvec = pos3d_vec_list[i]
            this_rvec = rot3d_vec_list[i]
            this_bounding_box = mtof_bd_imgpts_list[i]

            # Get image point of the corners
            if(Dmodel == "pinhole"):
                edges_imgpt,_ = cv2.projectPoints(np.array([outer_edges]).reshape(1,4,3), this_rvec, this_tvec, self.Kmat_new, Dmat)
                edges_imgpt_un = cv2.undistortPoints(edges_imgpt.reshape(4,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
            if(Dmodel == "fisheye"):
                edges_imgpt,_ = cv2.fisheye.projectPoints(np.array([outer_edges]).reshape(1,4,3), this_rvec, this_tvec, self.Kmat_new, Dmat)
                edges_imgpt_un = cv2.fisheye.undistortPoints(edges_imgpt.reshape(4,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
            result_edges = True

            # Clip bounding box to image edge to prevent contour leak
            this_bounding_box = np.clip(this_bounding_box, [0, 0], [NEW_WIDTH-1, NEW_HEIGHT-1])

            # Draw contour image and find contour
            contour_img = np.zeros((NEW_HEIGHT, NEW_WIDTH), dtype=np.uint8)
            cv2.line(contour_img, 
                np.array([int(this_bounding_box[0][0][0]),int(this_bounding_box[0][0][1])]), 
                np.array([int(this_bounding_box[1][0][0]),int(this_bounding_box[1][0][1])]), 
                (255), 3
            )
            cv2.line(contour_img, 
                np.array([int(this_bounding_box[1][0][0]),int(this_bounding_box[1][0][1])]), 
                np.array([int(this_bounding_box[2][0][0]),int(this_bounding_box[2][0][1])]), 
                (255), 3
            )
            cv2.line(contour_img, 
                np.array([int(this_bounding_box[2][0][0]),int(this_bounding_box[2][0][1])]), 
                np.array([int(this_bounding_box[3][0][0]),int(this_bounding_box[3][0][1])]), 
                (255), 3
            )
            cv2.line(contour_img, 
                np.array([int(this_bounding_box[3][0][0]),int(this_bounding_box[3][0][1])]), 
                np.array([int(this_bounding_box[0][0][0]),int(this_bounding_box[0][0][1])]), 
                (255), 3
            )
            contours, _ = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Now test each corner
            for i in range(4):
                edg_pt = edges_imgpt_un[i][0]
                # Do polygon test whether all edges are inside the contour
                result = cv2.pointPolygonTest(contours[0], edg_pt, measureDist=False)
                result_edges = (result_edges and (result >= 0))
            
            # Store result in another vector
            if(result_edges):
                valid_data_idx.append(1)
            else:
                valid_data_idx.append(0)
        print("        Valid sample count", valid_data_idx.count(1)/2, "/", len(valid_data_idx)/2)
        print("    Done!")

        # ================================================================
        # =================== 2nd-pass calibration =======================
        # ================================================================
        print("Second pass: re-calibrate without outliers...")

        # Extract only inliers
        depth_vec_list_inliers = depth_vec_list[np.array(valid_data_idx) > 0]
        image_vec_list_inliers = image_vec_list[np.array(valid_data_idx) > 0]

        # Calculate SVD again
        H = depth_vec_list_inliers.T @ image_vec_list_inliers
        U, _, Vt = np.linalg.svd(H)
        RotMat = Vt.T @ U.T
        RotR = R.from_matrix(RotMat)

        self.result_rvec = RotR.as_euler("xyz", degrees=False)

        print("        First pass result = ", self.result_rvec*180/math.pi)
        print("        Second pass result = ", RotR.as_euler("xyz", degrees=True))
        print("    Done!")

        # ================================================================
        # =================== Projection error cal =======================
        # ================================================================
        print("Calculating projection errors...")

        # Get data after calib
        meas_img_pt = np.array([meas_img_x_inter, meas_img_y_inter]).T
        meas_3d_pt = np.array([meas_mtof_x_inter, meas_mtof_y_inter, meas_mtof_z_inter]).T

        meas_img_pt = meas_img_pt[np.array(valid_data_idx) > 0]
        meas_3d_pt = meas_3d_pt[np.array(valid_data_idx) > 0]
        
        eval_err_x_list = []
        eval_err_y_list = []
        # Project 3D point using obtained calib result
        for i in range(len(meas_img_pt)):
            this_img_pt = meas_img_pt[i]
            this_3d_pt = meas_3d_pt[i]

            if(Dmodel == "pinhole"):
                self.eval_imgpoint, hole_jacobian = cv2.projectPoints(np.array([this_3d_pt]).reshape(1,1,3), self.result_rvec, T_C_MTOF, self.Kmat_new, Dmat)
                self.eval_imgpoint_undist = cv2.undistortPoints(np.array(self.eval_imgpoint).reshape(1,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
            if(Dmodel == "fisheye"):
                # Project those points to 2D frame using provided translation and calculated rotation
                self.eval_imgpoint, hole_jacobian = cv2.fisheye.projectPoints(np.array([this_3d_pt]).reshape(1,1,3), self.result_rvec, T_C_MTOF, self.Kmat_new, Dmat)
                self.eval_imgpoint_undist = cv2.fisheye.undistortPoints(np.array(self.eval_imgpoint).reshape(1,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)

            eval_err_x_list.append(abs(this_img_pt[0]-self.eval_imgpoint_undist[0][0][0]))
            eval_err_y_list.append(abs(this_img_pt[1]-self.eval_imgpoint_undist[0][0][1]))
        
        # Calculate all results
        eval_err_x_arr = np.array(eval_err_x_list)
        eval_err_y_arr = np.array(eval_err_y_list)

        eval_err_x = np.average(eval_err_x_list)
        eval_err_y = np.average(eval_err_y_list)

        eval_err_xy_arr = np.array([eval_err_x_arr, eval_err_y_arr]).T
        eval_err_xy = np.linalg.norm(eval_err_xy_arr, axis = 1)

        # Calculate error in zone size at avg distance used
        avg_dist_used = np.average(pos3d_vec_list[:,0,2])
        # Calculate zone size in pixels at that distance
        angle_per_zone = PHI_WH/self.zone_res
        zone_rot = R.from_euler("xyz", [0.0, angle_per_zone, 0.0], degrees=False)
        zone_size_vec = zone_rot.apply(np.array([0.0, 0.0, avg_dist_used]))
        center_vec = np.array([0.0, 0.0, avg_dist_used])
        if(Dmodel == "pinhole"):
            mtof_center_imgpoint, hole_jacobian = cv2.projectPoints(np.array([center_vec]).reshape(1,1,3), self.result_rvec, (T_C_MTOF), self.Kmat_new, Dmat)
            mtof_center_imgpoint_undist = cv2.undistortPoints(np.array(mtof_center_imgpoint).reshape(1,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
            mtof_zonesize_imgpoint, hole_jacobian = cv2.projectPoints(np.array([zone_size_vec]).reshape(1,1,3), self.result_rvec, (T_C_MTOF), self.Kmat_new, Dmat)
            mtof_zonesize_imgpoint_undist = cv2.undistortPoints(np.array(mtof_zonesize_imgpoint).reshape(1,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
        if(Dmodel == "fisheye"):
            mtof_center_imgpoint, hole_jacobian = cv2.fisheye.projectPoints(np.array([center_vec]).reshape(1,1,3), self.result_rvec, (T_C_MTOF), self.Kmat_new, Dmat)
            mtof_center_imgpoint_undist = cv2.fisheye.undistortPoints(np.array(mtof_center_imgpoint).reshape(1,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
            mtof_zonesize_imgpoint, hole_jacobian = cv2.fisheye.projectPoints(np.array([zone_size_vec]).reshape(1,1,3), self.result_rvec, (T_C_MTOF), self.Kmat_new, Dmat)
            mtof_zonesize_imgpoint_undist = cv2.fisheye.undistortPoints(np.array(mtof_zonesize_imgpoint).reshape(1,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
        zone_pxs = mtof_zonesize_imgpoint_undist - mtof_center_imgpoint
        zone_size_in_px = np.linalg.norm(zone_pxs[0][0])
        # Now see how much error in zone unit
        eval_err_zone_x = eval_err_x/zone_size_in_px
        eval_err_zone_y = eval_err_y/zone_size_in_px
        print("    Done!")

        # Plot error distribution in the image
        fig, ax = plt.subplots(1,1)
        scatter = plt.scatter(meas_img_pt[:,0], meas_img_pt[:,1], c=eval_err_xy, cmap='turbo', s=eval_err_xy/5)
        plt.xlim(0, NEW_WIDTH)
        plt.ylim(0, NEW_HEIGHT)
        plt.grid()
        plt.gca().invert_yaxis()
        ax.set_ylabel("board center in image y [pixel]")
        ax.set_xlabel("board center in image x [pixel]")
        plt.colorbar(scatter, label='Projection error [pixels]')

        # Plot error histrogram distribution
        density = stats.gaussian_kde(eval_err_xy)
        fig, ax = plt.subplots(1,1)
        ax.hist(eval_err_xy, bins=10, linewidth=0.5, edgecolor="white")
        ax.set(xlim=(0, np.max(eval_err_xy)), xticks=np.arange(1, 10)*(np.max(eval_err_xy)/10))
        plt.plot(eval_err_xy, density(eval_err_xy))
        ax.set_ylabel("occurance")
        ax.set_xlabel("Projection error [pixels]")
        ax.set_title("Projection error distribution histrogram")
        
        # Plot raw data
        fig, axs_raw = plt.subplots(2, 1)
        fig.canvas.set_window_title('Uncalibrated Image Point')
        common_time = common_time-common_time[0]
        #axs_raw[0].set_title("Uncalibrated azimuth")
        axs_raw[0].plot(common_time, depth_vec_list[:,0], color=(1.0, 0.0, 0.0), label='MToF [rad]')
        axs_raw[0].plot(common_time, image_vec_list[:,0], color=(0.0, 1.0, 0.0), label='Image [rad]')
        axs_raw[0].legend(loc='best')
        axs_raw[0].set_ylabel("Azimuth uncalibrated [rad]")
        axs_raw[0].set_xlabel("Sample time")
        axs_raw[0].grid()
        #axs_raw[1].set_title("Uncalibrated elevation")
        axs_raw[1].plot(common_time, depth_vec_list[:,1], color=(1.0, 0.0, 0.0), label='MToF [rad]')
        axs_raw[1].plot(common_time, image_vec_list[:,1], color=(0.0, 1.0, 0.0), label='Image [rad]')
        axs_raw[1].legend(loc='best')
        axs_raw[1].set_ylabel("Elevation uncalibrated [rad]")
        axs_raw[1].set_xlabel("Sample time")
        axs_raw[1].grid()

        # Plot corrected Data
        depth_vec_res = RotR.apply(depth_vec_list)
        fig, axs_res = plt.subplots(2, 1)
        fig.canvas.set_window_title('Calibrated Image Point')
        #axs_res[0].set_title("Calibrated azimuth")
        axs_res[0].plot(common_time, depth_vec_res[:,0], color=(1.0, 0.0, 0.0), label='MToF [rad]')
        axs_res[0].plot(common_time, image_vec_list[:,0], color=(0.0, 1.0, 0.0), label='Image [rad]')
        axs_res[0].legend(loc='best')
        axs_res[0].set_ylabel("Azimuth calibrated [rad]")
        axs_res[0].set_xlabel("Sample time")
        axs_res[0].grid()
        #axs_res[1].set_title("Calibrated elevation")
        axs_res[1].plot(common_time, depth_vec_res[:,1], color=(1.0, 0.0, 0.0), label='MToF [rad]')
        axs_res[1].plot(common_time, image_vec_list[:,1], color=(0.0, 1.0, 0.0), label='Image [rad]')
        axs_res[1].legend(loc='best')
        axs_res[1].set_ylabel("Elevation calibrated [rad]")
        axs_res[1].set_xlabel("Sample time")
        axs_res[1].grid()

        print("----------- Calibration result -----------------")
        print("  Roll offset: %.3f degrees"%(RotR.as_euler("xyz", degrees=True)[2]))
        print("  Pitch offset: %.3f degrees"%(RotR.as_euler("xyz", degrees=True)[0]))
        print("  Yaw offset: %.3f degrees"%(RotR.as_euler("xyz", degrees=True)[1]))
        print("  XYZ: %.3f %.3f %.3f degrees"%(RotR.as_euler("xyz", degrees=True)[0], RotR.as_euler("xyz", degrees=True)[1], RotR.as_euler("xyz", degrees=True)[2]))
        print("  Mean projection error x-axis: ", eval_err_x, "pixels")
        print("  Mean projection error y-axis: ", eval_err_y, "pixels")
        print("  Mean projection error x-axis: ", eval_err_zone_x, "zone(s) at", avg_dist_used,"meter depth")
        print("  Mean projection error y-axis: ", eval_err_zone_y, "zone(s) at", avg_dist_used,"meter depth")
        print("Calibration completed! Otsukaresama desu!")
        print()
        print("Close the plots to continue...")
        print("Then you will be able to view calibrated result image in /mtofal/debug/aruco/compressed")
        plt.show()
        

def main(args=None):
    rclpy.init(args=args)
    mtof_caliber = Mtofal()
    rclpy.spin(mtof_caliber)
    mtof_caliber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



