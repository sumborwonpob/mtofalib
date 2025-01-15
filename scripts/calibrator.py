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
IMAGE_HEIGHT = 720
# New image size
NEW_WIDTH = 1280 # In case you want to resize the image
NEW_HEIGHT = 720 # In case you want to resize the image
# Camera intrinsics (of the original image size)
# Camera matrix (does not support skew by default. If you need it, apply yourself)
Kmat = np.array([
    [588.985878, 0.000000, 640.660494],
    [0.000000, 597.650950, 361.873006],
    [0.000000, 0.000000, 1.000000]
])
# Distortion matrix: Pinhole [k1 k2 p1 p2 k3], Fisheye [k1 k2 k3 k4]
Dmodel = "pinhole" # pinhole or fisheye
Dmat = np.array([-0.257084, 0.037809, 0.002464, -0.000984, 0.000000])
# ::::::::::::::::::::: MToF Params :::::::::::::::::::::
T_C_MTOF = np.array([-0.025, 0.0, -0.02]) # MToF's position in the camera's coordinate frame in meters (X right, Y down, Z front)
VIS_DEPTH_IMAGE_SIZE = 400 # How big debug depth image in pixels (for debugging and for centroid calculation)
PHI_WH = 45 * 3.14159265359/180 # FOV of MToF in deg (converrted to rad) VL53L5CX/VL53L8CX = 45 deg
NEED_ROTATE = 1 # 0 = No rotation, 1-3 integer for number of times to rotate CW
# Target is to aim for the depth image to arrange from top-left to bottom-right, same as image pixel arrangement
# ::::::::::::::::::::: ROS-related Params ::::::::::::::::::::::
IMAGE_COMPRESSED = True
IMAGE_TOPIC = "/camera/compressed"
MTOF_TOPIC = "/mtof/data"
# ::::::::::::::::::::: Visualization :::::::::::::::::::::
# Depth TURBO color-map parameters
VIS_MAX_RANGE = 1.0 # Range in meters which will be in color blue
VIS_MIN_RANGE = 0.1 # Range in meters which will be in Color red
# ::::::::::::::::::::: Calib Params ::::::::::::
NUM_DATA_POINTS = 200 # How many data should be collected for the calibration (recommeded more than 100 with sufficient movement)
BOARD_OR_HOLE = True # True = use board calib mode, False = use hole calib mode
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
    z_array = []
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
        user_input = input("Type something and press enter when ready: ") # Type something, doesn't matter. It will go even if you say no.
        if(user_input == "something"):
            print("Haha. Very funny.")

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
                    self.image_center_array.append(image_vec)
                    self.depth_center_array.append(mtof_vec)
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
        if(BOARD_OR_HOLE): # Use board mode
            if(centroids[0] < centroids[1]):
                depth_segmented = depth_segmented<1
        else: # Use hole mode
            if(centroids[0] > centroids[1]):
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
        # Get timestamp
        depth_timestamp = np.array(self.depth_ts_array)
        image_timestamp = np.array(self.image_ts_array)

        # Calculate common time space and interpolate
        common_time = np.linspace(max(depth_timestamp[0], image_timestamp[0]), min(depth_timestamp[-1], image_timestamp[-1]), NUM_DATA_POINTS*2)
        depth_x_inter = interp1d(depth_timestamp, np.array(self.depth_center_array)[:,0], kind='linear', fill_value='extrapolate')(common_time) 
        img_x_inter = interp1d(image_timestamp, np.array(self.image_center_array)[:,0], kind='linear', fill_value='extrapolate')(common_time) 
        depth_y_inter = interp1d(depth_timestamp, np.array(self.depth_center_array)[:,1], kind='linear', fill_value='extrapolate')(common_time) 
        img_y_inter = interp1d(image_timestamp, np.array(self.image_center_array)[:,1], kind='linear', fill_value='extrapolate')(common_time) 
        depth_z_inter = interp1d(depth_timestamp, np.array(self.depth_center_array)[:,2], kind='linear', fill_value='extrapolate')(common_time) 
        img_z_inter = interp1d(image_timestamp, np.array(self.image_center_array)[:,2], kind='linear', fill_value='extrapolate')(common_time) 

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
        print(RotR.as_euler("xyz", degrees=True))
        
        # Plot raw data
        fig, axs_raw = plt.subplots(2, 1)
        fig.canvas.set_window_title('Uncalibrated Image Point')
        axs_raw[0].set_title("Uncalibrated azimuth")
        axs_raw[0].plot(depth_vec_list[:,0], color=(1.0, 0.0, 0.0), label='MToF [rad]')
        axs_raw[0].plot(image_vec_list[:,0], color=(0.0, 1.0, 0.0), label='Image [rad]')
        axs_raw[0].legend(loc='best')
        axs_raw[0].grid()
        axs_raw[1].set_title("Uncalibrated elevation")
        axs_raw[1].plot(depth_vec_list[:,1], color=(1.0, 0.0, 0.0), label='MToF [rad]')
        axs_raw[1].plot(image_vec_list[:,1], color=(0.0, 1.0, 0.0), label='Image [rad]')
        axs_raw[1].legend(loc='best')
        axs_raw[1].grid()

        # Plot corrected Data
        depth_vec_res = RotR.apply(depth_vec_list)
        fig, axs_res = plt.subplots(2, 1)
        fig.canvas.set_window_title('Calibrated Image Point')
        axs_res[0].set_title("Calibrated azimuth")
        axs_res[0].plot(depth_vec_res[:,0], color=(1.0, 0.0, 0.0), label='MToF [rad]')
        axs_res[0].plot(image_vec_list[:,0], color=(0.0, 1.0, 0.0), label='Image [rad]')
        axs_res[0].legend(loc='best')
        axs_res[0].grid()
        axs_res[1].set_title("Calibrated elevation")
        axs_res[1].plot(depth_vec_res[:,1], color=(1.0, 0.0, 0.0), label='MToF [rad]')
        axs_res[1].plot(image_vec_list[:,1], color=(0.0, 1.0, 0.0), label='Image [rad]')
        axs_res[1].legend(loc='best')
        axs_res[1].grid()

        self.result_rvec = RotR.as_euler("xyz", degrees=False)
        self.calibration_done = True

        print("----------- Calibration result -----------------")
        print("  Roll offset: %.3f degrees"%(RotR.as_euler("xyz", degrees=True)[2]))
        print("  Pitch offset: %.3f degrees"%(RotR.as_euler("xyz", degrees=True)[0]))
        print("  Yaw offset: %.3f degrees"%(RotR.as_euler("xyz", degrees=True)[1]))
        print("  XYZ: %.3f %.3f %.3f degrees"%(RotR.as_euler("xyz", degrees=True)[0], RotR.as_euler("xyz", degrees=True)[1], RotR.as_euler("xyz", degrees=True)[2]))
        print("Calibration completed! Otsukaresama desu!")
        print()
        print("Close the plots to continue...")
        plt.show()
        print("You can now view calibrated result image in /mtofal/debug/aruco/compressed")
        

def main(args=None):
    rclpy.init(args=args)
    mtof_caliber = Mtofal()
    rclpy.spin(mtof_caliber)
    mtof_caliber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

