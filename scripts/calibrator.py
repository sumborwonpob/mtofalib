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
IMAGE_HEIGHT = 800
# New image size
NEW_WIDTH = 1280 # In case you want to resize the image
NEW_HEIGHT = 800 # In case you want to resize the image
# Camera intrinsics (of the original image size)
# Camera matrix (does not support skew by default. If you need it, apply yourself)
Kmat = np.array([[497.17756530532347, 0.0, 630.0687626834599], [0.0, 497.83938338803733, 386.46553181529407], [0.0, 0.0, 1.0]])
# Distortion matrix: Pinhole [k1 k2 p1 p2 k3], Fisheye [k1 k2 k3 k4]
Dmodel = "fisheye" # pinhole or fisheye
Dmat = np.array([[-0.06090879252215075], [0.017015566724699678], [-0.007572310675681196], [-0.0005494543454422542]])
# ::::::::::::::::::::: MToF Params :::::::::::::::::::::
T_C_MTOF = np.array([-0.025, 0.0, 0.0]) # MToF's position in the camera's coordinate frame in meters (X right, Y down, Z front)
VIS_DEPTH_IMAGE_SIZE = 400 # How big debug depth image in pixels (for debugging and for centroid calculation)
PHI_WH = 0.785398 # FOV of MToF in rad 45 deg = 0.785398 rad
NEED_FLIP_HORIZONTAL = False # Horizontal flip is performed before rotation
NEED_FLIP_VERTICAL = True # Vertical flip is performed before rotation
NEED_ROTATE = 2 # 0 = No rotation, 1-3 integer for number of times to rotate CCW
# ::::::::::::::::::::: ROS-related Params ::::::::::::::::::::::
IMAGE_COMPRESSED = True
IMAGE_TOPIC = "/camera/compressed"
MTOF_TOPIC = "/mtof/data"
# ::::::::::::::::::::: Visualization :::::::::::::::::::::
# Depth TURBO color-map parameters
VIS_MAX_RANGE = 0.7 # Range in meters which will be in color blue
VIS_MIN_RANGE = 0.1 # Range in meters which will be in Color red
# ::::::::::::::::::::: Calib Params ::::::::::::
NUM_DATA_POINTS = 300 # How many data should be collected for the calibration (recommeded more than 100 with sufficient movement)
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
    obj_point_array = []
    image_roll_array = []
    depth_roll_array = []
    hole_time = None
    hole_center = None
    board_center = None
    void_center = None
    roll_offset = 0.0
    pitch_offset = 0.0
    yaw_offset = 0.0
    undist_imgPoints = None

    # :::::::::::::::::::::: Calib result :::::::::::::::::::::::::::
    result_rvec = None

    # :::::::::::::::::::::: Camera Params :::::::::::::::::::::::::::::::
    Kmat_new = None

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
        Kmat[0][0] = Kmat[0][0] * NEW_WIDTH / IMAGE_WIDTH
        Kmat[0][2] = Kmat[0][2] * NEW_WIDTH / IMAGE_WIDTH
        Kmat[1][1] = Kmat[1][1] * NEW_HEIGHT / IMAGE_HEIGHT
        Kmat[1][2] = Kmat[1][2] * NEW_HEIGHT / IMAGE_HEIGHT
        self.Kmat_new = Kmat

        #-------------- Calculate display font scale -----------
        self.font_scale = NEW_WIDTH/640

        print("")
        # ---------- Dewa, hajime mashou ka? ----------
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("::::::::::::::::::::::::: Mtofalib :::::::::::::::::::::::::")
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        # input("Ready?: ") # Type something, doesn't matter. It will go even if you say no.

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
        frame_debug = cv2.fisheye.undistortImage(frame_debug, self.Kmat_new, Dmat, Knew=self.Kmat_new)


        # --------------------------- Calibration Phrase ---------------------------------------------
        if(not self.calibration_done):
            # Get the image points and append object points
            objPoints_now = []
            imgPoints_now = []
            if(ids is not None and self.getting_data):
                for i in range(len(ids)):
                    # if(ids[i] == 11 or ids[i] == 13):
                    #     marker_type = cv2.MARKER_TRIANGLE_UP
                    # if(ids[i] == 12 or ids[i] == 14):
                    #      marker_type = cv2.MARKER_TRIANGLE_DOWN
                    #cv2.drawMarker(frame_debug, (int(corners[i][0][0][0]), int(corners[i][0][0][1])), (0, 0, 255), marker_type, 20)
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
                imgePointsProjected, jacobian_ = cv2.fisheye.projectPoints(np.array(object_points).reshape(len(object_points), 1, 3), rvec, tvec, self.Kmat_new, Dmat)
                undistortedNormVec = cv2.fisheye.undistortPoints(imgePointsProjected, self.Kmat_new, Dmat)
                imgePointsProjected = cv2.fisheye.undistortPoints(imgePointsProjected, self.Kmat_new, Dmat, None, self.Kmat_new)

                # Draw debug
                for i in range(imgePointsProjected.shape[0]):
                    if(i == 0 or i == 4):
                        marker_type = cv2.MARKER_TRIANGLE_UP
                        cv2.drawMarker(frame_debug, (int(imgePointsProjected[i][0][0]), int(imgePointsProjected[i][0][1])), (0, 0, 255), marker_type, int(20*self.font_scale))
                    if(i == 8 or i == 12):
                        marker_type = cv2.MARKER_TRIANGLE_DOWN
                        cv2.drawMarker(frame_debug, (int(imgePointsProjected[i][0][0]), int(imgePointsProjected[i][0][1])), (0, 0, 255), marker_type, int(20*self.font_scale))
                    cv2.drawMarker(frame_debug, (int(imgePointsProjected[i][0][0]), int(imgePointsProjected[i][0][1])), (0, 255, 0), cv2.MARKER_SQUARE, int(5*self.font_scale))
                cv2.line(frame_debug, (int(Kmat[0][2]), int(Kmat[1][2])), ((int(imgePointsProjected[16][0][0]), int(imgePointsProjected[16][0][1]))), (255, 0, 255), int(2*self.font_scale))
                cv2.putText(frame_debug, "%.2fm"%(tvec[2]), (int(NEW_WIDTH/2)-50,int(NEW_HEIGHT/2)-10),  cv2. FONT_HERSHEY_PLAIN, 2*self.font_scale, (0, 255, 0), int(3*self.font_scale), cv2.LINE_AA)
                

                # Save data
                if(self.hole_center is not None and not (math.isnan(self.hole_center[0]) or math.isnan(self.hole_center[1])) and self.new_tof_data):
                    # Project center point to the image using initially provided position
                    # Calculate 3D position from MToF
                    mtof_aziele = R.from_euler("xyz",[-self.hole_center[1], self.hole_center[0], 0.0], degrees=False)
                    #hole_point3d = mtof_aziele.apply(np.array(tvec[:,0]))
                    hole_point3d = mtof_aziele.apply(np.array([0.0, 0.0, tvec[2,0]]))
                    # Project that 3D point into image frame with provided relative position and zero rotation
                    hole_imgpoint, hole_jacobian = cv2.fisheye.projectPoints(np.array([hole_point3d]).reshape(1,1,3), np.array([0.0, 0.0, 0.0]), T_C_MTOF, self.Kmat_new, Dmat)
                    # Convert to normalized 2D vector
                    hole_imgpoint_undist = cv2.fisheye.undistortPoints(np.array(hole_imgpoint).reshape(1,1,2), self.Kmat_new, Dmat)[0][0]

                    # Convert 2D vector to 3D vector by adding constant z=1 and nromalize the vector
                    image_vec = np.array([undistortedNormVec[16][0][0], undistortedNormVec[16][0][1], 1])
                    mtof_vec = np.array([hole_imgpoint_undist[0], hole_imgpoint_undist[1], 1])
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
            for distances in self.depth_data:
                for distance in distances:
                    # Start from top-left
                    azimuth = -((col * angle_per_zone) - (PHI_WH/2) + (angle_per_zone/2))
                    elevation = -((row * angle_per_zone) - (PHI_WH/2) + (angle_per_zone/2))
                    zone_angle = R.from_euler("xy", [elevation, azimuth], degrees=False)
                    zone_point3d = zone_angle.apply(np.array([0.0, 0.0, distance]))
                    point3d_list.append(zone_point3d)
                    col+= 1
                row += 1
                col = 0
            # Project those points to 2D frame using provided translation and calculated rotation
            zone_imgpoint, hole_jacobian = cv2.fisheye.projectPoints(np.array([point3d_list]).reshape(1,64,3), self.result_rvec, -1*T_C_MTOF, self.Kmat_new, Dmat)
            zone_imgpoint_undist = cv2.fisheye.undistortPoints(np.array(zone_imgpoint).reshape(64,1,2), self.Kmat_new, Dmat, None, self.Kmat_new)
            #print(zone_imgpoint_undist)
            # Draw them
            for i in range(len(zone_imgpoint_undist)):
                imgpt = zone_imgpoint_undist[i][0]
                dist = self.depth_data[int(i/self.zone_res)][i%self.zone_res]
                color = self.calculateColorMap(dist)
                cv2.circle(frame_debug, (int(imgpt[0]), int(imgpt[1])), int(20*self.font_scale), color, int(2*self.font_scale))

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

        # Flip and rotate
        if(NEED_FLIP_HORIZONTAL):
            depth_data = np.flip(depth_data, axis=1)
        if(NEED_FLIP_VERTICAL):
            depth_data = np.flip(depth_data, axis=0)

        for i in range(NEED_ROTATE):
            depth_data = np.rot90(depth_data)

        # Store 
        self.depth_data = depth_data

        # Calculate center of mass
        # Clustering between the board and the hole
        centroids, _ = kmeans(depth_data.flatten(), 2)
        segmented, _ = vq(depth_data.flatten(), centroids)
        depth_segmented = np.reshape(segmented, (self.zone_res,self.zone_res))
        # Rearrage the index
        if(centroids[0] > centroids[1]):
            depth_segmented = depth_segmented<1
        # Find the hole center
        self.hole_time = msg.header.stamp.sec+(msg.header.stamp.nanosec*1e-9)
        self.hole_center = (np.array(center_of_mass(depth_segmented))-(self.zone_res/2))*(PHI_WH/self.zone_res) # Result is in radians

        # Draw depth image
        for i in range(self.zone_res):
            for j in range(self.zone_res):
                b,g,r = self.calculateColorMap(depth_data[i][j])
                depth_img[int(j*depth_img_pixels):int((j*depth_img_pixels)+depth_img_pixels), int(i*depth_img_pixels):int((i*depth_img_pixels)+depth_img_pixels)] = (b, g, r)
        
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

        # Calculate Delay
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
        fig, axs_raw = plt.subplots(3, 1)
        axs_raw[0].plot(depth_vec_list[:,0], color=(1.0, 0.0, 0.0), label='MToF X')
        axs_raw[0].plot(image_vec_list[:,0], color=(0.0, 1.0, 0.0), label='Image X')
        axs_raw[0].legend(loc='best')
        axs_raw[0].grid()
        axs_raw[1].plot(depth_vec_list[:,1], color=(1.0, 0.0, 0.0), label='MToF Y')
        axs_raw[1].plot(image_vec_list[:,1], color=(0.0, 1.0, 0.0), label='Image Y')
        axs_raw[1].legend(loc='best')
        axs_raw[1].grid()
        axs_raw[2].plot(depth_vec_list[:,2], color=(1.0, 0.0, 0.0), label='MToF Z')
        axs_raw[2].plot(image_vec_list[:,2], color=(0.0, 1.0, 0.0), label='Image Z')
        axs_raw[2].legend(loc='best')
        axs_raw[2].grid()

        # Plot corrected Data
        depth_vec_res = RotR.apply(depth_vec_list)
        fig, axs_res = plt.subplots(3, 1)
        axs_res[0].plot(depth_vec_res[:,0], color=(1.0, 0.0, 0.0), label='MToF X')
        axs_res[0].plot(image_vec_list[:,0], color=(0.0, 1.0, 0.0), label='Image x')
        axs_res[0].legend(loc='best')
        axs_res[0].grid()
        axs_res[1].plot(depth_vec_res[:,1], color=(1.0, 0.0, 0.0), label='MToF Y')
        axs_res[1].plot(image_vec_list[:,1], color=(0.0, 1.0, 0.0), label='Image Y')
        axs_res[1].legend(loc='best')
        axs_res[1].grid()
        axs_res[2].plot(depth_vec_res[:,2], color=(1.0, 0.0, 0.0), label='MToF Z')
        axs_res[2].plot(image_vec_list[:,2], color=(0.0, 1.0, 0.0), label='Image Z')
        axs_res[2].legend(loc='best')
        axs_res[2].grid()

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
        #plt.show()
        print("You can now view calibrated result image in /mtofal/debug/aruco/compressed")
        

def main(args=None):
    rclpy.init(args=args)
    mtof_caliber = Mtofal()
    rclpy.spin(mtof_caliber)
    mtof_caliber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

