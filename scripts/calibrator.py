#!/usr/bin/env python3

# I made this.

import cv2

import rclpy
from rclpy.node import Node
from rclpy import qos

import math
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.cluster.vq import kmeans, vq
from scipy import signal
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from mtof_camera_calibrator.msg import Mtof
bridge = CvBridge()

# ::::::::::::::::: Put your parameters here :::::::::::::::::::::::::
# ::::::::::::::::::::: Camera Params :::::::::::::::::::::
# Original image size
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 800
# New image size
NEW_WIDTH = 640 # In case you want to resize the image
NEW_HEIGHT = 400 # In case you want to resize the image
# Camera intrinsics (of the original image size)
# Camera matrix (does not support skew by default. If you need it, apply yourself)
Kmat = np.array([
    [564.414786, 0.0, 664.842070],
    [0.0, 565.798772, 370.252387],
    [0.0, 0.0, 1.0]
])
# Distortion matrix: Pinhole, [k1 k2 p1 p2 k3], fisheye is not supported. Implement the functions to cv2.fisheye yourself if you need it.
Dmat = np.array([
    -0.241306, 0.033984, 0.004612, -0.000143, 0.0
])
# ::::::::::::::::::::: MToF Params :::::::::::::::::::::
T_C_MTOF = np.array([0.0, -0.015, 0.0]) # MToF's position in the camera's coordinate frame in meters (X right, Y down, Z front)
VIS_DEPTH_IMAGE_SIZE = 400 # How big debug depth image in pixels (for debugging and for centroid calculation)
PHI_WH = 0.785398 # FOV of MToF in rad 45 deg = 0.785398 rad
NEED_FLIP_HORIZONTAL = False # Horizontal flip is performed before rotation
NEED_FLIP_VERTICAL = True # Vertical flip is performed before rotation
NEED_ROTATE = 0 # 0 = No rotation, 1-3 integer for number of times to rotate CCW
# ::::::::::::::::::::: ROS-related Params :::::::::::::::::::::
IMAGE_COMPRESSED = True
IMAGE_TOPIC = "/camera/compressed"
MTOF_TOPIC = "/mtof/data"
# ::::::::::::::::::::: Visualization :::::::::::::::::::::
# Depth TURBO color-map parameters
VIS_MAX_RANGE = 0.5 # Range in meters which will be in color blue
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

    is_doing_roll_calib = False
    is_doing_pitchyaw_calib = False

    # :::::::::::::::::::::: Calculation variables ::::::::::::::::::
    depth_center_array = []
    image_center_array = []
    z_array = []
    obj_point_array = []
    image_roll_array = []
    depth_roll_array = []
    hole_center = None
    board_center = None
    void_center = None
    roll_offset = 0.0
    pitch_offset = 0.0
    yaw_offset = 0.0

    # :::::::::::::::::::::: Camera Params :::::::::::::::::::::::::::::::
    Kmat_new = None
    cam_phi_width = 0.0
    cam_phi_height = 0.0

    # ::::::::::::::::::::::::::::::::::::::::::: CONSTRUCTOR :::::::::::::::::::::::::::::::::::::::::
    def __init__(self):
        super().__init__('mtofal')
        
        print("Initializing program....")

        # --------- Publishers ------------
        # Debug images publishers
        self.debug_pubber = self.create_publisher(CompressedImage, '/mtofal/debug/aruco/compressed', qos.qos_profile_sensor_data)
        self.depth_pubber = self.create_publisher(CompressedImage, '/mtofal/debug/depth/compressed', qos.qos_profile_sensor_data)

        # ------------ Subscribers --------
        if(IMAGE_COMPRESSED):
            self.image_subber = self.create_subscription(CompressedImage, IMAGE_TOPIC, self.image_callback, 10)
        else:
            self.image_subber = self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)

        self.tof_subber = self.create_subscription(Mtof, MTOF_TOPIC, self.tof_callback, 10)

        # Initialize Aruco dict
        self.dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT)

        # ------------ Calculate new Kmat -------------------
        Kmat[0][0] = Kmat[0][0] * NEW_WIDTH / IMAGE_WIDTH
        Kmat[0][2] = Kmat[0][2] * NEW_WIDTH / IMAGE_WIDTH
        Kmat[1][1] = Kmat[1][1] * NEW_HEIGHT / IMAGE_HEIGHT
        Kmat[1][2] = Kmat[1][2] * NEW_HEIGHT / IMAGE_HEIGHT
        self.Kmat_new = Kmat
        # ------------ Calculate camera FOV -----------------
        # Compute the undistorted corners of the image
        w = NEW_WIDTH
        h = NEW_HEIGHT
        corners = np.array([
            [0, 0],               # Top-left
            [w - 1, 0],           # Top-right
            [w - 1, h - 1],       # Bottom-right
            [0, h - 1]            # Bottom-left
        ], dtype=np.float32).reshape(-1, 1, 2)
        # Undistort points to find their actual angular positions in the image
        undistorted_corners = cv2.undistortPoints(corners, self.Kmat_new, Dmat, P=self.Kmat_new)
        # Calculate the horizontal and vertical field of view
        self.cam_phi_width = np.degrees(2 * np.arctan2(undistorted_corners[1][0][0] - undistorted_corners[0][0][0], 2 * self.Kmat_new,[0, 0]))
        self.cam_phi_height = np.degrees(2 * np.arctan2(undistorted_corners[2][0][1] - undistorted_corners[0][0][1], 2 * self.Kmat_new,[1, 1]))
        print("")
        # ---------- Dewa, hajime mashou ka? ----------
        print("::::::::::: STEP 1 Roll Calibration ::::::::::::")
        input("Ready?: ") # Type something, doesn't matter. It will go even if you say no.

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
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, self.dictionary, Kmat, Dmat)
        frame_debug = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Get the image points and append object points
        objPoints_now = []
        imgPoints_now = []
        if(ids is not None):
            for i in range(len(ids)):
                if(ids[i] == 11 or ids[i] == 13):
                    marker_type = cv2.MARKER_TRIANGLE_UP
                if(ids[i] == 12 or ids[i] == 14):
                     marker_type = cv2.MARKER_TRIANGLE_DOWN
                cv2.drawMarker(frame_debug, (int(corners[i][0][0][0]), int(corners[i][0][0][1])), (0, 0, 255), marker_type, 20)
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
                    imgPoints_now.append(((corners[i][0][j][0]), int(corners[i][0][j][1])))

            # Solve for pose of the camera w.r.t. to board origin (center 0,0 in your object points)
            retval, rvec, tvec, _ = cv2.solvePnPRansac(np.array(objPoints_now), np.array(imgPoints_now), Kmat, Dmat)
            frame_debug = cv2.undistort(frame_debug, Kmat, Dmat)
            # Draw in the frame
            cv2.drawFrameAxes(frame_debug, Kmat, Dmat, rvec, tvec, 0.06)

            # Project the rest of the board that is not inside the image
            imgePointsProjected, jacobian_ = cv2.projectPoints(np.array(object_points), rvec, tvec, Kmat, Dmat)
            imgePointsProjected = cv2.undistortPoints(imgePointsProjected, Kmat, Dmat, None, Kmat)

            # Draw debug
            for i in range(imgePointsProjected.shape[0]):
                cv2.drawMarker(frame_debug, (int(imgePointsProjected[i][0][0]), int(imgePointsProjected[i][0][1])), (0, 255, 0), cv2.MARKER_SQUARE, 5)
            cv2.line(frame_debug, (int(Kmat[0][2]), int(Kmat[1][2])), ((int(imgePointsProjected[16][0][0]), int(imgePointsProjected[16][0][1]))), (255, 0, 255), 2)


            # :::::::::::::::::::::::::::::::: Roll calibration ::::::::::::::::::::::::::::::::::::::
            if(not self.is_doing_roll_calib):
                if(self.depth_data is not None):
                    # Get the latest depth data
                    depth_data = self.depth_data
                    # Clustering between the board and the void
                    centroids, _ = kmeans(depth_data.flatten(), 2)
                    segmented, _ = vq(depth_data.flatten(), centroids)
                    depth_segmented = np.reshape(segmented, (self.zone_res,self.zone_res))
                    # rearrage such that index 0 is always board
                    if(centroids[0] > centroids[1]):
                        depth_segmented = depth_segmented<1
                    board_segmented = depth_segmented
                    void_segmented = depth_segmented<1

                    # Find center of both
                    self.board_center = center_of_mass(board_segmented)
                    self.void_center = center_of_mass(void_segmented)

                    # initialize two vectors
                    zero_roll_vec = [[0.0, 1.0, 0.0]] # One vector pointing down when there's no roll
                    measured_vec = [[
                        self.void_center[0] - self.board_center[0],
                        self.void_center[1] - self.board_center[1],
                        0.0
                    ]] # Another vector is the measured vector from CoM of the void to the board
                    # Now find the rotation that would rotate from one to another, that is our estimated board roll angle
                    rotation, _ = Rotation.align_vectors(measured_vec, zero_roll_vec)
                    rot_eul = rotation.as_euler("xyz", degrees=False)
                    print("Roll %d/%d, Depth:%.2f Image:%.2f"%(len(self.depth_roll_array), NUM_DATA_POINTS, rot_eul[2]*180/3.14, rvec[2]*180/3.14))
                    
                    # Append the data
                    if(self.new_tof_data and self.getting_data):
                        self.new_tof_data = False
                        self.depth_roll_array.append(rot_eul[2])
                        self.image_roll_array.append(rvec[2][0])

                    # If the data reaches the specified amount, do calibration
                    if(len(self.depth_roll_array) >= NUM_DATA_POINTS):
                            self.getting_data = False
                            self.nowDoRollCalculation()
                            print("Roll Calib complete!")
                            print("::::::::::: STEP 2 Pitch Yaw Calibration ::::::::::::")
                            input("Continue?")
                            self.is_doing_roll_calib = True
                            self.getting_data = True

                    
            # :::::::::::::::::::::::::::::::: Pitch/Yaw calibration ::::::::::::::::::::::::::::::::::::::
            if(self.is_doing_roll_calib and not self.is_doing_pitchyaw_calib):
                if(self.depth_data is not None):
                    # Get the latest depth data
                    depth_data = self.depth_data
                    # Clustering between the board and the hole
                    centroids, _ = kmeans(depth_data.flatten(), 2)
                    segmented, _ = vq(depth_data.flatten(), centroids)
                    depth_segmented = np.reshape(segmented, (self.zone_res,self.zone_res))
                    # Rearrage the index
                    if(centroids[0] > centroids[1]):
                        depth_segmented = depth_segmented<1
                    # Find the hole center
                    self.hole_center = center_of_mass(depth_segmented)
                    # Append the data
                    if(self.new_tof_data and self.getting_data):
                        self.new_tof_data = False
                        self.depth_center_array.append((self.hole_center[0]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res), self.hole_center[1]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res)))
                        self.image_center_array.append((imgePointsProjected[16][0][0], imgePointsProjected[16][0][1]))
                        self.z_array.append(tvec[2][0])
                        self.obj_point_array.append([0.0, 0.0, 0.0])
                        print("Data collected:", len(self.depth_center_array), "/", NUM_DATA_POINTS)
                        # If there's enough data, do the calculation
                        if(len(self.depth_center_array) >= NUM_DATA_POINTS):
                            self.getting_data = False
                            self.nowDoPitchYawCalculation()
                            print("Calib complete!")
                    
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

        # Draw depth image
        for i in range(self.zone_res):
            for j in range(self.zone_res):
                b,g,r = self.calculateColorMap(depth_data[i][j])
                depth_img[int(j*depth_img_pixels):int((j*depth_img_pixels)+depth_img_pixels), int(i*depth_img_pixels):int((i*depth_img_pixels)+depth_img_pixels)] = (b, g, r)
            
        # Draw debug lines during roll calib
        if(not self.is_doing_roll_calib):
            if(self.board_center is not None and self.void_center is not None):
                board_center = self.board_center
                void_center = self.void_center
                cv2.drawMarker(depth_img, (int(board_center[0]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res)), int(board_center[1]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res))), (255, 0, 255), cv2.MARKER_CROSS, 20, 5)
                cv2.drawMarker(depth_img, (int(void_center[0]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res)), int(void_center[1]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res))), (255, 0, 255), cv2.MARKER_CROSS, 20, 5)
                self.new_tof_data = True
                cv2.line(depth_img, (int(board_center[0]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res)), int(board_center[1]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res))), (int(void_center[0]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res)), int(void_center[1]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res))), (255, 0, 255), 2)
        # Draw debug lines during pitch/yaw calib
        if(self.is_doing_roll_calib and not self.is_doing_pitchyaw_calib):
            if(self.hole_center is not None and not (math.isnan(self.hole_center[0]) or math.isnan(self.hole_center[1]))):
                actual_center = self.hole_center
                cv2.drawMarker(depth_img, (int(actual_center[0]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res)), int(actual_center[1]*(VIS_DEPTH_IMAGE_SIZE/self.zone_res))), (255, 0, 255), cv2.MARKER_CROSS, 20, 5)
                self.new_tof_data = True
                cv2.line(depth_img, (int(VIS_DEPTH_IMAGE_SIZE/2), int(VIS_DEPTH_IMAGE_SIZE/2)), (int(actual_center[0]*VIS_DEPTH_IMAGE_SIZE/self.zone_res), int(actual_center[1]*VIS_DEPTH_IMAGE_SIZE/self.zone_res)), (255, 0, 255), 2)

        #  Make image and publish
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

    # ------------------ Function for roll calibration -----------------------
    def nowDoRollCalculation(self):
        print("Calibrating roll...")
        # Filter the data and match the phase
        b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.
        depth_roll_filt = signal.filtfilt(b, a, np.array(self.depth_roll_array), method="gust")
        img_roll_filt = signal.filtfilt(b, a, np.array(self.image_roll_array), method="gust")

        # Find the median
        depth_roll_median = np.median(depth_roll_filt)
        img_roll_median = np.median(img_roll_filt)
        
        print("  Roll offset = %.3f degrees"%((img_roll_median - depth_roll_median)*180/3.14))
        print("Close the plot window when you are ready to continue...")
        fig, axs = plt.subplots(1, 1) 
        axs.plot(depth_roll_filt, color=(1.0, 0.0, 0.0), label='MToF roll')
        axs.plot(img_roll_filt, color=(0.0, 1.0, 0.0), label='Image roll')
        axs.axhline(depth_roll_median, color=(1.0, 0.0, 0.0), linestyle='--', label='MToF roll median')
        axs.axhline(img_roll_median, color=(0.0, 1.0, 0.0), linestyle='--', label='Image roll median')
        axs.legend(loc='best')
        axs.grid()
        plt.show()
        self.roll_offset = (img_roll_median - depth_roll_median)*180/3.14

    # ------------------ Function for pitch calibration -----------------------
    def nowDoPitchYawCalculation(self):
        print("Calibrating pitch yaw...")

        # Construct intrinsics for MToF
        cxy = VIS_DEPTH_IMAGE_SIZE/2

        # Apply filter to the data
        b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.
        depth_x_filt = signal.filtfilt(b, a, np.array(self.depth_center_array)[:,0], method="gust")
        img_x_filt = signal.filtfilt(b, a, np.array(self.image_center_array)[:,0], method="gust")
        depth_y_filt = signal.filtfilt(b, a, np.array(self.depth_center_array)[:,1], method="gust")
        img_y_filt = signal.filtfilt(b, a, np.array(self.image_center_array)[:,1], method="gust")
        z_filt = signal.filtfilt(b, a, np.array(self.z_array), method="gust")

        depth_x_filt = depth_x_filt - cxy
        depth_y_filt = depth_y_filt - cxy
        img_x_filt = img_x_filt - Kmat[0][2]
        img_y_filt = img_y_filt - Kmat[1][2]

        #Calculate pixel offset induced by relative translation
        theta_x = np.arctan(T_C_MTOF[0]/z_filt)
        offset_x = (theta_x*NEW_WIDTH)/self.cam_phi_width
        theta_y = np.arctan(T_C_MTOF[1]/z_filt)
        offset_y = (theta_y*NEW_HEIGHT)/self.cam_phi_height

        # Check median from mtof
        depth_x_median = np.median(depth_x_filt)
        depth_y_median = np.median(depth_y_filt)

        # Find image median
        img_x_median = np.median(img_x_filt)
        img_y_median = np.median(img_y_filt)

        # Now make the data by matching the median offset
        depth_x_filt_centered = (depth_x_filt - depth_x_median) * PHI_WH / (VIS_DEPTH_IMAGE_SIZE)
        depth_y_filt_centered = (depth_y_filt - depth_y_median) * PHI_WH / (VIS_DEPTH_IMAGE_SIZE)
        img_x_filt_centered = (img_x_filt - img_x_median) * self.cam_phi_width / NEW_WIDTH
        img_y_filt_centered = (img_y_filt - img_y_median) * self.cam_phi_height / NEW_HEIGHT

        correlation = signal.correlate(img_x_filt_centered, depth_x_filt_centered, mode="full")
        lag = signal.correlation_lags(len(img_x_filt_centered), len(depth_x_filt_centered), mode="full")
        print(lag)

        self.yaw_offset = (img_x_median-depth_x_median)*(self.cam_phi_width / NEW_WIDTH)*180/3.14
        self.pitch_offset = (img_y_median-depth_y_median)*(self.cam_phi_height / NEW_HEIGHT)*180/3.14

        print("  Yaw offset: %.3f degrees"%(self.yaw_offset))
        print("  Pitch offset: %.3f degrees"%(self.pitch_offset))
        print("Close the plot window when you are ready to continue...")

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(depth_x_filt, color=(1.0, 0.0, 0.0), label='MToF X')
        axs[0].plot(img_x_filt, color=(0.0, 1.0, 0.0), label='Image x')
        axs[0].axhline(depth_x_median, color=(1.0, 0.0, 0.0), linestyle='--', label='MToF X median')
        axs[0].axhline(img_x_median, color=(0.0, 1.0, 0.0), linestyle='--', label='Image x median')
        axs[0].legend(loc='best')
        axs[0].grid()
        axs[1].plot(depth_y_filt, color=(1.0, 0.0, 0.0), label='MToF Y')
        axs[1].plot(img_y_filt, color=(0.0, 1.0, 0.0), label='Image Y')
        axs[1].axhline(depth_y_median, color=(1.0, 0.0, 0.0), linestyle='--', label='MToF Y median')
        axs[1].axhline(img_y_median, color=(0.0, 1.0, 0.0), linestyle='--', label='Image Y median')
        axs[1].legend(loc='best')
        axs[1].grid()
        fig, axs2 = plt.subplots(2, 1) 
        axs2[0].plot(depth_x_filt_centered, color=(1.0, 0.0, 0.0), label='MToF X centered')
        axs2[0].plot(img_x_filt_centered, color=(0.0, 1.0, 0.0), label='Image x centered')
        axs2[0].legend(loc='best')
        axs2[0].grid()
        axs2[1].plot(depth_y_filt_centered, color=(1.0, 0.0, 0.0), label='MToF Y centered')
        axs2[1].plot(img_y_filt_centered, color=(0.0, 1.0, 0.0), label='Image Y centered')
        axs2[1].legend(loc='best')
        axs2[1].grid()
        plt.show()

        print("----------- Calibration result -----------------")
        print("  Roll offset = %.3f degrees"%(self.roll_offset))
        print("  Pitch offset: %.3f degrees"%(self.pitch_offset))
        print("  Yaw offset: %.3f degrees"%(self.yaw_offset))
        print("Calibration completed! Otsukaresama desu!")
        

def main(args=None):
    rclpy.init(args=args)
    mtof_caliber = Mtofal()
    rclpy.spin(mtof_caliber)
    mtof_caliber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

