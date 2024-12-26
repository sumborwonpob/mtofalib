#!/usr/bin/env python3

import cv2
import time

import rclpy
from rclpy.node import Node
from rclpy import qos

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
bridge = CvBridge()

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"GREY"))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FPS, 50)
        cap.set(cv2.CAP_PROP_EXPOSURE, 100)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)

        self.image_publisher_ = self.create_publisher(Image, '/camera/image_raw', qos.qos_profile_sensor_data)
        self.publisher_ = self.create_publisher(CompressedImage, '/camera/compressed', qos.qos_profile_sensor_data)
        
        print("WIDTH: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("HEIGHT: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("FPS: ", cap.get(cv2.CAP_PROP_FPS))
        print("EXP: ", cap.get(cv2.CAP_PROP_EXPOSURE))
        print("BRIGHT: ", cap.get(cv2.CAP_PROP_BRIGHTNESS))
        if(not cap.isOpened()):
            quit()
        try:
            while True:
                 ret, frame = cap.read()
                 if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    image_message = bridge.cv2_to_compressed_imgmsg(frame)
                    image_message.header.stamp = self.get_clock().now().to_msg()
                    image_message.header.frame_id = "camera_link"
                    self.publisher_.publish(image_message)
        except KeyboardInterrupt:
        	quit()

def main(args=None):
    rclpy.init(args=args)

    webcam_publisher = WebcamPublisher()

    rclpy.spin(webcam_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    webcam_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()