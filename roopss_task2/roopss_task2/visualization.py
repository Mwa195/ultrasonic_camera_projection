#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import cv2
import numpy as np
import pickle
import math

class VisualizerNode(Node):
    """
    Visualizer Node: Subscribes to /sensor_data to collect the readings
    and project the 3D points to 2D then display on the camera
    """
    def __init__(self):
        """
        Initialize visualization_node
        """
        super().__init__("visualization_node") # Initialize visualization node

        # Create a subscriber to /sensor_data
        self.create_subscription(
            Vector3,
            "/sensor_data",
            self.sensorDataCallback,
            10
        )

        # Load the camera calibration data
        with open('/home/mwa/ros2_ws/src/roopss_task2/roopss_task2/calibration_data.pkl', 'rb') as f:
            data = pickle.load(f)
        # Store the camera matrix and distortion coefficients
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['distortion_coefficients']
        self.get_logger().info(f"Camera Matrix:\n{self.camera_matrix}")
        self.get_logger().info(f"Distortion Coefficients:\n{self.dist_coeffs}")
        
        # Open Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open webcam")
            raise ValueError("Could not open webcam")
        
        # Rotation matrix and translation vector to project the sensor data to the camera
        # Equivalent of sensor's x, y & z readings to the camera's x, y & z
        self.R = np.array([
            [0,  1,  0],
            [0,  0, -1],
            [1,  0,  0],
        ], dtype=np.float32)

        # Distance from camera to sensor
        self.t = np.array([
            [0.0], # To the right of the camera
            [0.17], # Below the camera
            [0.28], # In front of the camera
        ], dtype=np.float32)
        
        # Update feed
        self.timer = self.create_timer(0.03, self.updateWebcamFeed)
        
        self.u = None
        self.v = None
        self.last_angle = None
        self.last_distance = None
        
        # Trail
        self.trail_points = []  # List to store previous (u, v) points
        self.max_trail_length = 45  # Max number of trail points
        
        self.get_logger().info("Projection Node Initialized ✅")

    def updateWebcamFeed(self):
        """
        Updates the camera feed by drawing a trail of points respective to
        sensor reading and display the current readings on top
        """
        # Capture frames
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture frame")
            return

        # Draw trail points with fading
        for idx, (trail_u, trail_v) in enumerate(self.trail_points):
            alpha = idx / self.max_trail_length  # 0 (oldest) to 1 (newest)
            color_intensity = int(255 * alpha)
            color = (0, 0, color_intensity)  # Dark red to bright red
            radius = 5
            cv2.circle(frame, (trail_u, trail_v), radius, color, -1)
        
        # Draw latest projected red point
        if self.u is not None and self.v is not None:
            height, width = frame.shape[:2]
            if 0 <= self.u < width and 0 <= self.v < height:
                cv2.circle(frame, (self.u, self.v), 10, (0, 0, 255), -1)  # Red dot
            else:
                self.get_logger().warn(f"Point outside image bounds: \
                                       u={self.u}, v={self.v}, image size={width}x{height}")
        
        # Write text
        if self.last_angle is not None and self.last_distance is not None:
            text = f"Rel Angle: {self.last_angle:.1f} deg, Distance: {self.last_distance:.1f} cm"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        
        # Visualize Output
        cv2.imshow('Webcam Feed', frame)
        cv2.waitKey(1)

    def sensorDataCallback(self, msg: Vector3):
        """
        Receives the sensor readings from /sensor_data and converts the points
        from 3D to 2D plane
        """
        # Store readings
        angle = msg.x  # Relative angle (-60 to 60 degrees)
        distance = msg.y / 100.0  # Convert cm to meters
        self.get_logger().info(f"Received: Angle={angle:.1f} deg, Distance={msg.y:.1f} cm")
        
        if not (-60 <= angle <= 60):
            self.get_logger().warn(f"Angle out of range: {angle}")
            return
        if not (0.02 <= distance <= 4.0):
            self.get_logger().warn(f"Distance out of range: {distance*100:.1f} cm")
            return
        
        # Convert from Polar to Cartesian
        theta = math.radians(angle)
        x = distance * math.cos(theta)
        y = distance * math.sin(theta)
        y = -y  # Reverse left-right
        z = 0.0
        self.get_logger().info(f"Sensor frame point: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        
        # Create the 3D array of the sensor
        point_sensor = np.array([[x], [y], [z]], dtype=np.float32)
        
        # Transform the 3D array to camera frame
        point_camera = self.R @ point_sensor + self.t
        self.get_logger().info(f"Camera frame point: x={point_camera[0,0]:.3f}, \
                               y={point_camera[1,0]:.3f}, z={point_camera[2,0]:.3f}")
        
        # Project to 2D image
        point_camera_reshaped = point_camera.reshape(1, 1, 3)
        
        point_camera_2d, _ = cv2.projectPoints(
            point_camera_reshaped,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            self.camera_matrix,
            self.dist_coeffs
        )
        point_2d = point_camera_2d[0, 0]
        self.get_logger().info(f"Projected 2D point: u={point_2d[0]:.1f}, v={point_2d[1]:.1f}")
        
        if not np.isfinite(point_2d).all():
            self.get_logger().warn(f"Invalid 2D point (NaN or infinite): u={point_2d[0]}, v={point_2d[1]}")
            return
        
        # Store current point to be displayed
        self.u = int(point_2d[0])
        self.v = int(point_2d[1])
        self.last_angle = angle
        self.last_distance = msg.y
        self.get_logger().info(f"Updated point: u={self.u}, v={self.v}")

        # Update trail
        self.trail_points.append((self.u, self.v))
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)  # Remove oldest point


def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
