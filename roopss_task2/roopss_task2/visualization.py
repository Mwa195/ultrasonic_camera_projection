import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import cv2
import numpy as np
import pickle
import math
from ultralytics import YOLO

class VisualizerNode(Node):
    """
    Visualizer Node: Subscribes to /sensor_data to collect the readings
    and project the 3D points to 2D then display on the camera
    """
    def __init__(self):
        """
        Initialize visualization_node
        """
        super().__init__('visualization_node') # Initialize Node
        
        # Subscribe to ultrasonic sensor data
        self.create_subscription(Vector3, '/sensor_data', self.sensor_data_callback, 10)

        # Load camera calibration 
        with open('/home/mwa/ros2_ws/src/roopss_task2/roopss_task2/calibration_data.pkl', 'rb') as f:
            data = pickle.load(f)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['distortion_coefficients'].flatten()

        # Open camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().error('Could not open webcam')
            raise RuntimeError('Could not open webcam')

        # Sensor-to-camera transform
        self.R = np.array(
            [[0, 1, 0],
             [0, 0, -1],
             [1, 0, 0]
             ],dtype=np.float32)
        
        # Distance from camera to sensor
        self.t = np.array(
            [[0.0],
             [0.17],
             [0.28]
             ], dtype=np.float32)

        # Ultrasonic reference distance (meters)
        self.Z_ref = None
        self.point_cam = None

        # Load YOLO segmentation model
        self.yolo = YOLO('yolov8n-seg.pt')

        # Per-object stored info: {'center':(u,v), 'dist':cm}
        self.obj_info = []
        self.match_tol = 50  # pixels for matching centers

        # Update feed
        self.timer = self.create_timer(0.03, self.update_webcam_feed)
        self.get_logger().info('VisualizerNode initialized with object-mask logic')

    def sensor_data_callback(self, msg: Vector3):
        """
        Receives the sensor readings from /sensor_data and converts the points
        from 3D to 2D plane
        """
        angle = float(msg.x)
        dist_m = float(msg.y) / 100.0
        # Update reference if valid
        if 0.02 <= dist_m <= 4.0:
            if self.Z_ref != dist_m:
                self.get_logger().info(f'Ultrasonic distance updated: {dist_m*100:.1f} cm')
            self.Z_ref = dist_m
        else:
            self.get_logger().info('Ultrasonic lost sight — holding last distance')
        
        # Convert from Polar to Cartesian
        if self.Z_ref is not None:
            theta = math.radians(angle)
            x_s = self.Z_ref * math.cos(theta)
            y_s = -self.Z_ref * math.sin(theta)
            sensor_pt = np.array([[x_s], [y_s], [0.0]], dtype=np.float32)
            
            # Create the 3D array of the sensor
            self.point_cam = (self.R @ sensor_pt) + self.t

    def update_webcam_feed(self):
        """
        Updates the camera feed by drawing a point respective to
        sensor reading and detects an object using yolo then uses the
        sensor reading to say where it is
        """
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture frame')
            return

        # Project ultrasonic point
        beam_pt = None
        if self.point_cam is not None:
            p2d_s, _ = cv2.projectPoints(
                self.point_cam.reshape(1,1,3),
                np.zeros((3,1)), np.zeros((3,1)),
                self.camera_matrix, self.dist_coeffs)
            
            u_s, v_s = map(int, p2d_s[0,0])
            beam_pt = (u_s, v_s)
            cv2.circle(frame, beam_pt, 6, (255,0,0), -1)
            cv2.putText(frame, f'{self.Z_ref*100:.1f}cm', (u_s+10, v_s),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # YOLO segmentation detection
        results = self.yolo(frame, task='segment')[0]
        masks = results.masks.data  # torch.Tensor of shape (N, H, W)
        boxes = results.boxes.xyxy.cpu().numpy()  # (N,4)
        new_info = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            u_o = (x1 + x2) // 2
            v_o = (y1 + y2) // 2
            assigned = None
            
            # Convert mask to numpy
            mask_np = masks[idx].cpu().numpy().astype(np.uint8) * 255
            
            # Check mask at beam point
            if beam_pt is not None:
                if 0 <= v_s < mask_np.shape[0] and 0 <= u_s < mask_np.shape[1] and mask_np[v_s, u_s] > 0:
                    assigned = self.Z_ref * 100
            # If no assignment, reuse previous
            if assigned is None:
                for info in self.obj_info:
                    cx, cy = info['center']
                    if abs(cx - u_o) < self.match_tol and abs(cy - v_o) < self.match_tol:
                        assigned = info['dist']
                        break
            new_info.append({'center': (u_o, v_o), 'dist': assigned})

            # Draw mask contours
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0,255,0), 2)
            
            # Draw center and label
            cv2.circle(frame, (u_o, v_o), 5, (0,255,0), -1)
            if assigned is not None:
                cv2.putText(frame, f'{assigned:.1f}cm', (u_o-20, v_o-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        self.obj_info = new_info

        cv2.imshow('Ultrasonic + Multi-Object Seg', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args) # Initialize ROS2
    node = VisualizerNode() # Create Node as an instance of the Class
    try:
        rclpy.spin(node) # Keep the node spinning
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown() # Shutdown the node gracefully

if __name__ == '__main__':
    main()