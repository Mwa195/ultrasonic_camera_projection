#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from ultralytics import YOLO

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_detection_node")

        # Load YOLO model
        self.yolo = YOLO("yolov8n.pt")

        # Open camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open webcam")
            raise RuntimeError("Webcam not available")

        self.timer = self.create_timer(0.03, self.detect_and_display)

        self.get_logger().info("YOLO Detection Node Initialized ✅")

    def detect_and_display(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to read from camera")
            return

        results = self.yolo(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = self.yolo.names[class_id]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("YOLO Detections", frame)
        cv2.waitKey(1)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
