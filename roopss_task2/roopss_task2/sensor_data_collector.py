import rclpy
from rclpy.node import Node
import serial
import serial.tools.list_ports
from geometry_msgs.msg import Vector3

class SensorDataCollector(Node):
    """
    Sensor Data Collector: Collects sensor data from arduino
    using serial communication at baud rate (9600) then publishes the
    data to /sensor_data topic to be processed
    """
    def __init__(self):
        """
        Initialze sensor_data_collector node
        """
        super().__init__("sensor_data_collector") # Initialize the Node
        
        # Setup Port
        ports = serial.tools.list_ports.comports() # Get ports
        available_ports = [port.device for port in ports] # Store available porst
        
        # A parameter for the port with '/dev/ttyUSB0' as default port  
        self.declare_parameter("port", '/dev/ttyUSB0')

        # Create a publisher to /sensor_data
        self.data_pub = self.create_publisher(
            Vector3,
            "/sensor_data",
            10
        )

        # Message variable
        self.msg = Vector3()

        # Example: Check if chosen port is available
        target_port = self.get_parameter("port").get_parameter_value().string_value
        if target_port in available_ports:
            print(f"{target_port} is available!")
            # Start serial communication
            self.ser = serial.Serial(target_port, 9600)
            # Loop the main function
            self.timer = self.create_timer(0.05, self.getSensorData)
        else:
            print(f"{target_port} not found. Available ports: {available_ports}")
            rclpy.shutdown()
        
        self.get_logger().info("Sensor Data Collector Initialized✅")

            
    def getSensorData(self):
        """
        Collects data sent from arudino, stores them and publish to /sensor_data
        """
        if self.ser.in_waiting: # check if
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    angleStr, distStr = line.split(",")
                    servo_angle = int(angleStr)
                    distance = float(distStr)
                    # Convert servo angle to relative angle (-60 to 60)
                    relative_angle = servo_angle - 90  # 90 degrees is the center (0 degrees)
                    if 2 <= distance <= 400:
                        self.get_logger().info(f"Relative Angle: {relative_angle}, Distance: {distance}")
                        self.msg.x = float(relative_angle)  # Publish relative angle (-60 to 60)
                        self.msg.y = float(distance)
                        self.data_pub.publish(self.msg)
                    else:
                        self.get_logger().warn(f"Distance out of range: {distance} cm")
            except Exception as e:
                self.get_logger().warn(f"Error parsing line: {e}")


def main(args=None):
    rclpy.init(args=args) # Initialize ROS2
    node = SensorDataCollector() # Create Node as an instance of the Class
    try:
        rclpy.spin(node) # Keep the node spinning
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown() # Shutdown the node gracefully

if __name__ == "__main__":
    main()