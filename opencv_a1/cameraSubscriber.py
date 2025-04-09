import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')

        # ROS2 Subscriber (listens to /aruco_position)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/aruco_position',
            self.position_callback,
            10)

        self.previous_position = None  # Store last known position
        self.position_change_threshold = 3  # Min change in cm before printing

    def position_callback(self, msg):
        x, y, z = msg.data  # Extract position values

        # Check if position changed significantly
        if self.previous_position is None or any(
            abs(new - old) >= self.position_change_threshold
            for new, old in zip((x, y, z), self.previous_position)
        ):
            self.get_logger().info(f"Tracking ArUco - X={x:.1f} cm, Y={y:.1f} cm, Z={z:.1f} cm")
            self.previous_position = (x, y, z)  # Update last known position

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
