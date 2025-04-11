import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np

class ArucoImageViewer(Node):
    def __init__(self):
        super().__init__('aruco_image_viewer')

        # ROS2 Subscriber to image data
        self.subscription = self.create_subscription(
            Image,
            '/image_data',
            self.image_callback,
            10)

        self.bridge = CvBridge()

        # ArUco settings
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()
        self.target_id = 23
        self.camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5, dtype=np.float32)

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ArUco marker detection
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            ids = ids.flatten()
            if self.target_id in ids:
                index = np.where(int(ids) == self.target_id)[0][0]
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[index], 0.05, self.camera_matrix, self.dist_coeffs)

                aruco.drawDetectedMarkers(frame, [corners[index]])
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

                # Draw bounding box
                corner_points = corners[index][0]
                top_left = tuple(corner_points[0].astype(int))
                bottom_right = tuple(corner_points[2].astype(int))
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow("Aruco Image Viewer", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoImageViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
