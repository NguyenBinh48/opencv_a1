import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import cv2.aruco as aruco
import numpy as np

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        # ROS2 Publisher (publishes (X, Y, Z) coordinates)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/aruco_position', 10)

        self.publisher_2 = self.create_publisher(Image, '/image_data', 10)  

        # Camera setup
        self.cameraDevice = 0
        self.cap = cv2.VideoCapture(self.cameraDevice, cv2.CAP_V4L2)

        # ArUco dictionary and parameters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()

        # Camera calibration (Example values, replace with real ones)
        self.camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5, dtype=np.float32)

        # ArUco marker ID to track
        self.target_id = 23  

        # Timer to publish at 10 Hz

        FPS0 = 0.01666666666 # 60 FPS
        FPS1 = 0.02 # 50 FPS
        FPS2 = 0.03333333333 # 30 FPS
        FPS3 = 0.05 # 20 FPS

        self.FPS= FPS1
        self.timer = self.create_timer(self.FPS, self.publish_frame)

        self.bridge = CvBridge()

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to capture image")
            return
        
        msg_image_data = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher_2.publish(msg_image_data)



        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            ids = ids.flatten()

            if self.target_id in ids:
                index = np.where(int(ids) == self.target_id)[0][0]
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[index], 0.05, self.camera_matrix, self.dist_coeffs)

                if rvec is not None and tvec is not None:
                    x, y, z = tvec[0][0] * 100  # Convert meters to cm
                    msg = Float32MultiArray(data=[x, y, z])
                    self.publisher_.publish(msg)


                    # Draw bounding box around ArUco marker
                    corner_points = corners[index][0]
                    top_left = tuple(corner_points[0].astype(int))
                    bottom_right = tuple(corner_points[2].astype(int))
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                    # Draw ArUco marker and axes
                    aruco.drawDetectedMarkers(frame, corners)
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

        # Show the camera feed with detected ArUco marker
        # cv2.imshow("Camera Feed", frame)
        # cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
