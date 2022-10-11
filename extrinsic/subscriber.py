import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image


class SlamDataRecorder(Node):
    def __init__(self):
        super().__init__('slam_data_recorder')

        odom_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.color_sub = Subscriber(self, Image, "/color/preview/image")
        self.depth_sub = Subscriber(self, Image, "/stereo/depth")
        self.odom_sub = Subscriber(self, Odometry, "/odom", qos_profile=odom_qos_profile)

        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.odom_sub],
            30,
            0.5,
        )
        self.ts.registerCallback(self.callback)
        self.cb = CvBridge()
        self.count = 0

    def callback(self, color_msg, depth_msg, odom_msg):
        # color save
        cv2_img = self.cb.imgmsg_to_cv2(color_msg)
        cv2.imwrite(f'/home/simjoonyeol/ros2_ws/src/extrinsic/dataset/color/{self.count}.jpg', cv2_img)
        self.get_logger().info(f'Color Saved: {self.count}')

        # depth save
        cv2_img = self.cb.imgmsg_to_cv2(depth_msg)
        cv2.imwrite(f'/home/simjoonyeol/ros2_ws/src/extrinsic/dataset/depth/{self.count}.png', cv2_img)
        self.get_logger().info(f'Depth Saved: {self.count}')

        # tf save
        qx = odom_msg.pose.pose.orientation.z
        qy = odom_msg.pose.pose.orientation.y
        qz = -odom_msg.pose.pose.orientation.x
        qw = odom_msg.pose.pose.orientation.w

        tx = odom_msg.pose.pose.position.z
        ty = odom_msg.pose.pose.position.y
        tz = -odom_msg.pose.pose.position.x

        transformation_matrix = np.array(
            [[1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw, tx],
             [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw, ty],
             [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy, tz],
             [0.0, 0.0, 0.0, 1.0]])

        np.savetxt(f'/home/simjoonyeol/ros2_ws/src/extrinsic/dataset/pose/{self.count}.txt', transformation_matrix,
                   fmt='%f')
        self.get_logger().info(f'Odom Saved: {self.count}')

        self.count += 1


def main(args=None):
    if not os.path.exists('/home/simjoonyeol/ros2_ws/src/extrinsic/dataset'):
        os.mkdir('/home/simjoonyeol/ros2_ws/src/extrinsic/dataset')
        folder_list = ['color', 'depth', 'intrinsic', 'pose']
        for folder in folder_list:
            os.mkdir(f'/home/simjoonyeol/ros2_ws/src/extrinsic/dataset/{folder}')

    rclpy.init(args=args)
    slam_data_recorder = SlamDataRecorder()
    rclpy.spin(slam_data_recorder)
    slam_data_recorder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
