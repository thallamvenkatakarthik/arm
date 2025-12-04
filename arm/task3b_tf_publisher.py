#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Team ID:          3577
Theme:            KRISHI COBOT
Author List:      Anudeep, Karthik, Vishwa, Manikanta
Filename:         task3b_tf_publisher.py
Functions:        calculate_rectangle_area, detect_aruco, detect_bad_fruits, aruco_tf (class), main
Global variables: None

Purpose:
--------
ROS2 node that detects ArUco markers and bad fruits using RGB and Depth images.
Publishes TF frames for each detected ArUco marker and bad fruit so the UR5 can pick.
'''

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
import tf2_geometry_msgs   # noqa: F401 (kept for compatibility)


def calculate_rectangle_area(corners):
    c = np.array(corners)
    width = np.linalg.norm(c[0] - c[1])
    height = np.linalg.norm(c[1] - c[2])
    area = width * height
    return area, width


def detect_aruco(image):
    aruco_area_threshold = 1500
    cam_mat = np.array([[915.3, 0.0, 642.7],
                        [0.0, 914.03, 361.97],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_mat = np.zeros(5, dtype=np.float64)
    size_of_aruco_m = 0.13

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(gray)
    center_aruco_list, distance_from_rgb_list, angle_aruco_list = [], [], []
    width_aruco_list, valid_ids = [], []
    rvecs_out, tvecs_out = None, None

    if ids is None or len(ids) == 0:
        return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, valid_ids, None, None

    try:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, size_of_aruco_m, cam_mat, dist_mat)
        rvecs_out, tvecs_out = rvecs, tvecs
    except Exception as e:
        print(f"estimatePoseSingleMarkers exception: {e}")

    ids = ids.flatten()
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    for idx, corner in enumerate(corners):
        pts = corner[0].astype(np.float32)
        area, width = calculate_rectangle_area(pts)
        if area < aruco_area_threshold:
            continue

        cX, cY = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        yaw, marker_dist = 1.0, None

        if rvecs_out is not None and tvecs_out is not None:
            rvec, tvec = rvecs_out[idx].reshape(3), tvecs_out[idx].reshape(3)
            if np.isfinite(rvec).all() and np.isfinite(tvec).all() and tvec[2] > 0:
                try:
                    cv2.drawFrameAxes(
                        image, cam_mat, dist_mat, rvec, tvec, size_of_aruco_m * 0.5)
                except cv2.error:
                    pass
                rmat = cv2.Rodrigues(rvec)[0]
                yaw = math.atan2(rmat[1, 0], rmat[0, 0])
                marker_dist = float(tvec[2])

        center_aruco_list.append((cX, cY))
        distance_from_rgb_list.append(marker_dist)
        angle_aruco_list.append(yaw)
        width_aruco_list.append(width)
        valid_ids.append(int(ids[idx]))
        cv2.circle(image, (cX, cY), 6, (0, 255, 0), -1)

    return (center_aruco_list, distance_from_rgb_list,
            angle_aruco_list, width_aruco_list, valid_ids, rvecs_out, tvecs_out)


def detect_bad_fruits(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bad = np.array([0, 0, 100])
    upper_bad = np.array([179, 60, 255])
    mask = cv2.inRange(hsv, lower_bad, upper_bad)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tray_x1, tray_y1, tray_x2, tray_y2 = 4, 230, 340, 390
    filtered = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x >= tray_x1 and y >= tray_y1 and (x + w) <= tray_x2 and (y + h) <= tray_y2:
            if cv2.contourArea(cnt) > 200:
                filtered.append(cnt)
    return filtered


class aruco_tf(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher_3b')
        self.color_cam_sub = self.create_subscription(
            Image, '/camera/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depthimagecb, 10)

        self.bridge = CvBridge()
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.2, self.process_image)

        self.cv_image = None
        self.depth_image = None
        self.team_id = 3577

    def depthimagecb(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                data, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {str(e)}")

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Color image conversion failed: {str(e)}")

    def process_image(self):
        if self.cv_image is None:
            return

        centerCamX, centerCamY = 640.0, 360.0
        focalX, focalY = 931.1829833984375, 931.1829833984375

        CAMERA_OFFSET_X = -0.18
        CAMERA_OFFSET_Y = -0.01999
        CAMERA_OFFSET_Z = 1.532
        CAMERAA_OFFSET_X = -1.101
        CAMERAA_OFFSET_Y = -0.048
        CAMERAA_OFFSET_Z = 0.26

        img = self.cv_image.copy()
        c_list, d_list, a_list, w_list, ids, rvecs, tvecs = detect_aruco(img)
        bad_fruit_contours = detect_bad_fruits(img)

        depth_values = []
        for cnt in bad_fruit_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            if self.depth_image is not None:
                try:
                    raw = float(self.depth_image[int(cy), int(cx)])
                    depth_val = raw / 1000.0 if raw > 10.0 else raw
                    if np.isfinite(depth_val) and depth_val > 0.01:
                        depth_values.append(depth_val)
                except Exception:
                    pass

        common_depth = float(np.mean(depth_values)) if len(depth_values) > 0 else 0.55

        bad_fruit_id = 1
        for cnt in bad_fruit_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + int(0.2 * h)
            depth_value = common_depth

            X_cam = (float(cx) - centerCamX) * depth_value / focalX
            Y_cam = -(float(cy) - centerCamY) * depth_value / focalY
            Z_cam = depth_value

            base_x = CAMERA_OFFSET_X + Y_cam
            base_y = -(CAMERA_OFFSET_Y + X_cam)
            base_z = CAMERA_OFFSET_Z - Z_cam

            t_bad = TransformStamped()
            t_bad.header.stamp = self.get_clock().now().to_msg()
            t_bad.header.frame_id = 'base_link'
            t_bad.child_frame_id = f"{self.team_id}_bad_fruit_{bad_fruit_id}"
            t_bad.transform.translation.x = float(base_x)
            t_bad.transform.translation.y = float(base_y)
            t_bad.transform.translation.z = float(base_z)

            quat_down = R.from_euler('xyz', [math.pi, 0.0, 0.0]).as_quat()
            t_bad.transform.rotation.x = quat_down[0]
            t_bad.transform.rotation.y = quat_down[1]
            t_bad.transform.rotation.z = quat_down[2]
            t_bad.transform.rotation.w = quat_down[3]
            self.br.sendTransform(t_bad)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(img, f"bad_fruit_{bad_fruit_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            bad_fruit_id += 1

        # ------------- ArUco TFs (fertiliser + eBot top etc.) -------------
        if ids is not None and len(ids) > 0 and tvecs is not None:
            for idx, marker_id in enumerate(ids):
                if idx >= len(tvecs):
                    continue
                tvec = tvecs[idx].reshape(3)
                if not np.isfinite(tvec).all() or tvec[2] <= 0.0:
                    continue

                X_cam, Y_cam, Z_cam = float(tvec[0]), float(tvec[1]), float(tvec[2])

                base_x = CAMERAA_OFFSET_X + (Z_cam * math.cos(math.radians(8))) + (Y_cam * math.sin(math.radians(8)))
                base_y = -(CAMERAA_OFFSET_Y + X_cam)
                base_z = CAMERAA_OFFSET_Z - (Y_cam * math.cos(math.radians(8))) + (Z_cam * math.sin(math.radians(8)))

                if marker_id == 6:
                    base_z += -0.87
                    base_y += -0.04
                    base_x += -0.06

                try:
                    rvec = rvecs[idx].reshape(3)
                    rmat, _ = cv2.Rodrigues(rvec)

                    flip_y = np.array([[1, 0, 0],
                                       [0, -1, 0],
                                       [0, 0, 1]])
                    rmat = flip_y @ rmat

                    rot_z_down = R.from_euler('x', math.radians(180)).as_matrix()
                    rmat = rmat @ rot_z_down

                    tilt = R.from_euler('y', math.radians(8)).as_matrix()
                    rmat = tilt @ rmat

                    quat_marker = R.from_matrix(rmat).as_quat()
                except Exception:
                    quat_marker = R.from_euler('xyz', [0, 0, 0]).as_quat()

                t_obj = TransformStamped()
                t_obj.header.stamp = self.get_clock().now().to_msg()
                t_obj.header.frame_id = 'base_link'
                t_obj.child_frame_id = f"aruco_{marker_id}"
                t_obj.transform.translation.x = float(base_x)
                t_obj.transform.translation.y = float(base_y)
                t_obj.transform.translation.z = float(base_z)
                t_obj.transform.rotation.x = quat_marker[0]
                t_obj.transform.rotation.y = quat_marker[1]
                t_obj.transform.rotation.z = quat_marker[2]
                t_obj.transform.rotation.w = quat_marker[3]
                self.br.sendTransform(t_obj)

        cv2.imshow("Bad Fruit Contours + ArUco Detection", img)
        cv2.waitKey(1)


def main():
    rclpy.init(args=sys.argv)
    node = aruco_tf()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

