#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Team ID:          3577
Theme:            KRISHI COBOT
Author List:      Anudeep, Karthik, Vishwa, Manikanta
Filename:         task3b.py
Purpose:          Detect ArUco and bad fruits, publish TFs. Special orientation
                  correction applied for aruco_3 so UR5 will approach from the side.
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

# ----------------------------- ADDED CONSTANT -----------------------------
# ArUco marker ID on top of the eBot that indicates the DROP location
DROP_MARKER_ID = 6   # <-- CHANGE this to the actual ID used for drop marker
# -------------------------------------------------------------------------


# ----------------------------- HELPER FUNCTIONS -----------------------------

def calculate_rectangle_area(corners):
    c = np.array(corners)
    width = np.linalg.norm(c[0] - c[1])
    height = np.linalg.norm(c[1] - c[2])
    area = width * height
    return area, width


def detect_aruco(image):
    """
    Detect ArUco markers and return bounding / pose info.
    """
    # lowered threshold so markers slightly far still detected
    aruco_area_threshold = 800

    cam_mat = np.array([[915.3, 0.0, 642.7],
                        [0.0, 914.03, 361.97],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_mat = np.zeros(5, dtype=np.float64)
    size_of_aruco_m = 0.13  # meters (marker side)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(gray)
    center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, valid_ids = [], [], [], [], []
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

        if rvecs_out is not None and tvecs_out is not None and idx < len(tvecs_out):
            rvec = rvecs_out[idx].reshape(3)
            tvec = tvecs_out[idx].reshape(3)
            if np.isfinite(rvec).all() and np.isfinite(tvec).all() and tvec[2] > 0:
                try:
                    cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvec, size_of_aruco_m * 0.5)
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


# ----------------------------- MAIN CLASS ----------------------------------

class aruco_tf(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')
        self.color_cam_sub = self.create_subscription(
            Image, '/camera/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depthimagecb, 10)
        self.bridge = CvBridge()
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.12, self.process_image)
        self.cv_image = None
        self.depth_image = None
        self.team_id = 3577

    def depthimagecb(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
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

        # Camera intrinsics / image center
        centerCamX, centerCamY = 640.0, 360.0
        focalX, focalY = 931.1829833984375, 931.1829833984375

        # Offsets (tune these to your Gazebo camera mount)
        CAMERA_OFFSET_X = -0.18
        CAMERA_OFFSET_Y = -0.01999
        CAMERA_OFFSET_Z = 1.592
        # secondary camera/base offsets used for ArUco pose -> base_link conversion
        CAMERAA_OFFSET_X = -1.118
        CAMERAA_OFFSET_Y = -0.03
        CAMERAA_OFFSET_Z = 0.25

        img = self.cv_image.copy()
        c_list, d_list, a_list, w_list, ids, rvecs, tvecs = detect_aruco(img)
        bad_fruit_contours = detect_bad_fruits(img)

        # ---------------------- Bad fruits TF ----------------------
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

        # ---------------------- ArUco TFs (corrected orientation) ----------------------
        if ids is not None and len(ids) > 0 and tvecs is not None:
            for idx, marker_id in enumerate(ids):
                if idx >= len(tvecs):
                    continue
                tvec = tvecs[idx].reshape(3)
                # skip invalid pose
                if not np.isfinite(tvec).all() or tvec[2] <= 0.0:
                    continue

                # Camera-space marker coordinates (OpenCV convention)
                X_cam, Y_cam, Z_cam = float(tvec[0]), float(tvec[1]), float(tvec[2])

                # Convert marker camera coordinates to base_link coordinates
                # (these offsets are empirical; tune to your Gazebo camera mount)
                base_x = CAMERAA_OFFSET_X + (Z_cam * math.cos(math.radians(8))) + (Y_cam * math.sin(math.radians(8)))
                base_y = -(CAMERAA_OFFSET_Y + X_cam)
                base_z = CAMERAA_OFFSET_Z - (Y_cam * math.cos(math.radians(8))) + (Z_cam * math.sin(math.radians(8)))

                # marker-specific small tweaks (keep if you calibrated earlier)
                if marker_id == 6:
                    base_z += -0.87
                    base_y += -0.0
                    base_x += -0.06

                # Build a robust orientation quaternion for the marker frame
                try:
                    rvec = rvecs[idx].reshape(3)
                    rmat_cv, _ = cv2.Rodrigues(rvec)  # rotation matrix: OpenCV camera frame

                    # --------------------------
                    # Convert OpenCV -> ROS frame
                    # OpenCV camera frame: X right, Y down, Z forward
                    # ROS base_link/camera conventions: X right, Y forward, Z up (REP-103)
                    # We use a conversion matrix that maps cv coords -> ROS camera coords.
                    # --------------------------
                    cv_to_ros = np.array([
                        [1, 0, 0],
                        [0, 0, 1],
                        [0,-1, 0]
                    ])

                    # For side pick (marker_id == 3) we want the final marker frame to make the
                    # UR5 gripper approach from the side. We'll apply an extra rotation:
                    # rotate -90 degrees about the marker's Y axis (after cv->ros).
                    # This effectively makes the marker's +Z point sideways (so EEF can approach sideways).
                    # Note: angle in degrees for clarity.
                    side_pick_rotation = R.from_euler('xyz', [110, 90, 150], degrees=True).as_matrix()

                    # Compose final rotation matrix:
                    # 1) convert cv->ros
                    # 2) apply the side-pick rotation only for marker 3
                    rmat_ros = cv_to_ros @ rmat_cv

                    if marker_id == 3:
                        rmat_final = rmat_ros @ side_pick_rotation
                    else:
                        # for other markers keep top-down-ish orientation (rotate 180 deg about X)
                        # so that published TF's Z points upward in base_link frame (gripper top approach)
                        rot_x_180 = R.from_euler('x', 140, degrees=True).as_matrix()
                        rmat_final = rmat_ros @ rot_x_180

                    quat_marker = R.from_matrix(rmat_final).as_quat()

                except Exception as e:
                    self.get_logger().warn(f"ArUco orientation conversion failed for id {marker_id}: {e}")
                    # fallback to default quaternion pointing up
                    quat_marker = R.from_euler('xyz', [0, 0, 0]).as_quat()

                # Publish TF for this aruco marker
                t_obj = TransformStamped()
                t_obj.header.stamp = self.get_clock().now().to_msg()
                t_obj.header.frame_id = 'base_link'
                t_obj.child_frame_id = f"{self.team_id}_fertilizer_1"
                t_obj.transform.translation.x = float(base_x)
                t_obj.transform.translation.y = float(base_y)
                t_obj.transform.translation.z = float(base_z)
                t_obj.transform.rotation.x = float(quat_marker[0])
                t_obj.transform.rotation.y = float(quat_marker[1])
                t_obj.transform.rotation.z = float(quat_marker[2])
                t_obj.transform.rotation.w = float(quat_marker[3])

                self.br.sendTransform(t_obj)

                # ---------------------- ADDED: DROP LOCATION TF ----------------------
                # If this marker is the one on top of eBot, also publish a dedicated
                # drop-location frame: <team_id>_fertiliser_drop
                try:
                    if int(marker_id) == DROP_MARKER_ID:
                        t_drop = TransformStamped()
                        t_drop.header.stamp = t_obj.header.stamp
                        t_drop.header.frame_id = t_obj.header.frame_id   # 'base_link'
                        t_drop.child_frame_id = f"{self.team_id}_fertiliser_drop"

                        # Same position as computed for this ArUco
                        t_drop.transform.translation.x = float(base_x)
                        t_drop.transform.translation.y = float(base_y)
                        t_drop.transform.translation.z = float(base_z)

                        # Reuse same orientation (already corrected above)
                        t_drop.transform.rotation.x = float(quat_marker[0])
                        t_drop.transform.rotation.y = float(quat_marker[1])
                        t_drop.transform.rotation.z = float(quat_marker[2])
                        t_drop.transform.rotation.w = float(quat_marker[3])

                        self.br.sendTransform(t_drop)
                except Exception as e:
                    self.get_logger().warn(f"Drop TF publish failed for marker {marker_id}: {e}")
                # --------------------------------------------------------------------

        cv2.imshow("Bad Fruit Contours + ArUco Detection", img)
        cv2.waitKey(1)


def main():
    rclpy.init(args=sys.argv)
    node = aruco_tf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

