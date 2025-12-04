#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Team ID:          3577
Theme:            KRISHI COBOT
Author List:      Anudeep, Karthik, Vishwa, Manikanta 
Filename:         task3a_tf_publisher.py
Purpose:          Detect bad fruits and fertilizer can (ArUco) and publish TFs
                  Naming convention:
                    <team_id>_bad_fruit_<id>
                    <team_id>_fertilizer_1
                  All TFs published w.r.t. 'base_link'.
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
from sensor_msgs.msg import Image, CameraInfo
from scipy.spatial.transform import Rotation as R

# ---------------- CONFIG / TUNABLES ----------------
TEAM_ID = "3577"                    # used in TF names
MAX_BAD_FRUITS = 3                  # publish up to this many bad fruits
ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_SIZE_M = 0.13                 # marker side in meters (adjust if needed)
ARUCO_MIN_AREA = 800                # minimum pixel area to accept marker
BAD_FRUIT_MIN_AREA = 200            # min contour area to consider fruit
TRAY_ROI = (4, 230, 340, 390)       # crop for fruit detection (x1,y1,x2,y2) in pixels
DEPTH_MIN_M = 0.01                  # ignore very small depths
FALLBACK_DEPTH_M = 0.55             # fallback if no valid depth found
# offsets to convert camera coords to base_link — tune to your hardware
CAM_OFFSET_X = -0.18
CAM_OFFSET_Y = -0.01999
CAM_OFFSET_Z = 1.532
# If you calibrated a secondary offset used for ArUco->base_link, set here:
ARUCO_CAM_OFFSET_X = -1.101
ARUCO_CAM_OFFSET_Y = -0.048
ARUCO_CAM_OFFSET_Z = 0.26

# ----------------------------- HELPER FUNCTIONS -----------------------------

def calculate_rectangle_area(corners):
    c = np.array(corners)
    width = np.linalg.norm(c[0] - c[1])
    height = np.linalg.norm(c[1] - c[2])
    return width * height, width

def detect_aruco(image_gray):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rejected = detector.detectMarkers(image_gray)
    if ids is None or len(ids) == 0:
        return [], [], None
    ids = ids.flatten()
    return corners, ids, detector

def detect_bad_fruits(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # HSV range tuned for pale/bad fruit (adjust if needed)
    lower_bad = np.array([0, 0, 100])
    upper_bad = np.array([179, 60, 255])
    mask = cv2.inRange(hsv, lower_bad, upper_bad)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x1, y1, x2, y2 = TRAY_ROI
    filtered = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x >= x1 and y >= y1 and (x + w) <= x2 and (y + h) <= y2:
            if cv2.contourArea(cnt) > BAD_FRUIT_MIN_AREA:
                filtered.append((x, y, w, h, cnt))
    # sort by area descending (largest first) to have deterministic indexing
    filtered.sort(key=lambda t: cv2.contourArea(t[4]), reverse=True)
    return filtered[:MAX_BAD_FRUITS]

def depth_value_from_pixel(depth_img, px, py):
    # depth_img could be 16UC1 (millimeters) or 32FC1 (meters).
    try:
        raw = depth_img[int(py), int(px)]
    except Exception:
        return None
    if raw is None:
        return None
    # If dtype is uint16, typical scale is mm -> convert to meters
    if np.issubdtype(depth_img.dtype, np.integer):
        raw = float(raw)
        if raw <= 10.0:
            # often zero or invalid
            return None
        return raw / 1000.0
    else:
        # float32
        raw = float(raw)
        if not np.isfinite(raw) or raw <= 0.0:
            return None
        return raw

# ----------------------------- ROS2 NODE -----------------------------

class TaskTFPublisher(Node):
    def __init__(self):
        super().__init__('task3a_tf_publisher')
        # Subscribe to the exact topics required
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_cb, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_cb, 10)
        self.caminfo_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.caminfo_cb, 10)

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.12, self.timer_cb)

        self.cv_image = None
        self.depth_image = None
        self.cam_intrinsics = None  # will be a dict with fx, fy, cx, cy, cam_mat

        # default intrinsics (fallback)
        self.default_intr = {
            'fx': 931.1829833984375,
            'fy': 931.1829833984375,
            'cx': 640.0,
            'cy': 360.0,
            'cam_mat': np.array([[915.3, 0.0, 642.7],
                                 [0.0, 914.03, 361.97],
                                 [0.0, 0.0, 1.0]], dtype=np.float64)
        }

        self.get_logger().info("TaskTFPublisher initialized. Waiting for images...")

    def caminfo_cb(self, msg: CameraInfo):
        try:
            K = np.array(msg.k).reshape((3, 3))
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            self.cam_intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'cam_mat': K}
        except Exception as e:
            self.get_logger().warn(f"Failed to parse CameraInfo: {e}")

    def depth_cb(self, msg: Image):
        try:
            # keep raw encoding to decide conversion later
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")
            self.depth_image = None

    def color_cb(self, msg: Image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Color image conversion failed: {e}")
            self.cv_image = None

    def timer_cb(self):
        if self.cv_image is None:
            return

        intr = self.cam_intrinsics if self.cam_intrinsics is not None else self.default_intr
        fx = intr['fx']
        fy = intr['fy']
        cx = intr['cx']
        cy = intr['cy']
        cam_mat = intr['cam_mat']

        img = self.cv_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---- ArUco detection (for fertilizer can) ----
        corners, ids, detector = detect_aruco(gray)
        fertilizer_published = False
        if ids is not None and len(ids) > 0 and detector is not None:
            # compute poses if possible
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, ARUCO_SIZE_M, cam_mat, np.zeros(5))
            except Exception:
                rvecs, tvecs = None, None

            for idx, marker_id in enumerate(ids):
                pts = corners[idx][0].astype(np.float32)
                area, _ = calculate_rectangle_area(pts)
                if area < ARUCO_MIN_AREA:
                    continue
                # If we have pose estimate, use it to compute TF for fertilizer
                if tvecs is not None and idx < len(tvecs):
                    tvec = tvecs[idx].reshape(3)
                    if not np.isfinite(tvec).all() or tvec[2] <= 0.0:
                        continue

                    # Convert OpenCV marker pose into base_link coordinates
                    X_cam_cv, Y_cam_cv, Z_cam_cv = float(tvec[0]), float(tvec[1]), float(tvec[2])

                    # Apply empirical conversion (tune if needed)
                    base_x = ARUCO_CAM_OFFSET_X + (Z_cam_cv * math.cos(math.radians(8))) + (Y_cam_cv * math.sin(math.radians(8)))
                    base_y = -(ARUCO_CAM_OFFSET_Y + X_cam_cv)
                    base_z = ARUCO_CAM_OFFSET_Z - (Y_cam_cv * math.cos(math.radians(8))) + (Z_cam_cv * math.sin(math.radians(8)))

                    # Build rotation: convert rvec -> rotation matrix (OpenCV), then to ROS frame
                    try:
                        rvec = rvecs[idx].reshape(3)
                        rmat_cv, _ = cv2.Rodrigues(rvec)  # OpenCV camera frame
                        # OpenCV camera frame: X-right, Y-down, Z-forward
                        # Convert to ROS camera frame: X-right, Y-forward, Z-up
                        cv_to_ros = np.array([[1, 0, 0],
                                              [0, 0, 1],
                                              [0, -1, 0]])
                        rmat_ros = cv_to_ros @ rmat_cv

                        # Optionally rotate marker frame to standard orientation (so fertilizer frame is stable)
                        # rotate about X to make Z upwards if needed
                        rot_x = R.from_euler('x', 140, degrees=True).as_matrix()
                        rmat_final = rmat_ros @ rot_x

                        quat = R.from_matrix(rmat_final).as_quat()
                    except Exception:
                        quat = R.from_euler('xyz', [0, 0, 0]).as_quat()

                    # Publish fertilizer TF with required naming convention
                    t_fert = TransformStamped()
                    t_fert.header.stamp = self.get_clock().now().to_msg()
                    t_fert.header.frame_id = 'base_link'
                    t_fert.child_frame_id = f"{TEAM_ID}_fertilizer_1"
                    t_fert.transform.translation.x = float(base_x)
                    t_fert.transform.translation.y = float(base_y)
                    t_fert.transform.translation.z = float(base_z)
                    t_fert.transform.rotation.x = float(quat[0])
                    t_fert.transform.rotation.y = float(quat[1])
                    t_fert.transform.rotation.z = float(quat[2])
                    t_fert.transform.rotation.w = float(quat[3])
                    self.tf_broadcaster.sendTransform(t_fert)
                    fertilizer_published = True

                    # draw
                    cv2.drawFrameAxes(img, cam_mat, np.zeros(5), rvec, tvec, ARUCO_SIZE_M * 0.5)
                    cv2.putText(img, f"FERT_{marker_id}", (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    break  # only publish one fertilizer TF (first valid)
        # If no aruco found, fertilizer remains unpublished (robot will get no TF)

        # ---- Bad fruit detection and TF publishing ----
        bads = detect_bad_fruits(img)
        depth_values = []
        # collect candidate depths from each detection center for averaging
        for (x, y, w, h, cnt) in bads:
            cx = x + w // 2
            cy = y + h // 2
            if self.depth_image is not None:
                d = depth_value_from_pixel(self.depth_image, cx, cy)
                if d is not None and d > DEPTH_MIN_M and np.isfinite(d):
                    depth_values.append(d)

        common_depth = float(np.mean(depth_values)) if len(depth_values) > 0 else FALLBACK_DEPTH_M

        fruit_id = 1
        for (x, y, w, h, cnt) in bads:
            cx = x + w // 2
            cy = y + int(0.2 * h)  # slightly above center
            depth_val = common_depth
            # if possible, override with per-pixel depth
            if self.depth_image is not None:
                d_local = depth_value_from_pixel(self.depth_image, cx, cy)
                if d_local is not None and np.isfinite(d_local) and d_local > DEPTH_MIN_M:
                    depth_val = d_local

            # Project pixel -> camera coordinates (OpenCV camera frame)
            X_cam = (float(cx) - cx) * depth_val / fx   # bug: corrected below
            # above line erroneously uses local cx variable; fix actual formulas below

            X_cam = (float(cx) - intr['cx']) * depth_val / fx
            Y_cam = -(float(cy) - intr['cy']) * depth_val / fy
            Z_cam = depth_val

            # Convert camera coords to base_link coords (empirical offsets)
            base_x = CAM_OFFSET_X + Y_cam
            base_y = -(CAM_OFFSET_Y + X_cam)
            base_z = CAM_OFFSET_Z - Z_cam

            t_bad = TransformStamped()
            t_bad.header.stamp = self.get_clock().now().to_msg()
            t_bad.header.frame_id = 'base_link'
            t_bad.child_frame_id = f"{TEAM_ID}_bad_fruit_{fruit_id}"
            t_bad.transform.translation.x = float(base_x)
            t_bad.transform.translation.y = float(base_y)
            t_bad.transform.translation.z = float(base_z)

            # orientation: make frame point upwards (quat for 180 deg about X to align)
            quat_down = R.from_euler('xyz', [math.pi, 0.0, 0.0]).as_quat()
            t_bad.transform.rotation.x = float(quat_down[0])
            t_bad.transform.rotation.y = float(quat_down[1])
            t_bad.transform.rotation.z = float(quat_down[2])
            t_bad.transform.rotation.w = float(quat_down[3])

            self.tf_broadcaster.sendTransform(t_bad)

            # draw visuals
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(img, f"{TEAM_ID}_bad_fruit_{fruit_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            fruit_id += 1

        # Optionally show the image - comment out if running headless on remote server
        try:
            cv2.imshow("Detection - Bad Fruits & Fertilizer", img)
            cv2.waitKey(1)
        except Exception:
            # headless environment may raise
            pass

def main(args=None):
    rclpy.init(args=args)
    node = TaskTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()