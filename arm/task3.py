#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Team ID:          3577
Theme:            KRISHI COBOT
Author List:      D Anudeep, Karthik, Vishwa, Manikanta
Filename:         task3a_tf_publisher.py
Purpose:          Detect ArUco and bad fruits, publish TFs for Task 3A.
                  - 2 bad fruits: <team_id>_bad_fruit_<id>
                  - 1 fertilizer can: <team_id>_fertilizer_1
                  - Drop location on eBot top (ArUco): <team_id>_fertilizer_drop
                  All TFs are w.r.t. base_link.
Improved: rejects colourful (purple) good fruit using Lab-chroma test.
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

# ----------------------------- ADDED CONSTANTS -----------------------------
# ArUco marker ID on top of the eBot that indicates the DROP location
DROP_MARKER_ID = 6

# Maximum number of bad fruits to publish TF for (Task says 3 bad fruits, we use 2 grey ones)
MAX_BAD_FRUITS = 2

# Chromaticity threshold (CIELab chroma). Anything above this is considered "colourful" -> REJECT.
CHROMA_THRESH = 18.0

# Tray bbox (tune if needed)
TRAY_X1, TRAY_Y1, TRAY_X2, TRAY_Y2 = 108, 445, 355, 591
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


# ------------------ Improved bad fruit detection (reject colourful fruit) ------------------

def median_depth_in_bbox(depth_img, x, y, w, h):
    """Return median depth (meters) in bbox or None if invalid."""
    if depth_img is None:
        return None
    h_img, w_img = depth_img.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w_img, int(x + w))
    y2 = min(h_img, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth_img[y1:y2, x1:x2].astype(np.float32).flatten()
    roi = roi[np.isfinite(roi)]
    roi = roi[roi > 0]
    if roi.size == 0:
        return None
    med = float(np.median(roi))
    # convert mm->m heuristic if needed
    if med > 10.0:
        med = med / 1000.0
    if med < 0.01:
        return None
    return med


def detect_bad_fruits(image):
    """
    Improved detection:
     - Primary: detect green tops using HSV + HoughCircles
     - Expand each top to a body bbox
     - Fallback: grey-body contour detection (as original)
     - Reject any candidate whose Lab chroma > CHROMA_THRESH
    Returns: list of contours (rect-like arrays or real contours)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    img_h, img_w = image.shape[:2]

    # ---- 1) Green top mask (broader range) ----
    lower_green = np.array([40, 30, 50], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ---- 2) Hough circle on blurred mask to find circular tops ----
    mask_blur = cv2.GaussianBlur(mask, (7, 7), 0)
    circles = None
    try:
        circles = cv2.HoughCircles(mask_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=50, param2=12, minRadius=6, maxRadius=60)
    except Exception:
        circles = None

    detected_contours = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (cx, cy, r) in circles:
            x_full = max(0, cx - int(1.2 * r))
            y_full = max(0, cy - int(0.8 * r))
            w_full = min(img_w - x_full, int(2.4 * r))
            h_full = min(img_h - y_full, int(2.6 * r))
            rect = np.array([[[x_full, y_full]], [[x_full + w_full, y_full]],
                             [[x_full + w_full, y_full + h_full]], [[x_full, y_full + h_full]]])
            detected_contours.append(rect)

    # If Hough didn't find anything, try to find connected green blobs
    if len(detected_contours) == 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 40:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x_full = max(0, x - int(w * 0.5))
            w_full = min(img_w - x_full, int(w * 2.0))
            y_full = max(0, y - int(h * 0.2))
            h_full = min(img_h - y_full, int(h * 2.0))
            rect = np.array([[[x_full, y_full]], [[x_full + w_full, y_full]],
                             [[x_full + w_full, y_full + h_full]], [[x_full, y_full + h_full]]])
            detected_contours.append(rect)

    # Fallback: grey-body detection (if no green top found)
    if len(detected_contours) == 0:
        lower_grey_dark = np.array([0, 0, 18], dtype=np.uint8)
        upper_grey_dark = np.array([45, 65, 130], dtype=np.uint8)
        lower_grey_light = np.array([0, 0, 70], dtype=np.uint8)
        upper_grey_light = np.array([45, 70, 210], dtype=np.uint8)

        mask_dark = cv2.inRange(hsv, lower_grey_dark, upper_grey_dark)
        mask_light = cv2.inRange(hsv, lower_grey_light, upper_grey_light)
        mask_grey = cv2.bitwise_or(mask_dark, mask_light)

        kernel2 = np.ones((5, 5), np.uint8)
        mask_grey = cv2.morphologyEx(mask_grey, cv2.MORPH_OPEN, kernel2)
        mask_grey = cv2.morphologyEx(mask_grey, cv2.MORPH_CLOSE, kernel2, iterations=2)

        contours, _ = cv2.findContours(mask_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 70:
                detected_contours.append(cnt)

    # Final filter: tray overlap + Lab-chroma rejection
    filtered = []
    for rect in detected_contours:
        x, y, w, h = cv2.boundingRect(rect)
        if w <= 2 or h <= 2:
            continue

        # tray overlap check (allow partial overlap)
        inter_x1 = max(x, TRAY_X1); inter_y1 = max(y, TRAY_Y1)
        inter_x2 = min(x + w, TRAY_X2); inter_y2 = min(y + h, TRAY_Y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        rect_area = float(max(1, w * h))
        if inter_area < 0.2 * rect_area:
            continue

        # compute Lab chroma inside this bbox
        x0 = max(0, int(x)); y0 = max(0, int(y))
        x1 = min(img_w, int(x + w)); y1 = min(img_h, int(y + h))
        patch = lab[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        a = patch[:, :, 1].astype(np.float32) - 128.0
        b = patch[:, :, 2].astype(np.float32) - 128.0
        chroma = np.sqrt(a * a + b * b)
        mean_chroma = float(np.nanmean(chroma))
        # Reject colourful patches (purple good fruit)
        if mean_chroma > CHROMA_THRESH:
            # skip - likely colored fruit (good)
            continue

        # area threshold safe-check
        if cv2.contourArea(rect) < 50:
            continue

        filtered.append(rect)

    return filtered


def refine_with_green_top(hsv, x, y, w, h):
    """
    From a grey fruit body bbox (x,y,w,h), search above it for the green top.
    Returns:
        x_full, y_full, w_full, h_full, cx_top, cy_top
    If no green found, fall back to body bbox and body top-centre.
    """
    img_h, img_w = hsv.shape[:2]

    SEARCH_UP_FACTOR = 0.9
    roi_top_y = max(0, int(y - SEARCH_UP_FACTOR * h))
    roi_bottom_y = min(img_h, y + h)
    roi_left_x = max(0, x - int(0.25 * w))
    roi_right_x = min(img_w, x + w + int(0.25 * w))

    roi_hsv = hsv[roi_top_y:roi_bottom_y, roi_left_x:roi_right_x]
    if roi_hsv.size == 0:
        cx_top = x + w // 2
        cy_top = y
        return x, y, w, h, cx_top, cy_top

    lower_green = np.array([40, 35, 55], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(roi_hsv, lower_green, upper_green)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Hough circles
    top_centroid = None
    try:
        mask_blur = cv2.GaussianBlur(mask, (7, 7), 0)
        circles = cv2.HoughCircles(mask_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                   param1=40, param2=10, minRadius=4, maxRadius=40)
        if circles is not None:
            circles = np.uint16(np.around(circles))[0, :]
            circles = sorted(circles, key=lambda c: c[2], reverse=True)
            gx, gy, _ = circles[0]
            gx_global = roi_left_x + int(gx)
            gy_global = roi_top_y + int(gy)
            top_centroid = (gx_global, gy_global)
    except Exception:
        top_centroid = None

    if top_centroid is None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            top_cnt = max(contours, key=cv2.contourArea)
            tx, ty, tw, th = cv2.boundingRect(top_cnt)
            gx_global = roi_left_x + tx + tw // 2
            gy_global = roi_top_y + ty + th // 2
            top_centroid = (gx_global, gy_global)

    if top_centroid is None:
        cx_top = x + w // 2
        cy_top = y
        return x, y, w, h, cx_top, cy_top

    gx, gy = top_centroid
    g_w = int(0.6 * w) if w > 0 else 20
    g_h = int(0.5 * h) if h > 0 else 12
    gx0 = max(0, gx - g_w // 2)
    gy0 = max(0, gy - g_h // 2)

    x_full = min(x, gx0)
    y_full = min(y, gy0)
    x_full2 = max(x + w, gx0 + g_w)
    y_full2 = max(y + h, gy0 + g_h)
    w_full = x_full2 - x_full
    h_full = y_full2 - y_full

    pad_x = int(0.08 * w_full) + 2
    pad_y = int(0.08 * h_full) + 2
    x_full = max(0, x_full - pad_x)
    y_full = max(0, y_full - pad_y)
    w_full = min(img_w - x_full, w_full + 2 * pad_x)
    h_full = min(img_h - y_full, h_full + 2 * pad_y)

    cx_top = gx
    cy_top = gy

    return x_full, y_full, w_full, h_full, cx_top, cy_top


# ----------------------------- MAIN CLASS ----------------------------------

class aruco_tf(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')

        # Color & depth topics from remote hardware
        self.color_cam_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)

        self.bridge = CvBridge()
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.12, self.process_image)
        self.cv_image = None
        self.depth_image = None
        self.team_id = 3577  # used in TF names

    def depthimagecb(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {str(e)}")
            self.depth_image = None

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Color image conversion failed: {str(e)}")
            self.cv_image = None

    def process_image(self):
        if self.cv_image is None:
            return

        # Camera intrinsics / image center
        centerCamX, centerCamY = 640.0, 360.0
        focalX, focalY = 931.1829833984375, 931.1829833984375

        # Offsets (tune these to your Gazebo/real camera mount)
        CAMERA_OFFSET_X = 0.11
        CAMERA_OFFSET_Y = -0.01999
        CAMERA_OFFSET_Z = 1.452
        # secondary camera/base offsets used for ArUco pose -> base_link conversion
        CAMERAA_OFFSET_X = -1.118
        CAMERAA_OFFSET_Y = -0.08
        CAMERAA_OFFSET_Z = 0.26

        img = self.cv_image.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        c_list, d_list, a_list, w_list, ids, rvecs, tvecs = detect_aruco(img)
        bad_fruit_contours = detect_bad_fruits(img)

        # ---------------------- Bad fruits TF ----------------------
        depth_values = []
        for cnt in bad_fruit_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx_mid, cy_mid = x + w // 2, y + h // 2
            if self.depth_image is not None:
                try:
                    raw = float(self.depth_image[int(cy_mid), int(cx_mid)])
                    depth_val = raw / 1000.0 if raw > 10.0 else raw
                    if np.isfinite(depth_val) and depth_val > 0.01:
                        depth_values.append(depth_val)
                except Exception:
                    pass

        common_depth = float(np.mean(depth_values)) if len(depth_values) > 0 else 0.55

        bad_fruit_id = 1
        for cnt in bad_fruit_contours:
            # LIMIT to MAX_BAD_FRUITS
            if bad_fruit_id > MAX_BAD_FRUITS:
                break

            x_body, y_body, w_body, h_body = cv2.boundingRect(cnt)

            # extend to full fruit using green top, get top-centre
            x_full, y_full, w_full, h_full, cx_top, cy_top = refine_with_green_top(
                hsv, x_body, y_body, w_body, h_body
            )

            depth_value = common_depth

            # back-project using top-centre pixel
            X_cam = (float(cx_top) - centerCamX) * depth_value / focalX
            Y_cam = -(float(cy_top) - centerCamY) * depth_value / focalY
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

            # draw full fruit bbox and top-centre dot
            cv2.rectangle(img, (int(x_full), int(y_full)),
                          (int(x_full + w_full), int(y_full + h_full)),
                          (0, 255, 0), 2)
            cv2.circle(img, (int(cx_top), int(cy_top)), 6, (0, 255, 0), -1)
            cv2.putText(img, f"bad_fruit_{bad_fruit_id}", (int(x_full), int(y_full) - 10),
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
                base_x = CAMERAA_OFFSET_X + (Z_cam * math.cos(math.radians(8))) + (Y_cam * math.sin(math.radians(8)))
                base_y = -(CAMERAA_OFFSET_Y + X_cam)
                base_z = CAMERAA_OFFSET_Z - (Y_cam * math.cos(math.radians(8))) + (Z_cam * math.sin(math.radians(8)))

                # marker-specific small tweaks
                if marker_id == 6:
                    base_z += -0.87
                    base_y += -0.0
                    base_x += -0.06

                # Build a robust orientation quaternion for the marker frame
                try:
                    rvec = rvecs[idx].reshape(3)
                    rmat_cv, _ = cv2.Rodrigues(rvec)  # rotation matrix: OpenCV camera frame

                    # Convert OpenCV -> ROS camera frame
                    cv_to_ros = np.array([
                        [1, 0, 0],
                        [0, 0, 1],
                        [0,-1, 0]
                    ])

                    # Side-pick extra rotation for specific marker (e.g. marker_id == 3)
                    side_pick_rotation = R.from_euler('xyz', [110, 90, 150], degrees=True).as_matrix()

                    # Compose final rotation matrix:
                    rmat_ros = cv_to_ros @ rmat_cv

                    if marker_id == 3:
                        rmat_final = rmat_ros @ side_pick_rotation
                    else:
                        # for other markers keep top-down-ish orientation
                        rot_x_140 = R.from_euler('x', 140, degrees=True).as_matrix()
                        rmat_final = rmat_ros @ rot_x_140

                    quat_marker = R.from_matrix(rmat_final).as_quat()

                except Exception as e:
                    self.get_logger().warn(f"ArUco orientation conversion failed for id {marker_id}: {e}")
                    # fallback to default quaternion
                    quat_marker = R.from_euler('xyz', [0, 0, 0]).as_quat()

                # Fertilizer can TF
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

                # DROP LOCATION TF if this is the drop marker
                try:
                    if int(marker_id) == DROP_MARKER_ID:
                        t_drop = TransformStamped()
                        t_drop.header.stamp = t_obj.header.stamp
                        t_drop.header.frame_id = t_obj.header.frame_id   # 'base_link'
                        t_drop.child_frame_id = f"{self.team_id}_fertilizer_drop"

                        t_drop.transform.translation.x = float(base_x)
                        t_drop.transform.translation.y = float(base_y)
                        t_drop.transform.translation.z = float(base_z)

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