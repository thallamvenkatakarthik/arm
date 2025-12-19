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

Updated improvements:
 - per-fruit median depth back-projection
 - CIELab chroma filter to reject colored fruits (purple)
 - temporal smoothing (low-pass) for TF poses to remove RViz jitter
 - improved green-top refinement and bbox expansion
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
DROP_MARKER_ID = 6
MAX_BAD_FRUITS = 2
# smoothing alpha: 0.2-0.4 recommended (lower -> smoother, higher -> more responsive)
SMOOTH_ALPHA = 0.28
CHROMA_THRESH = 18.0   # lab chroma threshold to reject colourful fruits (tune if needed)
# tray crop (frame-specific)
TRAY_X1, TRAY_Y1, TRAY_X2, TRAY_Y2 = 108, 445, 355, 591
# -------------------------------------------------------------------------


# ---------- Temporal smoothing helper ----------
class SmoothedPublisher:
    """
    Low-pass filter for pose (translation + quaternion).
    """
    def __init__(self, alpha=0.28):
        self.alpha = float(alpha)
        self.store = {}  # key -> {'pos': np.array(3), 'quat': np.array(4)}

    @staticmethod
    def _normalize(q):
        q = np.array(q, dtype=float)
        n = np.linalg.norm(q)
        return q / n if n > 1e-12 else np.array([0, 0, 0, 1.0])

    @staticmethod
    def _slerp(q0, q1, t):
        q0 = SmoothedPublisher._normalize(q0)
        q1 = SmoothedPublisher._normalize(q1)
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        DOT_THRESH = 0.9995
        if dot > DOT_THRESH:
            res = q0 + t*(q1 - q0)
            res /= np.linalg.norm(res)
            return res
        theta_0 = math.acos(max(min(dot, 1.0), -1.0))
        theta = theta_0 * t
        q2 = q1 - q0 * dot
        q2 /= np.linalg.norm(q2)
        return q0*math.cos(theta) + q2*math.sin(theta)

    def update(self, key, pos, quat):
        pos = np.array(pos, dtype=float)
        quat = np.array(quat, dtype=float)
        quat = self._normalize(quat)
        if key not in self.store:
            self.store[key] = {'pos': pos, 'quat': quat}
            return pos, quat
        prev = self.store[key]
        new_pos = (1.0 - self.alpha) * prev['pos'] + self.alpha * pos
        new_quat = self._slerp(prev['quat'], quat, self.alpha)
        new_quat /= np.linalg.norm(new_quat)
        self.store[key] = {'pos': new_pos, 'quat': new_quat}
        return new_pos, new_quat


# ----------------------------- HELPER FUNCTIONS -----------------------------

def calculate_rectangle_area(corners):
    c = np.array(corners)
    width = np.linalg.norm(c[0] - c[1])
    height = np.linalg.norm(c[1] - c[2])
    return width * height, width


def median_depth_in_bbox(depth_img, x, y, w, h, clip_min=0.01, invalid_thresh=10.0):
    """
    Compute robust median depth (in meters) inside bbox. Handles mm->m conversion heuristics.
    """
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
    if roi.size == 0:
        return None
    roi = roi[np.isfinite(roi)]
    roi = roi[roi > 0]
    if roi.size == 0:
        return None
    med = float(np.median(roi))
    if med > invalid_thresh:
        med = med / 1000.0
    if med < clip_min:
        return None
    return med


def detect_aruco(image):
    aruco_area_threshold = 800
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
    center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, valid_ids = [], [], [], [], []
    rvecs_out, tvecs_out = None, None

    if ids is None or len(ids) == 0:
        return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, valid_ids, None, None

    try:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, size_of_aruco_m, cam_mat, dist_mat)
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
    """
    1) Try to find green tops (Hough / connected components)
    2) Expand to full fruit bbox
    3) Fallback to grey-body detection
    4) Filter by tray region and low CIELab chroma (reject purple)
    Returns list of contours (rect-like arrays or real contours)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    img_h, img_w = image.shape[:2]

    # green-top mask
    lower_green = np.array([40, 30, 50], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    detected_rects = []

    # Hough circles for green tops
    try:
        mask_blur = cv2.GaussianBlur(mask, (7, 7), 0)
        circles = cv2.HoughCircles(mask_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
                                   param1=50, param2=12, minRadius=5, maxRadius=70)
    except Exception:
        circles = None

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (cx, cy, r) in circles:
            x_full = max(0, cx - int(1.2 * r))
            y_full = max(0, cy - int(0.8 * r))
            w_full = min(img_w - x_full, int(2.4 * r))
            h_full = min(img_h - y_full, int(2.6 * r))
            rect = np.array([[[x_full, y_full]], [[x_full + w_full, y_full]],
                             [[x_full + w_full, y_full + h_full]], [[x_full, y_full + h_full]]])
            detected_rects.append(rect)

    # if none found, use green contours
    if len(detected_rects) == 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 40:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x_full = max(0, x - int(0.5 * w))
            w_full = min(img_w - x_full, int(2.0 * w))
            y_full = max(0, y - int(0.2 * h))
            h_full = min(img_h - y_full, int(2.0 * h))
            rect = np.array([[[x_full, y_full]], [[x_full + w_full, y_full]],
                             [[x_full + w_full, y_full + h_full]], [[x_full, y_full + h_full]]])
            detected_rects.append(rect)

    # fallback to grey-body detection if none found
    if len(detected_rects) == 0:
        lower_grey_dark = np.array([0, 0, 18], dtype=np.uint8)
        upper_grey_dark = np.array([45, 65, 130], dtype=np.uint8)
        lower_grey_light = np.array([0, 0, 70], dtype=np.uint8)
        upper_grey_light = np.array([45, 70, 210], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_grey_dark, upper_grey_dark)
        mask2 = cv2.inRange(hsv, lower_grey_light, upper_grey_light)
        mask_grey = cv2.bitwise_or(mask1, mask2)
        kernel2 = np.ones((5, 5), np.uint8)
        mask_grey = cv2.morphologyEx(mask_grey, cv2.MORPH_OPEN, kernel2)
        mask_grey = cv2.morphologyEx(mask_grey, cv2.MORPH_CLOSE, kernel2, iterations=2)
        contours, _ = cv2.findContours(mask_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 70:
                detected_rects.append(cnt)

    # filter by tray area and chroma
    filtered = []
    for rect in detected_rects:
        x, y, w, h = cv2.boundingRect(rect)
        inter_x1 = max(x, TRAY_X1); inter_y1 = max(y, TRAY_Y1)
        inter_x2 = min(x + w, TRAY_X2); inter_y2 = min(y + h, TRAY_Y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter_area < 0.18 * (w * h):
            continue
        # chroma check on Lab
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(img_w, x + w); y1 = min(img_h, y + h)
        patch = lab[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        a = patch[:, :, 1].astype(np.float32) - 128.0
        b = patch[:, :, 2].astype(np.float32) - 128.0
        chroma = np.sqrt(a * a + b * b)
        mean_chroma = float(np.nanmean(chroma))
        if mean_chroma > CHROMA_THRESH:
            # colored -> likely purple good fruit; reject
            continue
        if cv2.contourArea(rect) < 50:
            continue
        filtered.append(rect)

    return filtered


def refine_with_green_top(hsv, x, y, w, h, depth_img=None):
    """
    Refine bbox using green top detection (Hough + contour fallback).
    Returns expanded bbox and top centroid.
    """
    img_h, img_w = hsv.shape[:2]
    SEARCH_UP_FACTOR = 0.9
    roi_top_y = max(0, int(y - SEARCH_UP_FACTOR * h))
    roi_bottom_y = min(img_h, y + h)
    roi_left_x = max(0, int(x - 0.25 * w))
    roi_right_x = min(img_w, int(x + w + 0.25 * w))
    roi = hsv[roi_top_y:roi_bottom_y, roi_left_x:roi_right_x]
    if roi.size == 0:
        return x, y, w, h, x + w // 2, y

    lower_green = np.array([40, 35, 55], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(roi, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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
        return x, y, w, h, x + w // 2, y

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

    return x_full, y_full, w_full, h_full, gx, gy


# ----------------------------- MAIN CLASS ----------------------------------

class aruco_tf(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')

        # subs
        self.color_cam_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)

        self.bridge = CvBridge()
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.12, self.process_image)
        self.cv_image = None
        self.depth_image = None
        self.team_id = 3577
        self.smoother = SmoothedPublisher(alpha=SMOOTH_ALPHA)

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

        # intrinsics & centers (keep your values)
        centerCamX, centerCamY = 640.0, 360.0
        focalX, focalY = 931.1829833984375, 931.1829833984375

        CAMERA_OFFSET_X = 0.11
        CAMERA_OFFSET_Y = -0.01999
        CAMERA_OFFSET_Z = 1.452
        CAMERAA_OFFSET_X = -1.118
        CAMERAA_OFFSET_Y = -0.08
        CAMERAA_OFFSET_Z = 0.26

        img = self.cv_image.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        c_list, d_list, a_list, w_list, ids, rvecs, tvecs = detect_aruco(img)
        bad_fruit_contours = detect_bad_fruits(img)

        # ---------------------- Bad fruits TF ----------------------
        bad_fruit_id = 1

        for cnt in bad_fruit_contours:
            if bad_fruit_id > MAX_BAD_FRUITS:
                break

            x_body, y_body, w_body, h_body = cv2.boundingRect(cnt)

            # refine using green top
            x_full, y_full, w_full, h_full, cx_top, cy_top = refine_with_green_top(hsv, x_body, y_body, w_body, h_body, depth_img=self.depth_image)

            # depth: try median in full bbox, fallback to small window near top, then fallback value
            depth_value = None
            if self.depth_image is not None:
                depth_value = median_depth_in_bbox(self.depth_image, x_full, y_full, w_full, h_full)
                if depth_value is None:
                    depth_value = median_depth_in_bbox(self.depth_image, max(0, cx_top - 6), max(0, cy_top - 6), 12, 12)
            if depth_value is None:
                depth_value = 0.55

            # back-project top-centre
            X_cam = (float(cx_top) - centerCamX) * depth_value / focalX
            Y_cam = -(float(cy_top) - centerCamY) * depth_value / focalY
            Z_cam = depth_value

            base_x = CAMERA_OFFSET_X + Y_cam
            base_y = -(CAMERA_OFFSET_Y + X_cam)
            base_z = CAMERA_OFFSET_Z - Z_cam

            # orientation: point down
            quat_down = R.from_euler('xyz', [math.pi, 0.0, 0.0]).as_quat()

            # smoothing
            key = f"{self.team_id}_bad_fruit_{bad_fruit_id}"
            (sm_pos, sm_quat) = self.smoother.update(key, (base_x, base_y, base_z), quat_down)

            t_bad = TransformStamped()
            t_bad.header.stamp = self.get_clock().now().to_msg()
            t_bad.header.frame_id = 'base_link'
            t_bad.child_frame_id = key
            t_bad.transform.translation.x = float(sm_pos[0])
            t_bad.transform.translation.y = float(sm_pos[1])
            t_bad.transform.translation.z = float(sm_pos[2])
            t_bad.transform.rotation.x = float(sm_quat[0])
            t_bad.transform.rotation.y = float(sm_quat[1])
            t_bad.transform.rotation.z = float(sm_quat[2])
            t_bad.transform.rotation.w = float(sm_quat[3])
            self.br.sendTransform(t_bad)

            # annotate for debugging
            cv2.rectangle(img, (int(x_full), int(y_full)), (int(x_full + w_full), int(y_full + h_full)), (0, 255, 0), 2)
            cv2.circle(img, (int(cx_top), int(cy_top)), 6, (0, 255, 0), -1)
            cv2.putText(img, f"bad_fruit_{bad_fruit_id}", (int(x_full), int(y_full) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            bad_fruit_id += 1

        # ---------------------- ArUco TFs (unchanged orientation logic) ----------------------
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
                    base_x += -0.06

                try:
                    rvec = rvecs[idx].reshape(3)
                    rmat_cv, _ = cv2.Rodrigues(rvec)
                    cv_to_ros = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                    side_pick_rotation = R.from_euler('xyz', [110, 90, 150], degrees=True).as_matrix()
                    rmat_ros = cv_to_ros @ rmat_cv
                    if marker_id == 3:
                        rmat_final = rmat_ros @ side_pick_rotation
                    else:
                        rot_x_140 = R.from_euler('x', 140, degrees=True).as_matrix()
                        rmat_final = rmat_ros @ rot_x_140
                    quat_marker = R.from_matrix(rmat_final).as_quat()
                except Exception as e:
                    self.get_logger().warn(f"ArUco orientation conversion failed for id {marker_id}: {e}")
                    quat_marker = R.from_euler('xyz', [0, 0, 0]).as_quat()

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

                # DROP TF for marker 6
                try:
                    if int(marker_id) == DROP_MARKER_ID:
                        t_drop = TransformStamped()
                        t_drop.header.stamp = t_obj.header.stamp
                        t_drop.header.frame_id = t_obj.header.frame_id
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