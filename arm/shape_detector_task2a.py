#!/usr/bin/env python3
'''
# Team ID:          3577
# Theme:            Krishi coBot
# Author List:      KARTHIK, ANUDEEP, JOSH VISWANADH, MANIKANTA
# Filename:         shape_detector_task2a.py
# Functions:        convex_hull, polygon_angles, __init__, odom_cb, scan_cb, main
# Global variables: None
'''

"""
Robust Shape Detector (Triangle / Square) with Strong De-Duplication
Now with: STOP ROBOT 0.1m AFTER TRIANGLE DETECTION

- Triangle  →  FERTILIZER_REQUIRED  → triggers stop-after-distance
- Square    →  BAD_HEALTH
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import numpy as np
import math
import time
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def convex_hull(points):
    '''
    Purpose:
    ---
    Compute the convex hull of a set of 2D points using Monotone Chain Algorithm.

    Input Arguments:
    ---
    `points` : [list of (float, float)]
        List of x,y points extracted from Lidar scan.

    Returns:
    ---
    `hull` : [list of (float, float)]
        Ordered set of vertices forming the convex hull polygon.

    Example call:
    ---
    hull = convex_hull([(0.1,0.2),(0.2,0.1),(0.3,0.3)])
    '''

    pts = np.array(points)
    if pts.shape[0] <= 1:
        return pts.tolist()

    # Sort by x then y
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]

    def cross(o, a, b):
        # Cross-product used for left/right turn check
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    # Build upper hull
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    return lower[:-1] + upper[:-1]


def polygon_angles(vertices):
    '''
    Purpose:
    ---
    Compute internal angles of a polygon formed by given vertices.

    Input Arguments:
    ---
    `vertices` : [list of (float, float)]
        Polygon vertices.

    Returns:
    ---
    `angles` : [list of float]
        Internal polygon angles in degrees.

    Example call:
    ---
    polygon_angles([(0,0),(1,0),(0.5,1)])
    '''
    pts = np.array(vertices)
    n = len(pts)
    angles = []

    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p = pts[i]
        p_next = pts[(i + 1) % n]

        v1 = p_prev - p
        v2 = p_next - p
        num = np.dot(v1, v2)
        den = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        ang = math.degrees(math.acos(np.clip(num/den, -1.0, 1.0)))
        angles.append(ang)

    return angles



# ---------------------------------------------------------------------
# Shape Detector Node
# ---------------------------------------------------------------------

class ShapeDetectorFrontView(Node):

    def __init__(self):
        '''
        Purpose:
        ---
        Initialize Lidar-based shape-detection node with:
        - spatial filtering
        - polygon analysis
        - edge-counting fallback
        - strong de-duplication
        - STOP robot 0.1m after triangle detection

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        '''
        super().__init__('shape_detector_task2a_frontview_poly')

        # Subscribers
        self.sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 20)

        # Publishers
        self.pub = self.create_publisher(String, '/detection_status', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- Robot pose (filled by odom_cb) ---
        self.robot_x = 0.0     # Current x position
        self.robot_y = 0.0     # Current y position
        self.robot_yaw = 0.0   # Current yaw angle

        # --- STOP-after-Triangle logic ---
        self.stop_active = False
        self.stop_reference = (0.0, 0.0)    # Position where triangle was detected
        self.stop_distance = 0.10           # Stop after moving 0.1m

        # --- Lidar filtering parameters ---
        self.min_range = 0.20
        self.max_range = 2.50
        self.angle_window = math.radians(60)

        # Region in front of the robot
        self.forward_min_x = 0.15
        self.forward_max_x = 1.40
        self.lateral_half_width = 0.60

        # Shape width boundaries
        self.min_shape_width = 0.10
        self.max_shape_width = 0.70

        # Processing throttling
        self.process_period = 0.12
        self.last_proc_time = 0.0

        # Detection smoothing
        self.confirm_window = 4
        self.type_history = []

        # De-duplication
        self.last_status = None
        self.last_centroid = (0.0, 0.0)
        self.min_change_dist = 0.3
        self.min_change_time = 3.0
        self.last_pub_time = 0.0
        self.detected_spots = []
        self.spot_merge_radius = 0.40

        # Edge smoothing
        self.edge_smooth = []
        self.dist_thresh = 0.06
        self.smooth_window = 5

        self.get_logger().info("✅ ShapeDetectorFrontView ready (with stop-after-triangle).")

    # -----------------------------------------------------------------
    def odom_cb(self, msg):
        '''
        Purpose:
        ---
        Update robot's odometry (x, y, yaw) from /odom topic.

        Input Arguments:
        ---
        `msg` : [nav_msgs.msg.Odometry]
            Contains robot pose information.

        Returns:
        ---
        None

        Example call:
        ---
        odom_cb(Odometry())
        '''
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Store pose
        self.robot_x = float(p.x)
        self.robot_y = float(p.y)
        self.robot_yaw = yaw

    # -----------------------------------------------------------------
    def scan_cb(self, msg):
        '''
        Purpose:
        ---
        Main Lidar-based shape processing:
        - filter points
        - compute width, depth variance
        - edge-counting heuristic
        - convex-hull polygon analysis
        - classification into triangle/square
        - strong de-duplication
        - publish detection + robot coordinates
        - trigger STOP-0.1m routine

        Input Arguments:
        ---
        `msg` : [sensor_msgs.msg.LaserScan]
            Lidar scan data.

        Returns:
        ---
        None
        '''
        now = time.time()
        if now - self.last_proc_time < self.process_period:
            return
        self.last_proc_time = now

        # ---------------- BASIC VALID RANGE FILTER ----------------
        ranges = np.array(msg.ranges)
        mask = np.isfinite(ranges) & (ranges > self.min_range) & (ranges < self.max_range)
        if not np.any(mask):
            return

        angles_full = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        angles = angles_full[mask]
        ranges_valid = ranges[mask]

        # Compute coordinates
        xs = ranges_valid * np.cos(angles)
        ys = ranges_valid * np.sin(angles)

        # Limit points to ±60° FOV
        a_mask = np.abs(angles) < self.angle_window
        xs = xs[a_mask]; ys = ys[a_mask]
        if len(xs) < 10:
            return

        pts_all = np.vstack((xs, ys)).T

        # ---------------- FORWARD SPATIAL FILTER ----------------
        fmask = (
            (pts_all[:,0] > self.forward_min_x) &
            (pts_all[:,0] < self.forward_max_x) &
            (np.abs(pts_all[:,1]) < self.lateral_half_width)
        )
        pts = pts_all[fmask]
        if len(pts) < 10:
            return

        xs_f = pts[:,0]
        ys_f = pts[:,1]

        # ---------------- SHAPE WIDTH / DEPTH ESTIMATION ----------------
        width = float(np.max(xs_f) - np.min(xs_f))
        depth_var = float(np.std(ys_f))
        if width < self.min_shape_width or width > self.max_shape_width:
            return

        flat_ratio = width / (depth_var + 1e-3)

        # ---------------- EDGE COUNTING HEURISTIC ----------------
        dists = np.sqrt(np.diff(xs_f)**2 + np.diff(ys_f)**2)
        edge_count = int(np.sum(dists > self.dist_thresh))

        self.edge_smooth.append(edge_count)
        if len(self.edge_smooth) > self.smooth_window:
            self.edge_smooth.pop(0)
        smooth_edges = round(np.mean(self.edge_smooth))

        edge_type = None
        if (smooth_edges >= 2) and (depth_var < 0.04) and (width > 0.25) and (flat_ratio > 8):
            edge_type = "FERTILIZER_REQUIRED"

        # ---------------- POLYGON ANALYSIS ----------------
        centroid_tmp = np.mean(pts, axis=0)
        angles_pts = np.arctan2(pts[:,1] - centroid_tmp[1], pts[:,0] - centroid_tmp[0])
        order = np.argsort(angles_pts)
        hull = convex_hull(pts[order])

        poly_type = None
        n_corners = 0  # kept for legacy, not used actively

        # ---------------- CLASSIFICATION RULES ----------------
        if (width > 0.60) and (depth_var < 0.02) and (flat_ratio > 30):
            poly_type = "FERTILIZER_REQUIRED"

        elif (0.48 < width < 0.60) and (depth_var < 0.02) and (flat_ratio > 20):
            poly_type = "BAD_HEALTH"

        elif (0.15 < width < 0.30) and (depth_var > 0.30) and (flat_ratio < 2.0) and (n_corners >= 3):
            poly_type = "BAD_HEALTH"

        else:
            return

        # Final choice
        final_type = poly_type if poly_type is not None else edge_type
        if final_type is None:
            return

        # ---------------- TEMPORAL STABILITY CHECK ----------------
        self.type_history.append(final_type)
        if len(self.type_history) > self.confirm_window:
            self.type_history.pop(0)
        if self.type_history.count(final_type) < self.confirm_window - 1:
            return

        # ---------------- REMOVE DUPLICATES ----------------
        cx_use = float(np.mean(xs_f))
        cy_use = float(np.mean(ys_f))

        # Against previously detected regions
        for (px, py) in self.detected_spots:
            if math.hypot(cx_use - px, cy_use - py) < self.spot_merge_radius:
                return

        moved = math.hypot(cx_use - self.last_centroid[0], cy_use - self.last_centroid[1])
        if (
            self.last_status == final_type and
            moved < self.min_change_dist and
            (now - self.last_pub_time) < self.min_change_time
        ):
            return

        # ---------------- PUBLISH DETECTION ----------------
        self.last_status = final_type
        self.last_centroid = (cx_use, cy_use)
        self.last_pub_time = now

        msg_out = String()
        msg_out.data = f"{final_type},{self.robot_x:.3f},{self.robot_y:.3f}"
        self.pub.publish(msg_out)

        self.detected_spots.append((cx_use, cy_use))
        if len(self.detected_spots) > 10:
            self.detected_spots.pop(0)

        # ---------------- START STOP-AFTER-0.1m ROUTINE ----------------
        if final_type == "FERTILIZER_REQUIRED":
            self.stop_active = True
            self.stop_reference = (self.robot_x, self.robot_y)

        self.get_logger().info(
            f"📍 {final_type} at ({cx_use:.3f},{cy_use:.3f}) "
            f"width={width:.3f} depthVar={depth_var:.3f} flatRatio={flat_ratio:.1f}"
        )

        # ----------------------------------------------------------
        # STOP ROBOT AFTER MOVING 0.1m
        # ----------------------------------------------------------
        if self.stop_active:
            dist = math.hypot(
                self.robot_x - self.stop_reference[0],
                self.robot_y - self.stop_reference[1]
            )

            # Stop if moved 0.1m after triangle
            if dist >= self.stop_distance:
                stop_cmd = Twist()
                stop_cmd.linear.x = 0.0
                stop_cmd.angular.z = 0.0
                self.cmd_pub.publish(stop_cmd)

                self.get_logger().warn(
                    f"🛑 Robot STOPPED after moving {dist:.3f} m from triangle detection."
                )

                self.stop_active = False


# ---------------------------------------------------------------------
def main(args=None):
    '''
    Purpose:
    ---
    Entry point for starting the shape detector node.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    main()
    '''
    rclpy.init(args=args)
    node = ShapeDetectorFrontView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
