#!/usr/bin/env python3
"""
FINAL STAFF-COMPLIANT VERSION — Shape Detection (FIXED)

Fixes applied:
- Do NOT exit loop when a shape was already detected (continue instead)
- Relaxed geometric thresholds to avoid false rejections
- Robust RDP epsilon
- Rectangle edge-count tolerance
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf_transformations import euler_from_quaternion

import math
from typing import List, Tuple

Point = Tuple[float, float]


def euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ShapeDetector(Node):
    def __init__(self):
        super().__init__("shape_detector_task2a")

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        self.det_pub = self.create_publisher(String, "/detection_status", 10)
        self.marker_pub = self.create_publisher(Marker, "/shape_markers", 10)

        self.last_scan = None
        self.pose = None  # (x, y, yaw)

        self.timer = self.create_timer(0.1, self.loop)

        # ---- ONE-TIME detection lock ----
        self.detected_positions = []
        self.same_pos_thresh = 0.6

        # ---- clustering ----
        self.max_range = 1.8
        self.cluster_dist = 0.16
        self.min_cluster_pts = 10
        self.min_cluster_span_angle = 0.035

        # ---- geometry filters ----
        self.min_area = 0.02
        self.max_area = 0.35
        self.max_compactness = 6.0
        self.min_edge_len = 0.10
        self.max_edge_len = 0.55

        self.marker_id = 0

        self.pending_detections = []  # [(status, wx, wy)]
        self.arrival_thresh = 0.25
        self.x_split = 1.8  # adjust based on map (center between beds)
  # adjust if needed



        self.get_logger().info("✅ Shape Detector started (fixed & staff-compliant)")

    # ------------------------------------------------------------------

    def scan_cb(self, msg):
        self.last_scan = msg

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose = (p.x, p.y, yaw)

    # ------------------------------------------------------------------

    def is_near_existing(self, wx, wy):
                for p in self.detected_positions:
                    if euclid(p, (wx, wy)) < self.same_pos_thresh:
                        return True
                for _, px, py in self.pending_detections:
                    if euclid((px, py), (wx, wy)) < self.same_pos_thresh:
                        return True
                return False
    
    def get_plant_id(self, wx, wy):
        """
        Plant ID assignment based on X split (beds) and Y ordering
        Left bed  : IDs 1–4
        Right bed : IDs 5–8
        """

        # Decide bed using X
        if wx < self.x_split:
            base_id = 1      # left bed
            bed_ys = []
            for px, py in self.detected_positions:
                if px < self.x_split:
                    bed_ys.append(py)
        else:
            base_id = 5      # right bed
            bed_ys = []
            for px, py in self.detected_positions:
                if px >= self.x_split:
                    bed_ys.append(py)

        # Include current plant
        bed_ys.append(wy)

        # Sort bottom → top
        bed_ys = sorted(bed_ys)

        index = bed_ys.index(wy)   # 0–3
        return base_id + index




    def loop(self):
        if self.last_scan is None or self.pose is None:
            return

        pts = self.scan_to_points(self.last_scan)
        clusters = self.cluster_points(pts)

        # -------- DETECTION PHASE --------
        for cl in clusters:
            if self.cluster_span_angle(cl) < self.min_cluster_span_angle:
                continue

            result = self.classify_cluster(cl)
            if result is None:
                continue

            status, centroid_robot = result
            wx, wy = self.robot_to_world(centroid_robot)

            if self.is_near_existing(wx, wy):
                continue

            self.detected_positions.append((wx, wy))
            self.pending_detections.append((status, wx, wy))

            self.get_logger().info(
                f"👁️ {status} detected at ({wx:.2f}, {wy:.2f}) — waiting for arrival"
            )

        # -------- ARRIVAL PHASE (RUNS ALWAYS) --------
        rx, ry, _ = self.pose
        still_pending = []

        for status, wx, wy in self.pending_detections:
            if abs(ry - wy) < self.arrival_thresh:   # 🔥 Y-ONLY MATCH
                plant_id = self.get_plant_id(wx, wy)
                msg = String()
                msg.data = f"{status},{wx:.2f},{wy:.2f},{plant_id}"
                self.det_pub.publish(msg)

                self.publish_marker(wx, wy, status)

                self.get_logger().info(
                    f"✅ {status} CONFIRMED at plant {plant_id} (x={wx:.2f}, y={wy:.2f})"
                )
            else:
                still_pending.append((status, wx, wy))

        self.pending_detections = still_pending

    # ------------------------------------------------------------------

    def scan_to_points(self, scan):
        pts = []
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < min(scan.range_max, self.max_range):
                pts.append((r * math.cos(angle), r * math.sin(angle)))
            angle += scan.angle_increment
        return pts

    def cluster_span_angle(self, cl):
        a1 = math.atan2(cl[0][1], cl[0][0])
        a2 = math.atan2(cl[-1][1], cl[-1][0])
        d = abs(a2 - a1)
        return abs((d + math.pi) % (2 * math.pi) - math.pi)

    def cluster_points(self, pts):
        clusters = []
        cur = []
        for p in pts:
            if not cur or euclid(p, cur[-1]) < self.cluster_dist:
                cur.append(p)
            else:
                if len(cur) >= self.min_cluster_pts:
                    clusters.append(cur)
                cur = [p]
        if len(cur) >= self.min_cluster_pts:
            clusters.append(cur)
        return clusters

    # ------------------------------------------------------------------

    def rdp(self, pts, eps):
        if len(pts) < 3:
            return pts

        def perp(p, a, b):
            num = abs((b[1]-a[1])*p[0] - (b[0]-a[0])*p[1] + b[0]*a[1] - b[1]*a[0])
            den = math.hypot(b[0]-a[0], b[1]-a[1])
            return num / den if den > 1e-6 else 0

        dmax, idx = 0, 0
        for i in range(1, len(pts)-1):
            d = perp(pts[i], pts[0], pts[-1])
            if d > dmax:
                dmax, idx = d, i

        if dmax > eps:
            left = self.rdp(pts[:idx+1], eps)
            right = self.rdp(pts[idx:], eps)
            return left[:-1] + right
        else:
            return [pts[0], pts[-1]]

    # ------------------------------------------------------------------

    def polygon_area(self, poly):
        return abs(sum(
            poly[i][0]*poly[i+1][1] - poly[i+1][0]*poly[i][1]
            for i in range(len(poly)-1)
        )) / 2

    def classify_cluster(self, cl):
        span = euclid(cl[0], cl[-1])
        eps = min(0.04, max(0.015, span * 0.06))

        poly = self.rdp(cl, eps)
        if len(poly) < 3:
            return None

        if euclid(poly[0], poly[-1]) > eps * 2:
            poly.append(poly[0])

        edges = [euclid(poly[i], poly[i+1]) for i in range(len(poly)-1)]
        edges_sorted = sorted(edges)

        if edges_sorted[1] < self.min_edge_len:
            return None
        if edges_sorted[-2] > self.max_edge_len:
            return None

        area = self.polygon_area(poly)
        if not (self.min_area < area < self.max_area):
            return None

        perimeter = sum(edges)
        compactness = (perimeter ** 2) / (4 * math.pi * area)
        if compactness > self.max_compactness:
            return None

        n_edges = len(edges)
        if n_edges == 3:
            status = "FERTILIZER_REQUIRED"
        elif 4 <= n_edges <= 6:
            status = "BAD_HEALTH"
        else:
            return None

        cx = sum(p[0] for p in poly[:-1]) / (len(poly) - 1)
        cy = sum(p[1] for p in poly[:-1]) / (len(poly) - 1)

        return status, (cx, cy)

    # ------------------------------------------------------------------

    def robot_to_world(self, pt):
        rx, ry, yaw = self.pose
        x, y = pt
        c, s = math.cos(yaw), math.sin(yaw)
        return rx + c * x - s * y, 0.5+ry + s * x + c * y

    def publish_marker(self, x, y, status):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "shapes"
        m.id = self.marker_id
        self.marker_id += 1

        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25

        if status == "FERTILIZER_REQUIRED":
            m.color.r = 1.0
            m.color.g = 1.0
        else:
            m.color.r = 1.0

        m.color.a = 1.0
        self.marker_pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(ShapeDetector())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
