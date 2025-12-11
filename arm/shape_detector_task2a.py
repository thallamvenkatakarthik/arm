#!/usr/bin/env python3
"""
FINAL WORKING VERSION — Shape Detection with RViz Markers (centroid + span fixes)

Fixes:
 - publish_marker now uses 'odom' frame so markers placed at world (wx,wy) centroid.
 - added cluster angular-span check to avoid classifying tiny sweeps/partial scans.
 - small defensive checks for centroid math.
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


def rdp(points: List[Point], eps: float) -> List[Point]:
    """Ramer–Douglas–Peucker simplification."""
    if len(points) < 3:
        return points

    def perp_dist(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        num = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax)
        den = math.hypot(bx - ax, by - ay)
        return num / den if den > 1e-6 else 0.0

    first, last = 0, len(points) - 1
    max_d = -1.0
    idx = -1
    for i in range(first + 1, last):
        d = perp_dist(points[i], points[first], points[last])
        if d > max_d:
            max_d = d
            idx = i

    if max_d > eps:
        left = rdp(points[: idx + 1], eps)
        right = rdp(points[idx:], eps)
        return left[:-1] + right
    else:
        return [points[first], points[last]]


class ShapeDetector(Node):
    def __init__(self):
        super().__init__("shape_detector_task2a")

        # ROS: subscribers
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        # ROS: publishers
        self.det_pub = self.create_publisher(String, "/detection_status", 10)
        self.marker_pub = self.create_publisher(Marker, "/shape_markers", 10)

        # Store latest data
        self.last_scan = None
        self.pose = None  # (x, y, yaw)

        # Timer - 10 Hz
        self.timer = self.create_timer(0.1, self.loop)

        # Detection debounce
        self.last_detection_time = 0.0
        self.cooldown = 2.5        # seconds
        self.last_detection_pos = None
        self.same_pos_thresh = 0.18

        # Clustering parameters
        self.max_range = 1.8
        self.cluster_dist = 0.16
        self.min_cluster_pts = 10

        # cluster angular span threshold (radians) — helps avoid classifying thin sweeps
        self.min_cluster_span_angle = 0.06

        # Marker ID counter
        self.marker_id = 0

        self.get_logger().info("✅ FINAL Shape Detector + RViz markers started (centroid + span fixes).")

    # ----------------------------------------------------------------------
    # ROS CALLBACKS
    # ----------------------------------------------------------------------

    def scan_cb(self, msg):
        self.last_scan = msg

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose = (p.x, p.y, yaw)

    # ----------------------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------------------

    def loop(self):
        if self.last_scan is None or self.pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_detection_time < self.cooldown:
            return

        pts = self.scan_to_points(self.last_scan)
        clusters = self.cluster_points(pts)

        for cl in clusters:
            # reject tiny angular-span clusters (likely single-beam or sweeping artifact)
            if self._cluster_span_angle(cl) < self.min_cluster_span_angle:
                continue

            out = self.classify_cluster(cl)
            if out is None:
                continue

            status, centroid_robot = out
            wx, wy = self.robot_to_world(centroid_robot)

            # avoid duplicates near same location
            if self.last_detection_pos is not None:
                if euclid((wx, wy), self.last_detection_pos) < self.same_pos_thresh:
                    continue

            # publish detection (centroid in world coords)
            msg = String()
            msg.data = f"{status},{wx:.2f},{wy:.2f}"
            self.det_pub.publish(msg)

            self.last_detection_time = now
            self.last_detection_pos = (wx, wy)

            # publish marker at centroid in odom frame (world coords)
            self.publish_marker(wx, wy, status)

            self.get_logger().info(f"📡 DETECTED {status} at ({wx:.2f},{wy:.2f}) (centroid)")

            break  # one detection per cycle allowed

    # ----------------------------------------------------------------------
    # POINT PROCESSING
    # ----------------------------------------------------------------------

    def scan_to_points(self, scan: LaserScan):
        pts = []
        angle = scan.angle_min

        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < min(scan.range_max, self.max_range):
                pts.append((r * math.cos(angle), r * math.sin(angle)))
            angle += scan.angle_increment

        return pts

    def _cluster_span_angle(self, cluster_pts: List[Point]) -> float:
        """Estimate angular span of cluster using endpoints (atan2)."""
        if len(cluster_pts) < 2:
            return 0.0
        a1 = math.atan2(cluster_pts[0][1], cluster_pts[0][0])
        a2 = math.atan2(cluster_pts[-1][1], cluster_pts[-1][0])
        d = abs(a2 - a1)
        # wrap to [-pi, pi]
        while d > math.pi:
            d -= 2 * math.pi
        return abs(d)

    def cluster_points(self, pts):
        clusters = []
        if not pts:
            return clusters

        cur = [pts[0]]
        for i in range(1, len(pts)):
            if euclid(pts[i], pts[i - 1]) < self.cluster_dist:
                cur.append(pts[i])
            else:
                if len(cur) >= self.min_cluster_pts:
                    clusters.append(cur)
                cur = [pts[i]]

        if len(cur) >= self.min_cluster_pts:
            clusters.append(cur)

        return clusters

    # ----------------------------------------------------------------------
    # SHAPE CLASSIFICATION
    # ----------------------------------------------------------------------

    def polygon_area(self, poly: List[Point]) -> float:
        area = 0.0
        for i in range(len(poly) - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def classify_cluster(self, cl):
        # RDP simplify
        span = euclid(cl[0], cl[-1])
        eps = max(0.02, span * 0.08)

        poly = rdp(cl, eps)
        if len(poly) < 3:
            return None

        # close polygon
        if euclid(poly[0], poly[-1]) > eps * 2:
            poly.append(poly[0])

        n_edges = len(poly) - 1
        if n_edges not in (3, 4, 5):
            return None

        # area filter (kills fake walls)
        area = self.polygon_area(poly)
        if not (0.015 < area < 0.40):
            self.get_logger().debug(f"Rejected polygon: area {area:.4f}")
            return None

        # edges
        edges = [euclid(poly[i], poly[i + 1]) for i in range(n_edges)]
        mean_len = sum(edges) / n_edges

        if not (0.09 < mean_len < 0.60):
            self.get_logger().debug(f"Rejected polygon: mean edge {mean_len:.4f}")
            return None

        # compactness filter
        perimeter = sum(edges)
        comp = (perimeter * perimeter) / (4 * math.pi * area)
        if comp > 4.2:
            self.get_logger().debug(f"Rejected polygon: compactness {comp:.4f}")
            return None

        # classification
        if n_edges == 3:
            status = "FERTILIZER_REQUIRED"
        elif n_edges in (4, 5):
            status = "BAD_HEALTH"
        else:
            return None

        # centroid (polygon centroid)
        cx = cy = 0.0
        A2 = 0.0  # 2*A
        for i in range(len(poly) - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]
            cross = x1 * y2 - x2 * y1
            A2 += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross

        if abs(A2) < 1e-8:
            return None

        # Proper centroid formula: Cx = (1/(6A)) * sum((x1+x2)*cross)
        # A2 = 2*A -> denominator 6A == 3*A2
        cx = cx / (3.0 * A2)
        cy = cy / (3.0 * A2)

        # centroid is returned in laser/robot frame (same as points)
        return status, (cx + 0.5, cy +0.5)

    # ----------------------------------------------------------------------
    # WORLD TRANSFORM + MARKERS
    # ----------------------------------------------------------------------

    def robot_to_world(self, pt):
        rx, ry, yaw = self.pose
        x_r, y_r = pt
        c, s = math.cos(yaw), math.sin(yaw)
        return (
            rx + c * x_r - s * y_r,
            ry + s * x_r + c * y_r,
        )

    def publish_marker(self, x, y, status):
        m = Marker()
        # **IMPORTANT**: publish marker in odom frame because x,y are world coords
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
            m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0  # Yellow
        else:
            m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0  # Red

        m.color.a = 1.0

        m.lifetime.sec = 20

        self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
