#!/usr/bin/env python3
"""
ebot_spec_mission.py (final)

Behavior:
1. Start at origin (P3 = [-1.53, -6.61, -1.57])
2. Rotate 80° right (-80°)
3. Move forward until obstacle detected
4. Rotate 60° left (+60°)
5. Move to P1
6. After reaching P1, move forward until obstacle detected
7. Rotate 90° left, then move to P2 using grid-aligned axis logic (updated here)
8. From P2, go to P3 (origin)
9. AFTER P3: turn right 90°, move until obstacle, turn left 90°, move until obstacle, then STOP
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion
from std_msgs.msg import String
import math
import time


def deg2rad(d): return d * math.pi / 180.0


class EbotSpecMission(Node):
    def __init__(self):
        super().__init__('ebot_spec_mission')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 20)
        self.pub = self.create_publisher(String, '/detection_status', 10)
        self.status_sub = self.create_subscription(String, '/detection_status', self.status_cb, 10)
        self.fertilizer_detached = True
        self.detach_sub = self.create_subscription(String,'/fertilizer_detach_status',self.detach_cb,10)

        self.global_pause = False
        self.global_pause_start = None
        self.global_pause_duration = 2.0


        self.P3 = (-1.53, -5.55, 1.57)
        self.P1 = (0.26, -1.95, 1.57)
        self.P2 = (-1.48, -0.67, -1.57)
        self.P_INT = (0.13, 1.5, 1.57)


        self.pose = None
        self.front = 10.0
        self.state = 'INIT_ROTATE'
        self.target_yaw = None

        self.linear_speed = 2.0
        self.angular_speed = 1.0
        self.obstacle_threshold = 0.72
        self.emergency_stop = 0.5
        self.dist_tolerance = 0.15
        self.yaw_tolerance = deg2rad(5)

        self.init_turn = -deg2rad(70.0)
        self.correction_turn = deg2rad(90.0)
        self.after_p1_left_turn = deg2rad(90.0)

        # NEW: turns to use after P3
        self.after_p3_right_turn = -deg2rad(90.0)  # turn right 90°
        self.after_p3_left_turn = -deg2rad(110.0)    # then left 90°

        # self.pause_timer = None
        self.pause_start_time = None
        self.pause_duration = 2.0  # seconds
        self.fertilizer_pause_duration = 7.0
        self.create_timer(0.1, self._timer_cb)
        self.get_logger().info("✅ eBot Spec Mission initialized (with axis alignment navigation to P2).")

    # ---------------- TIMER ----------------
    def _timer_cb(self):
        if self.state == 'STOP':
            self.cmd_pub.publish(Twist())

    def detach_cb(self, msg):
        if msg.data.strip() == "fertilizer_detached":
            self.fertilizer_detached = True
            self.get_logger().info("🧲 Fertilizer detachment received — bot will resume when 2 sec is complete.")

    # ---------------- ODOM ----------------
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose = (p.x, p.y, yaw)
        self._run_state()

    # ---------------- SCAN ----------------
    def scan_cb(self, msg):
        n = len(msg.ranges)
        if n == 0:
            return
        mid = n // 2
        sec = max(1, n // 72)
        values = [r for r in msg.ranges[mid-sec:mid+sec] if r and r > 0.0]
        self.front = min(values) if values else 10.0
        self._run_state()


    # ---------------- STATUS CALLBACK (GLOBAL PAUSE + COORD PUBLISH) ----------------
    def status_cb(self, msg):
        # Ignore our own DOCK_STATION message from P1 pause
        if msg.data.startswith("DOCK_STATION"):
            return

        # Trigger pause only if not already in pause
        if not self.global_pause and self.pose is not None:
            x, y, _ = self.pose

            # Publish coordinates immediately
            coord_msg = String()
            coord_msg.data = f"GLOBAL_PAUSE,{x:.2f},{y:.2f},0"
            # self.pub.publish(coord_msg)
            self.get_logger().info(f"📡 Published global pause coordinates: {coord_msg.data}")

            # Start global pause
            self.global_pause = True
            self.global_pause_start = self.get_clock().now()
            self.get_logger().info(f"⏸ Global pause triggered by detection: {msg.data}")


    # ---------------- STATE MACHINE ----------------
    def _run_state(self):
        if self.pose is None:
            return
        # GLOBAL STOP WHEN ANY DETECTION HAPPENS
        if self.global_pause:
            # Stop robot
            stop_cmd = Twist()
            self.cmd_pub.publish(stop_cmd)

            # Time check
            current = self.get_clock().now()
            elapsed = (current - self.global_pause_start).nanoseconds / 1e9
            if elapsed >= self.global_pause_duration:
                self.global_pause = False
                self.get_logger().info("▶️ Global pause over. Resuming mission.")
            return   # <-- IMPORTANT: don't run state machine when paused
        cmd = Twist()
        x, y, yaw = self.pose

        # 1️⃣ Rotate -80° right
        if self.state == 'INIT_ROTATE':
            if self.target_yaw is None:
                self.target_yaw = self._normalize(yaw + self.init_turn)
                self.get_logger().info("Rotating 80° right...")
            yaw_err = self._normalize(self.target_yaw - yaw)
            if abs(yaw_err) > self.yaw_tolerance:
                cmd.angular.z = math.copysign(self.angular_speed, yaw_err)
            else:
                self.state = 'MOVE_UNTIL_OBSTACLE'
                self.target_yaw = None
                self.get_logger().info("✅ Initial rotation complete. Moving forward until obstacle.")

        # 2️⃣ Move straight until obstacle detected   
        elif self.state == 'MOVE_UNTIL_OBSTACLE':
            if self.front < self.emergency_stop:
                self.state = 'CORRECT_LEFT_60'
            elif self.front < self.obstacle_threshold:
                 self.state = 'CORRECT_LEFT_60'
                 self.get_logger().info("🧱 Obstacle detected. Rotating 60° left...")
            else:
                cmd.linear.x = self.linear_speed

        # 3️⃣ Rotate +60° left
        elif self.state == 'CORRECT_LEFT_60':
            if self.target_yaw is None:
                self.target_yaw = self._normalize(self.correction_turn)
                self.get_logger().info("Turning 60° left...")
            yaw_err = self._normalize(self.target_yaw - yaw)
            if abs(yaw_err) > self.yaw_tolerance:
                cmd.angular.z = math.copysign(self.angular_speed, yaw_err)
            else:
                self.state = 'MOVE_TO_P1'
                self.target_yaw = None
                self.get_logger().info("✅ Left turn complete. Navigating to P1.")

        # 4️⃣ Move to P1
        elif self.state == 'MOVE_TO_P1':
            gx, gy, _ = self.P1
            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)
            goal_yaw = math.atan2(dy, dx)
            yaw_err = self._normalize(goal_yaw - yaw)
            # if self.front < self.obstacle_threshold:
            #     cmd = Twist()
            # elif abs(yaw_err) > self.yaw_tolerance:
            #     cmd.angular.z = math.copysign(self.angular_speed, yaw_err)
            # else:
            cmd.linear.x = self.linear_speed
            if dist < self.dist_tolerance:
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                self.state = 'PAUSE_AT_P1'
                self.pause_start_time = self.get_clock().now()
                self.get_logger().info("🛑 Reached P1. Pausing for 2 seconds...")



        # 🕒 Pause for 2 seconds at P1
        elif self.state == 'PAUSE_AT_P1':
            cmd = Twist()
            self.cmd_pub.publish(cmd)  # ensure robot stays still

            # Get current odometry-based coordinates
            x, y, yaw = self.pose

            # Publish detection message immediately after stopping (only once)
            if not hasattr(self, 'p1_detection_published'):
                msg = String()
                msg.data = f"DOCK_STATION,{x:.2f},{y:.2f},0"  # use live odometry position
                self.pub.publish(msg)
                self.get_logger().info(f"📡 Published detection status: {msg.data}")
                self.p1_detection_published = True  # prevents re-sending

            # Calculate elapsed pause time
            current_time = self.get_clock().now()
            elapsed = (current_time - self.pause_start_time).nanoseconds / 1e9

            # After 2 seconds, continue mission
            if elapsed >= self.fertilizer_pause_duration and self.fertilizer_detached:
                self.state = 'MOVE_TO_INTERMEDIATE'
                self.fertilizer_detached = False 
                if hasattr(self, 'p1_detection_published'):
                    del self.p1_detection_published  # clean up flag
                self.get_logger().info(f"✅ Pause complete ({self.pause_duration}s). Continuing mission.")


        # 5️⃣ After P1 — move straight until obstacle
        elif self.state == 'AFTER_P1_MOVE_UNTIL_OBSTACLE':
            if self.front < 0.5:
                self.state = 'TURN_LEFT_90'
                self.get_logger().info("🧱 Obstacle detected after P1. Turning 90° left...")
            else:
                cmd.linear.x = self.linear_speed


        # 5️⃣ Move to INTERMEDIATE POINT (replaces move-until-obstacle logic)
        elif self.state == 'MOVE_TO_INTERMEDIATE':
            gx, gy, _ = self.P_INT
            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)

            goal_yaw = math.atan2(dy, dx)
            yaw_err = self._normalize(goal_yaw - yaw)

            # Rotate toward intermediate point
            if abs(yaw_err) > self.yaw_tolerance:
                cmd.angular.z = math.copysign(self.angular_speed, yaw_err)
            else:
                cmd.linear.x = self.linear_speed

            # Arrival check
            if dist < self.dist_tolerance:
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                self.state = 'TURN_LEFT_90'
                self.get_logger().info("✅ Reached intermediate point. Proceeding with 90° left turn.")


        # 6️⃣ Rotate +90° left
        elif self.state == 'TURN_LEFT_90':
            if self.target_yaw is None:
                self.target_yaw = self._normalize(yaw + self.after_p1_left_turn)
            yaw_err = self._normalize(self.target_yaw - yaw)
            if abs(yaw_err) > self.yaw_tolerance:
                cmd.angular.z = math.copysign(self.angular_speed, yaw_err)
            else:
                self.state = 'MOVE_TO_P2_GRID'
                self.target_yaw = None
                self.get_logger().info("✅ Left 90° complete. Moving to P2 (grid alignment).")

        # 7️⃣ Move to P2 using axis alignment logic (with obstacle-based 90° left turn)
        elif self.state == 'MOVE_TO_P2_GRID':
            gx, gy, _ = self.P2
            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)

            # Step 1️⃣ — Move along X direction first
            if not hasattr(self, 'finished_x_move'):
                target_angle = 0.0 if dx > 0 else math.pi
                angle_error = self._normalize(target_angle - yaw)

                if abs(angle_error) > 0.15:
                    # Rotate to face ±X
                    cmd.angular.z = self.angular_speed * math.copysign(1.0, angle_error)
                else:
                    # Move straight along X until near goal OR obstacle detected
                    cmd.linear.x = self.linear_speed

                    # --- obstacle detection while moving in X ---
                    if self.front < self.obstacle_threshold:
                        self.get_logger().info("🧱 Obstacle detected while moving in X → turning 90° left.")
                        self.state = 'TURN_LEFT_TO_Y_AFTER_OBS'
                        self.target_yaw = None
                        cmd = Twist()
                        self.cmd_pub.publish(cmd)
                        return

                # If X coordinate is reached (within tolerance)
                if abs(dx) < self.dist_tolerance:
                    self.get_logger().info("✅ Finished moving along X. Proceeding toward Y axis to reach P2.")
                    self.finished_x_move = True

            # Step 2️⃣ — After X done (or after left turn), move along Y
            elif hasattr(self, 'finished_x_move'):
                target_angle = math.pi / 2 if dy > 0 else -math.pi / 2
                angle_error = self._normalize(target_angle - yaw)

                if abs(angle_error) > 0.15:
                    # Align toward Y axis
                    cmd.angular.z = self.angular_speed * math.copysign(1.0, angle_error)
                else:
                    # Move straight along Y axis
                    cmd.linear.x = self.linear_speed

                # Check arrival at P2
                if dist < self.dist_tolerance:
                    self.state = 'MOVE_TO_P3'
                    self.get_logger().info("✅ Reached P2. Moving to P3 (origin).")

        # 🔁 Sub-state — turn 90° left if obstacle encountered along X
        elif self.state == 'TURN_LEFT_TO_Y_AFTER_OBS':
            if self.target_yaw is None:
                self.target_yaw = self._normalize(yaw + math.pi / 2)
                self.get_logger().info("↩️ Turning 90° left to bypass obstacle and align along Y.")
            yaw_err = self._normalize(self.target_yaw - yaw)

            if abs(yaw_err) > self.yaw_tolerance:
                cmd.angular.z = math.copysign(self.angular_speed, yaw_err)
            else:
                self.get_logger().info("✅ 90° left turn complete. Continuing along Y toward P2.")
                self.target_yaw = None
                self.finished_x_move = True   # mark X stage as done
                self.state = 'MOVE_TO_P2_GRID'  # resume normal grid motion

        # 8️⃣ Move to P3
        elif self.state == 'MOVE_TO_P3':
            gx, gy, _ = self.P3
            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)
            goal_yaw = math.atan2(dy, dx)
            yaw_err = self._normalize(goal_yaw - yaw)
            if abs(yaw_err) > self.yaw_tolerance:
                cmd.angular.z = math.copysign(self.angular_speed, yaw_err)
            else:
                cmd.linear.x = self.linear_speed
            if dist < self.dist_tolerance:
                # Instead of STOP immediately, start the AFTER_P3 sequence
                self.state = 'TURN_RIGHT_AFTER_P3'
                self.get_logger().info("📍 Reached P3. Starting post-P3 sequence: turn right 90°, move until obstacle, turn left 90°, move until obstacle.")

        # ---- NEW: AFTER P3 SEQUENCE ----
        # Turn right 90° after P3
        elif self.state == 'TURN_RIGHT_AFTER_P3':
            if self.target_yaw is None:
                self.target_yaw = self._normalize(yaw + self.after_p3_right_turn)
                self.get_logger().info("↪️ Turning right 90° after P3...")
            yaw_err = self._normalize(self.target_yaw - yaw)
            if abs(yaw_err) > self.yaw_tolerance:
                cmd.angular.z = math.copysign(self.angular_speed, yaw_err)
            else:
                self.target_yaw = None
                self.state = 'MOVE_AFTER_P3_RIGHT'
                self.get_logger().info("✅ Right 90° complete. Moving forward until obstacle...")

        # Move forward after right turn until obstacle
        elif self.state == 'MOVE_AFTER_P3_RIGHT':
            if self.front < self.emergency_stop:
                # immediate emergency -> stop and proceed to next turn
                self.get_logger().info("🧱 Emergency obstacle detected after right move; stopping and rotating left 90°.")
                self.state = 'TURN_LEFT_AFTER_P3'
            # elif self.front < self.obstacle_threshold:
            #     self.get_logger().info("🧱 Obstacle detected after right move. Rotating left 90°...")
            #     self.state = 'TURN_LEFT_AFTER_P3'
            else:
                cmd.linear.x = self.linear_speed

        # Turn left 90° after obstacle encountered
        elif self.state == 'TURN_LEFT_AFTER_P3':
            if self.target_yaw is None:
                self.target_yaw = self._normalize(yaw + self.after_p3_left_turn)
                self.get_logger().info("↩️ Turning left 90° after encountering obstacle...")
            yaw_err = self._normalize(self.target_yaw - yaw)
            if abs(yaw_err) > self.yaw_tolerance:
                cmd.angular.z = math.copysign(self.angular_speed, yaw_err)
            else:
                self.target_yaw = None
                self.state = 'MOVE_AFTER_P3_LEFT_UNTIL_OBS'
                self.get_logger().info("✅ Left 90° complete. Moving forward until obstacle...")

        # Move forward after left turn until obstacle, then STOP
        elif self.state == 'MOVE_AFTER_P3_LEFT_UNTIL_OBS':
            if self.front < self.emergency_stop:
                self.get_logger().info("🧱 Emergency obstacle detected after left move; stopping mission.")
                self.state = 'STOP'
            elif self.front < self.obstacle_threshold:
                self.get_logger().info("🧱 Obstacle detected after left move; stopping mission.")
                self.state = 'STOP'
            else:
                cmd.linear.x = self.linear_speed

        # 9️⃣ Stop
        elif self.state == 'STOP':
            cmd = Twist()

        # Publish
        self.cmd_pub.publish(cmd)

    # ---------------- Helpers ----------------
    def _normalize(self, ang):
        while ang > math.pi:
            ang -= 2 * math.pi
        while ang < -math.pi:
            ang += 2 * math.pi
        return ang


def main(args=None):
    rclpy.init(args=args)
    node = EbotSpecMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down.")
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()