#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
import numpy as np
import math
import time

# ---- Target Waypoints ----
WAYPOINTS = [
    {'position': [-0.214, -0.532, 0.557], 'orientation': [0.707, 0.028, 0.034, 0.707]},  # P1
    {'position': [-0.159, 0.501, 0.415],  'orientation': [0.029, 0.997, 0.045, 0.033]},  # P2
    {'position': [-0.806, 0.010, 0.182],  'orientation': [-0.684, 0.726, 0.05, 0.008]}   # P3
]


class UR5ServoController(Node):
    def __init__(self):
        super().__init__('precise_cartesian_servo')

        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Controller configuration
        self.rate_hz = 30.0
        self.dt = 1.0 / self.rate_hz
        self.create_timer(self.dt, self.update_loop)

        # Gains and limits
        self.kp_lin = 0.8
        self.kp_ang = 0.4
        self.max_lin = 0.2
        self.max_ang = 0.5
        self.tolerance_pos = 0.15
        self.tolerance_ori = 0.15

        # TF & state management
        self.base_frame = 'base_link'
        self.ee_frame = 'wrist_3_link'
        self.tf_delay = 1.0
        self.start_time = time.time()

        self.initial_pose = None
        self.sequence = []
        self.current_target_index = 0
        self.waiting = False
        self.wait_timer = None
        self.active = False

        self.get_logger().info("UR5 servo controller initialized. Waiting to capture initial pose...")

    # ------------- TF and Quaternion Helpers -------------
    def get_current_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1)
            )
            pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            quat = np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
            return pos, self.normalize_quat(quat)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None, None

    def normalize_quat(self, q):
        n = np.linalg.norm(q)
        return q / n if n > 1e-8 else np.array([0, 0, 0, 1])

    def quat_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    def quat_conjugate(self, q):
        return np.array([-q[0], -q[1], -q[2], q[3]])

    def quat_to_axis_angle(self, q):
        q = self.normalize_quat(q)
        w = max(min(q[3], 1.0), -1.0)
        angle = 2 * math.acos(w)
        s = math.sqrt(max(1 - w*w, 0))
        axis = np.array([1, 0, 0]) if s < 1e-6 else q[:3] / s
        return axis, angle

    def pose_error(self, tpos, tquat, cpos, cquat):
        pos_err = tpos - cpos
        q_err = self.quat_multiply(tquat, self.quat_conjugate(cquat))
        axis, ang = self.quat_to_axis_angle(q_err)
        ori_err = axis * ang
        return pos_err, ori_err

    # ------------- Main Loop -------------
    def update_loop(self):
        if (time.time() - self.start_time) < self.tf_delay:
            return

        curr_pos, curr_quat = self.get_current_pose()
        if curr_pos is None:
            return

        # Capture initial pose once
        if not self.active and self.initial_pose is None:
            self.initial_pose = {'position': curr_pos, 'orientation': curr_quat}
            self.sequence = self.create_motion_sequence()
            self.active = True
            self.get_logger().info("✅ Initial pose recorded and motion sequence created.")
            return

        if not self.active or self.waiting:
            return

        target = self.sequence[self.current_target_index]
        tpos = np.array(target['position'])
        tquat = self.normalize_quat(np.array(target['orientation']))

        pos_err, ori_err = self.pose_error(tpos, tquat, curr_pos, curr_quat)
        pos_dist = np.linalg.norm(pos_err)
        ori_dist = np.linalg.norm(ori_err)

        # If reached target
        if pos_dist < self.tolerance_pos and ori_dist < self.tolerance_ori:
            self.get_logger().info(f"Reached target {self.current_target_index+1}/{len(self.sequence)} "
                                   f"| pos_err={pos_dist:.3f}, ori_err={ori_dist:.3f}")
            self.waiting = True
            self.wait_timer = self.create_timer(1.0, self.advance_target)
            self.pub.publish(Twist())
            return

        # Velocity command
        cmd = Twist()
        cmd.linear.x = float(np.clip(self.kp_lin * pos_err[0], -self.max_lin, self.max_lin))
        cmd.linear.y = float(np.clip(self.kp_lin * pos_err[1], -self.max_lin, self.max_lin))
        cmd.linear.z = float(np.clip(self.kp_lin * pos_err[2], -self.max_lin, self.max_lin))
        cmd.angular.x = float(np.clip(self.kp_ang * ori_err[0], -self.max_ang, self.max_ang))
        cmd.angular.y = float(np.clip(self.kp_ang * ori_err[1], -self.max_ang, self.max_ang))
        cmd.angular.z = float(np.clip(self.kp_ang * ori_err[2], -self.max_ang, self.max_ang))
        self.pub.publish(cmd)

    # ------------- Waypoint Sequencing -------------
    def create_motion_sequence(self):
        """
        Creates motion sequence:
        Initial → P1 → Initial → anticlockwise arc to P2 → P3
        """
        seq = []

        init = self.initial_pose
        seq.append(init)        # Initial
        seq.append(WAYPOINTS[0])  # P1
        seq.append(init)        # Return to Initial

        # Add an intermediate offset path for anticlockwise motion to P2
        anticlockwise_offset = np.array(init['position']) + np.array([0.0, 0.25, 0.0])
        anticlockwise_pose = {
            'position': anticlockwise_offset.tolist(),
            'orientation': init['orientation'].tolist()
        }

        seq.append(anticlockwise_pose)  # Shoulder swing path
        seq.append(WAYPOINTS[1])        # P2
        seq.append(WAYPOINTS[2])        # P3
        return seq

    def advance_target(self):
        self.current_target_index += 1
        if self.wait_timer:
            self.wait_timer.cancel()
            self.wait_timer = None
        self.waiting = False

        if self.current_target_index >= len(self.sequence):
            self.get_logger().info("✅ Motion sequence complete.")
            self.active = False
        else:
            self.get_logger().info(f"➡️ Moving to target {self.current_target_index+1}/{len(self.sequence)}")


# ------------- Main Entrypoint -------------
def main(args=None):
    rclpy.init(args=args)
    node = UR5ServoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down UR5 servo controller.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()