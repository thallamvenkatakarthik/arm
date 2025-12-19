#!/usr/bin/env python3
'''
Team ID:          3577
Theme:            Krishi coBot
Author List:      D Anudeep, Karthik, Vishwa, Manikanta
Filename:         task4a_manipulation.py
Purpose:          UR5 servo-based pick & place on REAL HARDWARE
Behavior (UNCHANGED):
  - Initial -> P1 -> Initial -> anticlockwise offset -> P2 -> P3
  - P1: pick fertilizer -> place on drop
  - P2: pick bad fruits -> drop at P3
  - Stops at P3
'''

import rclpy
from rclpy.node import Node

import numpy as np
import math
import time
from threading import Thread

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import SetBool
from std_msgs.msg import String, Float32
from scipy.spatial.transform import Rotation as R


# ---------------- Waypoints (UNCHANGED) ----------------
WAYPOINTS = [
    {'position': [-0.214, -0.532, 0.557], 'orientation': [0.707, 0.028, 0.034, 0.707]},
    {'position': [-0.159,  0.501, 0.415], 'orientation': [0.029, 0.997, 0.045, 0.033]},
    {'position': [-0.806,  0.010, 0.182], 'orientation': [-0.684, 0.726, 0.05, 0.008]}
]

# ---------------- Frames ----------------
FERTILISER_FRAME = '3577_fertilizer_1'
EBOT_TOP_FRAME   = '3577_fertilizer_drop'
BAD_FRUIT_FRAMES = ['3577_bad_fruit_3', '3577_bad_fruit_2', '3577_bad_fruit_1']

# ---------------- Motion Parameters ----------------
PRE_Z_OFFSET = 0.10
LIFT_Z_AFTER_ATTACH = 0.20
TRASH_CLEARANCE_Z = 0.20

SAFE_Z_VEL = 0.05
FORCE_THRESH = 8.0


class UR5ServoPickPlace(Node):

    def __init__(self):
        super().__init__('ur5_servo_pick_place')

        # Servo publisher (HARDWARE)
        self.pub = self.create_publisher(
            TwistStamped, '/delta_twist_cmds', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Magnet service (replaces Attach/Detach)
        self.magnet_client = self.create_client(SetBool, '/magnet')
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for magnet service...')

        # Force feedback
        self.force_z = 0.0
        self.create_subscription(Float32, '/net_wrench', self.force_cb, 10)

        # Sync logic (UNCHANGED)
        self.dock_reached = False
        self.create_subscription(String, '/detection_status', self.dock_status_cb, 10)

        # Servo params (slowed for hardware)
        self.rate_hz = 30.0
        self.dt = 1.0 / self.rate_hz
        self.kp_lin = 1.0
        self.kp_ang = 0.6
        self.max_lin = 0.10
        self.max_ang = 0.5
        self.tolerance_pos = 0.02
        self.tolerance_ori = 0.20

        self.base_frame = 'base_link'
        self.ee_frame = 'wrist_3_link'
        self.start_time = time.time()
        self.tf_delay = 1.0

        self.initial_pose = None
        self.sequence = []
        self.current_target_index = 0
        self.waiting = False
        self.active = False

        self.create_timer(self.dt, self.update_loop)
        self.get_logger().info('✅ UR5 hardware servo node ready')

    # ---------------- Safety ----------------
    def stop(self):
        self.pub.publish(TwistStamped())
        time.sleep(0.05)

    # ---------------- Callbacks ----------------
    def force_cb(self, msg):
        self.force_z = msg.data

    def dock_status_cb(self, msg):
        if msg.data.startswith("DOCK_STATION"):
            self.dock_reached = True

    # ---------------- Magnet ----------------
    def magnet(self, state):
        req = SetBool.Request()
        req.data = state
        self.magnet_client.call_async(req)
        time.sleep(0.3)
        self.stop()

    # ---------------- TF Helpers ----------------
    def get_current_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1))

            pos = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ])
            quat = np.array([
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w
            ])
            return pos, quat
        except Exception:
            return None, None

    def get_tf_pose(self, frame):
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=2))

            pos = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ])
            quat = np.array([
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w
            ])
            return pos, quat
        except Exception:
            return None, None

    # ---------------- Servo Control (UNCHANGED LOGIC) ----------------
    def pose_error(self, tpos, tquat, cpos, cquat):
        return tpos - cpos, np.zeros(3)

    def publish_twist_for_error(self, pos_err):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear.x = float(np.clip(self.kp_lin * pos_err[0], -self.max_lin, self.max_lin))
        cmd.twist.linear.y = float(np.clip(self.kp_lin * pos_err[1], -self.max_lin, self.max_lin))
        cmd.twist.linear.z = float(np.clip(self.kp_lin * pos_err[2], -self.max_lin, self.max_lin))
        self.pub.publish(cmd)

    def move_and_wait(self, target_pos):
        while rclpy.ok():
            curr_pos, _ = self.get_current_pose()
            if curr_pos is None:
                continue

            err = target_pos - curr_pos
            if np.linalg.norm(err) < self.tolerance_pos:
                self.stop()
                return

            self.publish_twist_for_error(err)
            time.sleep(self.dt)

    # ---------------- Pick / Place (HARDWARE SAFE, LOGIC SAME) ----------------
    def pick_object(self, frame):
        pos, _ = self.get_tf_pose(frame)
        if pos is None:
            return False

        self.move_and_wait(pos + np.array([0, 0, PRE_Z_OFFSET]))

        # Force-limited descent
        while self.force_z < FORCE_THRESH:
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.twist.linear.z = -SAFE_Z_VEL
            self.pub.publish(cmd)
            time.sleep(self.dt)

        self.stop()
        self.magnet(True)

        self.move_and_wait(pos + np.array([0, 0, LIFT_Z_AFTER_ATTACH]))
        return True

    def place_object(self, pos):
        self.move_and_wait(pos + np.array([0, 0, PRE_Z_OFFSET]))
        self.move_and_wait(pos)
        self.magnet(False)
        self.move_and_wait(pos + np.array([0, 0, TRASH_CLEARANCE_Z]))

    # ---------------- Sequence (UNCHANGED) ----------------
    def create_motion_sequence(self):
        init = self.initial_pose
        seq = [
            init,
            WAYPOINTS[0],
            init,
            {'position': (np.array(init['position']) + np.array([0, 0.25, 0])).tolist()},
            WAYPOINTS[1],
            WAYPOINTS[2]
        ]
        return seq

    def perform_action_for_target(self, idx):
        if idx == 1:
            if self.pick_object(FERTILISER_FRAME):
                pos, _ = self.get_tf_pose(EBOT_TOP_FRAME)
                if pos is not None:
                    self.place_object(pos)

        elif idx == 4:
            for fruit in BAD_FRUIT_FRAMES:
                if self.pick_object(fruit):
                    self.place_object(np.array(WAYPOINTS[2]['position']))

        self.current_target_index += 1
        self.waiting = False

    # ---------------- Main Loop ----------------
    def update_loop(self):
        if not self.dock_reached:
            return
        if (time.time() - self.start_time) < self.tf_delay:
            return

        curr_pos, _ = self.get_current_pose()
        if curr_pos is None:
            return

        if self.initial_pose is None:
            self.initial_pose = {'position': curr_pos.tolist()}
            self.sequence = self.create_motion_sequence()
            self.active = True
            return

        if self.waiting or not self.active:
            return

        target = self.sequence[self.current_target_index]
        target_pos = np.array(target['position'])

        if np.linalg.norm(target_pos - curr_pos) < self.tolerance_pos:
            self.stop()
            self.waiting = True
            Thread(
                target=lambda: self.perform_action_for_target(self.current_target_index),
                daemon=True
            ).start()
            return

        self.publish_twist_for_error(target_pos - curr_pos)


def main():
    rclpy.init()
    node = UR5ServoPickPlace()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

