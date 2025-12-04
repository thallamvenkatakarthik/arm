#!/usr/bin/env python3
'''
Team ID:          3577
Theme:            Krishi coBot
Author List:      D Anudeep, Karthik, Vishwa, Manikanta
Filename:         task2b_servo_pick_place.py
Purpose:          UR5 servo-based precise pick & place
Behavior:
  - Captures initial pose
  - Picks fertilizer → returns home → places on eBot → returns home
  - Picks bad fruits (1, 2, 3) sequentially → lifts 0.08m → drops in dustbin → returns home
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
from linkattacher_msgs.srv import AttachLink, DetachLink
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import time
from threading import Thread

# ---------------- Frames ----------------
FERTILISER_FRAME = 'aruco_3'
EBOT_TOP_FRAME = 'aruco_6'
BAD_FRUIT_FRAMES = ['3577_bad_fruit_1', '3577_bad_fruit_2', '3577_bad_fruit_3']

FERTILISER_MODEL = 'fertiliser_can'
BAD_FRUIT_MODEL = 'bad_fruit'

# ---------------- Motion Parameters ----------------
PRE_Z_OFFSET = 0.12               # pre-approach height
GRASP_Z_OFFSET = -0.01            # descend slightly to attach
FERTILIZER_LIFT = 0.15            # fertilizer lift height
FRUIT_LIFT = 0.08                 # fruit lift height
EBOT_APPROACH_OFFSET = 0.18       # approach height for eBot
ATTACH_DISTANCE_THRESH = 0.10     # attach distance threshold

# Dustbin static position (could also be TF-based if needed)
DUSTBIN_POS = [-0.806, 0.010, 0.182]
DUSTBIN_ORI = [-0.684, 0.726, 0.05, 0.008]


class UR5PickPlace(Node):
    def __init__(self):
        super().__init__('ur5_servo_pick_place')

        # ROS2 Setup
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        # Control Parameters
        self.rate_hz = 30.0
        self.dt = 1.0 / self.rate_hz
        self.kp_lin = 1.2
        self.kp_ang = 0.6
        self.max_lin = 0.2
        self.max_ang = 0.5
        self.tolerance_pos = 0.15
        self.tolerance_ori = 0.15

        self.base_frame = 'base_link'
        self.ee_frame = 'wrist_3_link'

        self.initial_pose = None
        self.task_started = False

        self.create_timer(self.dt, self.control_loop)
        self.get_logger().info('✅ UR5 pick-place node initialized. Waiting to capture initial pose...')

    # ---------------- Quaternion / TF helpers ----------------
    def normalize_quat(self, q):
        q = np.array(q, dtype=float)
        n = np.linalg.norm(q)
        return q / n if n > 1e-8 else np.array([0, 0, 0, 1.0])

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
        w = np.clip(q[3], -1.0, 1.0)
        angle = 2.0 * math.acos(w)
        s = math.sqrt(max(1 - w*w, 0))
        axis = np.array([1, 0, 0]) if s < 1e-6 else q[:3] / s
        return axis, angle

    def get_current_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.ee_frame, rclpy.time.Time())
            pos = np.array([trans.transform.translation.x,
                            trans.transform.translation.y,
                            trans.transform.translation.z])
            quat = np.array([trans.transform.rotation.x,
                             trans.transform.rotation.y,
                             trans.transform.rotation.z,
                             trans.transform.rotation.w])
            return pos, self.normalize_quat(quat)
        except Exception:
            return None, None

    def get_tf_pose(self, frame):
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, frame, rclpy.time.Time())
            pos = np.array([trans.transform.translation.x,
                            trans.transform.translation.y,
                            trans.transform.translation.z])
            quat = np.array([trans.transform.rotation.x,
                             trans.transform.rotation.y,
                             trans.transform.rotation.z,
                             trans.transform.rotation.w])
            pos[2] += 0.02  # adjust slightly above marker
            quat = R.from_euler('xyz', [math.pi, 0, 0]).as_quat()
            return pos, self.normalize_quat(quat)
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed for {frame}: {e}')
            return None, None

    # ---------------- Motion control ----------------
    def pose_error(self, tpos, tquat, cpos, cquat):
        pos_err = tpos - cpos
        q_err = self.quat_multiply(tquat, self.quat_conjugate(cquat))
        axis, ang = self.quat_to_axis_angle(q_err)
        return pos_err, axis * ang

    def publish_twist(self, pos_err, ori_err):
        cmd = Twist()
        cmd.linear.x = float(np.clip(self.kp_lin * pos_err[0], -self.max_lin, self.max_lin))
        cmd.linear.y = float(np.clip(self.kp_lin * pos_err[1], -self.max_lin, self.max_lin))
        cmd.linear.z = float(np.clip(self.kp_lin * pos_err[2], -self.max_lin, self.max_lin))
        cmd.angular.x = float(np.clip(self.kp_ang * ori_err[0], -self.max_ang, self.max_ang))
        cmd.angular.y = float(np.clip(self.kp_ang * ori_err[1], -self.max_ang, self.max_ang))
        cmd.angular.z = float(np.clip(self.kp_ang * ori_err[2], -self.max_ang, self.max_ang))
        self.pub.publish(cmd)

    def move_to_pose(self, pos, quat, timeout=15.0):
        start = time.time()
        while time.time() - start < timeout:
            cpos, cquat = self.get_current_pose()
            if cpos is None:
                time.sleep(self.dt)
                continue
            pos_err, ori_err = self.pose_error(np.array(pos), np.array(quat), cpos, cquat)
            if np.linalg.norm(pos_err) < self.tolerance_pos:
                self.pub.publish(Twist())
                return True
            self.publish_twist(pos_err, ori_err)
            time.sleep(self.dt)
        self.get_logger().warn('move_to_pose timeout')
        return False

    # ---------------- Attach / Detach ----------------
    def attach(self, model):
        req = AttachLink.Request()
        req.model1_name = model
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        self.attach_client.call_async(req)
        self.get_logger().info(f'Attached {model}')

    def detach(self, model):
        req = DetachLink.Request()
        req.model1_name = model
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        self.detach_client.call_async(req)
        self.get_logger().info(f'Detached {model}')

    # ---------------- Pick / Place operations ----------------
    def pick_object(self, frame, model, lift_height):
        pos, quat = self.get_tf_pose(frame)
        if pos is None:
            return False
        self.move_to_pose(pos + np.array([0, 0, PRE_Z_OFFSET]), quat)
        self.move_to_pose(pos + np.array([0, 0, GRASP_Z_OFFSET]), quat)
        self.attach(model)
        time.sleep(0.4)
        self.move_to_pose(pos + np.array([0, 0, lift_height]), quat)
        self.get_logger().info(f'✅ Picked {model} and lifted {lift_height}m')
        return True

    def place_object(self, pos, quat, model):
        pos = np.array(pos)
        quat = self.normalize_quat(np.array(quat))
        self.move_to_pose(pos + np.array([0, 0, EBOT_APPROACH_OFFSET]), quat)
        self.move_to_pose(pos + np.array([0, 0, GRASP_Z_OFFSET + 0.03]), quat)
        self.detach(model)
        time.sleep(0.3)
        self.move_to_pose(pos + np.array([0, 0, 0.15]), quat)
        self.get_logger().info(f'🧲 Placed {model}')
        return True

    # ---------------- Task flow ----------------
    def control_loop(self):
        if not self.task_started:
            cpos, cquat = self.get_current_pose()
            if cpos is None:
                return
            self.initial_pose = {'pos': cpos, 'quat': cquat}
            self.task_started = True
            self.get_logger().info('Initial pose captured. Starting task sequence...')
            Thread(target=self.execute_task_sequence, daemon=True).start()

    def go_home(self):
        pos, quat = self.initial_pose['pos'], self.initial_pose['quat']
        pos = np.array(pos) + np.array([0, 0, 0.05])
        self.get_logger().info('Returning to initial position...')
        self.move_to_pose(pos, quat)

    def execute_task_sequence(self):
        # 1️⃣ Pick fertilizer
        self.get_logger().info('Picking fertilizer...')
        if self.pick_object(FERTILISER_FRAME, FERTILISER_MODEL, FERTILIZER_LIFT):
            self.go_home()

            # 2️⃣ Place fertilizer on eBot
            ebot_pos, ebot_quat = self.get_tf_pose(EBOT_TOP_FRAME)
            if ebot_pos is not None:
                self.get_logger().info('Moving to eBot top...')
                self.place_object(ebot_pos, ebot_quat, FERTILISER_MODEL)
                self.go_home()

        # 3️⃣ Pick and place bad fruits sequentially
        for fruit in BAD_FRUIT_FRAMES:
            self.get_logger().info(f'Picking {fruit}...')
            if self.pick_object(fruit, BAD_FRUIT_MODEL, FRUIT_LIFT):
                self.move_to_pose(DUSTBIN_POS, DUSTBIN_ORI)
                self.place_object(DUSTBIN_POS, DUSTBIN_ORI, BAD_FRUIT_MODEL)

        self.get_logger().info('✅ All tasks completed successfully!')


def main(args=None):
    rclpy.init(args=args)
    node = UR5PickPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()