#!/usr/bin/env python3
'''
Team ID:          3577
Theme:            Krishi coBot
Author List:      D Anudeep, Karthik, Vishwa, Manikanta
Filename:         task3b_ur5_servo_pick_place.py
Purpose:
--------
UR5 servo-based integrated pick & place for Task 3B.

Behavior (same as Task2B):
  - Sequence: Initial -> P1 -> Initial -> anticlockwise offset -> P2 -> P3
  - P1: pick fertiliser (aruco_3) -> place on eBot top (aruco_6)
  - P2: pick bad fruits (3577_bad_fruit_1/2/3) one-by-one -> drop at P3
  - Stops at P3

New for Task 3B:
  - Subscribes to /detection_status and only starts motion AFTER
    receiving a DOCK_STATION,x,y,0 message (eBot docked).
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
from std_msgs.msg import String

import numpy as np
import math
import time
from threading import Thread
from linkattacher_msgs.srv import AttachLink, DetachLink
from scipy.spatial.transform import Rotation as R


# ---------------- Waypoints ----------------
WAYPOINTS = [
    {'position': [-0.214, -0.532, 0.557], 'orientation': [0.707, 0.028, 0.034, 0.707]},  # P1 (fertilizer)
    {'position': [-0.159, 0.501, 0.415],  'orientation': [0.029, 0.997, 0.045, 0.033]},  # P2 (fruits)
    {'position': [-0.806, 0.010, 0.182],  'orientation': [-0.684, 0.726, 0.05, 0.008]}   # P3 (dustbin)
]

# ---------------- Object Frames ----------------
FERTILISER_FRAME = 'aruco_3'
EBOT_TOP_FRAME = 'aruco_6'
BAD_FRUIT_FRAMES = ['3577_bad_fruit_1', '3577_bad_fruit_2', '3577_bad_fruit_3']

# ---------------- Models ----------------
FERTILISER_MODEL = 'fertiliser_can'
BAD_FRUIT_MODEL = 'bad_fruit'

# ---------------- Motion Parameters ----------------
PRE_Z_OFFSET = 0.01             # approach above object
GRASP_Z_OFFSET = -0.02          # go slightly down to grasp
LIFT_Z_AFTER_ATTACH = 0.09      # lift a bit
LIFT_ZZ_AFTER_ATTACH = 0.25     # lift after placing
ATTACH_DISTANCE_THRESH = 0.10   # distance to trigger attach


class UR5ServoPickPlace(Node):
    def __init__(self):
        super().__init__('ur5_servo_pick_place_3b')

        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # attach/detach services
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        # subscribe to detection_status for DOCK_STATION sync
        self.status_sub = self.create_subscription(
            String, '/detection_status', self.status_cb, 10)

        # servo control params
        self.rate_hz = 30.0
        self.dt = 1.0 / self.rate_hz
        self.kp_lin = 1.2
        self.kp_ang = 0.6
        self.max_lin = 0.25
        self.max_ang = 0.5
        self.tolerance_pos = 0.10
        self.tolerance_ori = 0.12

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

        # will be set True after DOCK_STATION arrives
        self.ready_to_start = False

        self.create_timer(self.dt, self.update_loop)
        self.get_logger().info('✅ UR5 servo pick-place (Task3B) initialised. Waiting for DOCK_STATION...')

    # ---------- detection_status callback ----------
    def status_cb(self, msg: String):
        parts = msg.data.split(',')
        if len(parts) < 1:
            return
        if parts[0] == "DOCK_STATION":
            if not self.ready_to_start:
                self.ready_to_start = True
                self.get_logger().info('🚢 DOCK_STATION received – starting UR5 sequence.')

    # ---------- Quaternion helpers ----------
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
        w = max(min(q[3], 1.0), -1.0)
        angle = 2.0 * math.acos(w)
        s = math.sqrt(max(1 - w*w, 0))
        axis = np.array([1.0, 0.0, 0.0]) if s < 1e-6 else q[:3] / s
        return axis, angle

    # ---------- TF Helpers ----------
    def get_current_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame,
                rclpy.time.Time(), rclpy.duration.Duration(seconds=1))
            pos = np.array([trans.transform.translation.x,
                            trans.transform.translation.y,
                            trans.transform.translation.z], dtype=float)
            quat = np.array([trans.transform.rotation.x,
                             trans.transform.rotation.y,
                             trans.transform.rotation.z,
                             trans.transform.rotation.w], dtype=float)
            return pos, self.normalize_quat(quat)
        except Exception:
            return None, None

    def get_tf_pose(self, frame_name):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, frame_name,
                rclpy.time.Time(), rclpy.duration.Duration(seconds=2))
            pos = np.array([trans.transform.translation.x,
                            trans.transform.translation.y,
                            trans.transform.translation.z], dtype=float)
            quat = np.array([trans.transform.rotation.x,
                             trans.transform.rotation.y,
                             trans.transform.rotation.z,
                             trans.transform.rotation.w], dtype=float)
            if 'aruco' in frame_name or 'bad_fruit' in frame_name:
                pos[2] += 0.015
                quat = R.from_euler('xyz', [math.pi, 0, 0]).as_quat()
            return pos, self.normalize_quat(quat)
        except Exception as e:
            self.get_logger().warn(f'get_tf_pose: failed for {frame_name}: {e}')
            return None, None

    # ---------- Pose control ----------
    def pose_error(self, tpos, tquat, cpos, cquat):
        pos_err = tpos - cpos
        q_err = self.quat_multiply(tquat, self.quat_conjugate(cquat))
        axis, ang = self.quat_to_axis_angle(q_err)
        return pos_err, axis * ang

    def publish_twist_for_error(self, pos_err, ori_err):
        cmd = Twist()
        cmd.linear.x = float(np.clip(self.kp_lin * pos_err[0], -self.max_lin, self.max_lin))
        cmd.linear.y = float(np.clip(self.kp_lin * pos_err[1], -self.max_lin, self.max_lin))
        cmd.linear.z = float(np.clip(self.kp_lin * pos_err[2], -self.max_lin, self.max_lin))
        cmd.angular.x = float(np.clip(self.kp_ang * ori_err[0], -self.max_ang, self.max_ang))
        cmd.angular.y = float(np.clip(self.kp_ang * ori_err[1], -self.max_ang, self.max_ang))
        cmd.angular.z = float(np.clip(self.kp_ang * ori_err[2], -self.max_ang, self.max_ang))
        self.pub.publish(cmd)

    # ---------- Motion Sequence ----------
    def create_motion_sequence(self):
        seq = []
        init = self.initial_pose
        seq.append(init)
        seq.append(WAYPOINTS[0])
        seq.append(init)
        anticlockwise_offset = np.array(init['position']) + np.array([0.0, 0.25, 0.0])
        anticlockwise_pose = {'position': anticlockwise_offset.tolist(),
                              'orientation': init['orientation']}
        seq.append(anticlockwise_pose)
        seq.append(WAYPOINTS[1])
        seq.append(WAYPOINTS[2])
        return seq

    # ---------- Attach / Detach ----------
    def attach_model(self, model1_name):
        if not self.attach_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('attach service unavailable')
            return False
        req = AttachLink.Request()
        req.model1_name = model1_name
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        self.attach_client.call_async(req)
        self.get_logger().info(f'Attached {model1_name}')
        return True

    def detach_model(self, model1_name):
        if not self.detach_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('detach service unavailable')
            return False
        req = DetachLink.Request()
        req.model1_name = model1_name
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        self.detach_client.call_async(req)
        self.get_logger().info(f'Detached {model1_name}')
        return True

    # ---------- Pick / Place ----------
    def move_and_wait(self, target_pos, target_quat, timeout=10.0):
        target_pos = np.array(target_pos)
        target_quat = self.normalize_quat(target_quat)
        start = time.time()
        while time.time() - start < timeout:
            curr_pos, curr_quat = self.get_current_pose()
            if curr_pos is None:
                time.sleep(self.dt)
                continue
            pos_err, ori_err = self.pose_error(target_pos, target_quat, curr_pos, curr_quat)
            pos_dist = np.linalg.norm(pos_err)
            ori_dist = np.linalg.norm(ori_err)
            if pos_dist < self.tolerance_pos and ori_dist < self.tolerance_ori:
                self.pub.publish(Twist())
                return True
            self.publish_twist_for_error(pos_err, ori_err)
            time.sleep(self.dt)
        self.pub.publish(Twist())
        self.get_logger().warn('move_and_wait: timeout')
        return False

    def pick_and_attach(self, tf_frame, model_name):
        pos, quat = self.get_tf_pose(tf_frame)
        if pos is None:
            return False

        pre = pos + np.array([0, 0, PRE_Z_OFFSET])
        self.move_and_wait(pre, quat)

        grasp = pos + np.array([0, 0, GRASP_Z_OFFSET])
        self.move_and_wait(grasp, quat)

        cpos, _ = self.get_current_pose()
        dist = np.linalg.norm(cpos - pos)
        self.get_logger().info(f'Distance before attach: {dist:.3f} m')

        if dist > ATTACH_DISTANCE_THRESH:
            self.get_logger().warn('Too far from object — adjusting down slightly')
            deeper = pos + np.array([0, 0, -0.02])
            self.move_and_wait(deeper, quat)
            cpos, _ = self.get_current_pose()
            dist = np.linalg.norm(cpos - pos)

        if dist < ATTACH_DISTANCE_THRESH:
            self.attach_model(model_name)
            time.sleep(0.5)
        else:
            self.get_logger().error('Still too far to attach!')
            return False

        lifted = pos + np.array([0, 0, LIFT_Z_AFTER_ATTACH])
        self.move_and_wait(lifted, quat)
        self.get_logger().info(f"✅ Picked and safely lifted {model_name}")
        return True

    def place_and_detach(self, pos, quat, model_name):
        pos = np.array(pos)
        quat = self.normalize_quat(np.array(quat))

        pre = pos + np.array([0, 0, PRE_Z_OFFSET])
        self.move_and_wait(pre, quat)

        lower = pos + np.array([0, 0, GRASP_Z_OFFSET])
        self.move_and_wait(lower, quat)

        self.detach_model(model_name)
        time.sleep(0.3)

        lifted = pos + np.array([0, 0, LIFT_ZZ_AFTER_ATTACH])
        self.move_and_wait(lifted, quat)
        self.get_logger().info(f"🧲 Detached and lifted after placing {model_name}")
        return True

    # ---------- Task Actions ----------
    def perform_action_for_target(self, idx):
        try:
            if idx == 1:
                self.get_logger().info('[ACTION] At P1: pick fertiliser & place on eBot top')
                if self.pick_and_attach(FERTILISER_FRAME, FERTILISER_MODEL):
                    ebot_pos, ebot_quat = self.get_tf_pose(EBOT_TOP_FRAME)
                    if ebot_pos is not None:
                        self.place_and_detach(ebot_pos, ebot_quat, FERTILISER_MODEL)

            elif idx == 4:
                self.get_logger().info('[ACTION] At P2: picking bad fruits')
                trash_pos = WAYPOINTS[2]['position']
                trash_quat = WAYPOINTS[2]['orientation']
                for fruit in BAD_FRUIT_FRAMES:
                    if self.pick_and_attach(fruit, BAD_FRUIT_MODEL):
                        self.place_and_detach(trash_pos, trash_quat, BAD_FRUIT_MODEL)
                        self.move_and_wait(WAYPOINTS[1]['position'], WAYPOINTS[1]['orientation'])
        finally:
            self.advance_target()

    # ---------- Servo Loop ----------
    def update_loop(self):
        if (time.time() - self.start_time) < self.tf_delay:
            return

        # wait until eBot docked
        if not self.ready_to_start:
            return

        curr_pos, curr_quat = self.get_current_pose()
        if curr_pos is None:
            return

        if not self.active and self.initial_pose is None:
            self.initial_pose = {
                'position': curr_pos.tolist(),
                'orientation': curr_quat.tolist()
            }
            self.get_logger().info(f'Initial pose captured: {self.initial_pose}')
            self.sequence = self.create_motion_sequence()
            self.active = True
            self.get_logger().info('Motion sequence created.')
            return

        if not self.active or self.waiting:
            return

        target = self.sequence[self.current_target_index]
        tpos = np.array(target['position'])
        tquat = self.normalize_quat(np.array(target['orientation']))

        pos_err, ori_err = self.pose_error(tpos, tquat, curr_pos, curr_quat)
        pos_dist, ori_dist = np.linalg.norm(pos_err), np.linalg.norm(ori_err)

        if pos_dist < self.tolerance_pos and ori_dist < self.tolerance_ori:
            self.pub.publish(Twist())
            self.waiting = True
            self.wait_timer = self.create_timer(
                0.8, lambda: self._timer_action_wrapper(self.current_target_index))
            return

        self.publish_twist_for_error(pos_err, ori_err)

    def _timer_action_wrapper(self, idx):
        if self.wait_timer:
            self.wait_timer.cancel()
            self.wait_timer = None
        Thread(target=lambda: self.perform_action_for_target(idx), daemon=True).start()

    def advance_target(self):
        self.current_target_index += 1
        self.waiting = False
        if self.current_target_index >= len(self.sequence):
            self.pub.publish(Twist())
            self.get_logger().info('✅ All waypoints and tasks completed. Stopping at P3.')
            self.active = False
        else:
            self.get_logger().info(
                f'Moving to next target {self.current_target_index+1}/{len(self.sequence)}')

def main(args=None):
    rclpy.init(args=args)
    node = UR5ServoPickPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down UR5 servo pick-place node.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

