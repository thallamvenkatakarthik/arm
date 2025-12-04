# Team ID:          3577
# Theme:            Multi Waypoint Navigation
# Author List:      KARTHIK, ANUDEEP, JOSH VISWANADH, MANIKANTA
# Filename:         ebot_nav_task1A.py
# Functions:        __init__, odom_callback, scan_callback, navigate_to_goal, normalize_angle, main
# Global variables: None


import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from tf_transformations import euler_from_quaternion
import math


class MultiWaypointNav(Node):
    '''
    Purpose:
    ---
    A ROS2 Node to navigate a robot through multiple waypoints,
    avoiding obstacles detected with a laser scanner.

    Functions:
    ---
    __init__: Initializes publishers, subscribers, and parameters.
    odom_callback: Processes odometry updates and triggers navigation.
    scan_callback: Processes laser scan data to detect obstacles.
    navigate_to_goal: Controls the robot to reach the current goal.
    normalize_angle: Normalizes angles to the range [-pi, pi].
    '''

    def __init__(self):
        '''
        Purpose:
        ---
        Initialize the node, publishers, subscribers and navigation parameters.
        Sets up the waypoint goals and initializes status variables.

        Inputs:
        ---
        None

        Returns:
        ---
        None
        '''
        super().__init__('multi_waypoint_nav')

        # Publishers and Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Waypoints: (x-position, y-position, yaw orientation in radians)
        self.goals = [
             (0.26, -1.95, 1.57 ),
            ( -1.48, -0.67, -1.57),
            (-1.53, -6.61, -1.57 )
        ]

        # Index of the current goal in the goals list
        self.goal_index = 0

        # Current pose of the robot as (x, y, yaw)
        self.pose = None

        # Flag set when obstacle is detected in front
        self.obstacle_detected = False

        # Navigation parameters
        self.dist_tolerance = 0.15                # Distance tolerance to waypoint (meters)
        self.yaw_tolerance = 0.1                  # Yaw tolerance to waypoint (radians)
        self.linear_speed = 0.5                    # Linear speed (m/s)
        self.angular_speed = 0.6                   # Angular speed (rad/s)
        self.obstacle_distance_threshold = 0.5    # Minimum obstacle distance to trigger avoidance (meters)

    def odom_callback(self, msg):
        '''
        Purpose:
        ---
        Callback function for Odometry messages.
        Updates the robot's current pose and commands navigation if goals remain.

        Input Arguments:
        ---
        msg : Odometry message
            Current odometry data including position and orientation.

        Returns:
        ---
        None
        '''
        pos = msg.pose.pose
        x, y = pos.position.x, pos.position.y
        q = pos.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.pose = (x, y, yaw)

        if self.goal_index < len(self.goals):
            self.navigate_to_goal()
        else:
            # Stop robot once all goals are reached
            self.cmd_pub.publish(Twist())

    def scan_callback(self, msg):
        '''
        Purpose:
        ---
        Callback function for LaserScan messages.
        Detects obstacles directly in front of the robot within a 60-degree arc.

        Input Arguments:
        ---
        msg : LaserScan message
            Current laser scan ranges around the robot.

        Returns:
        ---
        None
        '''
        # Combine ranges from front 30 degrees left and right (assuming 360 values)
        front_arc = msg.ranges[0:30] + msg.ranges[330:360]

        min_distance = min(front_arc) if front_arc else float('inf')

        # Set obstacle detected flag if an object is too close
        self.obstacle_detected = (min_distance < self.obstacle_distance_threshold)

    def navigate_to_goal(self):
        '''
        Purpose:
        ---
        Controls the robot to navigate toward the current waypoint goal.
        Handles obstacle avoidance by turning in place if obstacle detected.

        Inputs:
        ---
        None

        Returns:
        ---
        None
        '''
        cmd = Twist()

        if self.obstacle_detected:
            # Simple avoidance: move forward slowly and turn left
            cmd.linear.x = 0.5
            cmd.angular.z = 0.2
        elif self.pose is not None:
            gx, gy, gyaw = self.goals[self.goal_index]
            x, y, yaw = self.pose
            dx, dy = gx - x, gy - y

            distance = math.hypot(dx, dy)

            if distance > self.dist_tolerance:
                # Rotate toward the goal position
                target_angle = math.atan2(dy, dx)
                angle_error = self.normalize_angle(target_angle - yaw)

                if abs(angle_error) > 0.2:
                    cmd.angular.z = self.angular_speed * math.copysign(1.0, angle_error)
                else:
                    cmd.linear.x = self.linear_speed
                    cmd.angular.z = 0.8 * angle_error
            else:
                # Close enough to goal position, align final yaw
                yaw_error = self.normalize_angle(gyaw - yaw)
                if abs(yaw_error) > self.yaw_tolerance:
                    cmd.angular.z = self.angular_speed * math.copysign(1.0, yaw_error)
                else:
                    self.get_logger().info(f"Reached goal {self.goal_index + 1}")
                    self.goal_index += 1

        self.cmd_pub.publish(cmd)

    @staticmethod
    def normalize_angle(angle):
        '''
        Purpose:
        ---
        Normalize an angle to the range [-pi, pi].

        Input Arguments:
        ---
        angle : float
            Angle in radians to normalize.

        Returns:
        ---
        float: Normalized angle within [-pi, pi].
        '''
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi

        return angle


def main():
    '''
    Purpose:
    ---
    Initialize the ROS2 node and keep it spinning for callbacks.

    Inputs:
    ---
    None

    Returns:
    ---
    None
    '''
    rclpy.init()
    node = MultiWaypointNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# Main entry point
if __name__ == '__main__':
    main()
