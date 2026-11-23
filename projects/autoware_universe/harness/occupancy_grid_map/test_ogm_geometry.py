# Copyright 2021 Tier IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
import time
import unittest

from geometry_msgs.msg import TransformStamped
import launch
import launch.actions
from launch_ros.substitutions import FindPackageShare
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import launch_testing.tools
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest
import rclpy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

PACKAGE_NAME = "autoware_probabilistic_occupancy_grid_map"
INPUT_TOPIC_RAW = "/raw"
INPUT_TOPIC_OBSTACLE = "/obstacle"


# test launcher
@pytest.mark.launch_test
def generate_test_description():
    """Launch file test description.

    Returns:
        _type_: launch.LaunchDescription
    """
    # get launch file path
    launch_file_path = (
        FindPackageShare(PACKAGE_NAME).find(PACKAGE_NAME)
        + "/launch/pointcloud_based_occupancy_grid_map.launch.py"
    )
    launch_args = [
        ("input/obstacle_pointcloud", INPUT_TOPIC_OBSTACLE),
        ("input/raw_pointcloud", INPUT_TOPIC_RAW),
    ]
    # action to include launch file
    test_launch_file = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(launch_file_path),
        launch_arguments=launch_args,
    )

    return launch.LaunchDescription(
        [
            test_launch_file,
            launch_testing.actions.ReadyToTest(),
        ]
    )


# util functions
def get_pointcloud_msg(pts: list):
    """Create ros2 point cloud message from list of points.

    Args:
        pts (list): list of points [[x, y, z], ...]

    Returns:
        PointCloud2: ros2 point cloud message
    """
    msg = PointCloud2()
    np_pts = np.array(pts, dtype=np.float32).reshape(-1, 3)
    binary_pts = np_pts.tobytes()

    # set current time
    now = rclpy.clock.Clock().now()
    msg.header.stamp = now.to_msg()
    msg.header.frame_id = "base_link"
    msg.height = 1
    msg.width = np_pts.shape[0]
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    # msg.data = b""
    # msg_data_list = []
    # for pt in pts:
    #     msg_data_list.append(struct.pack('fff', pt[0], pt[1], pt[2]))
    # msg.data = b"".join(msg_data_list)
    msg.data = binary_pts
    return msg


def parse_pointcloud_msg(msg: PointCloud2):
    """Parse ros2 point cloud message to list of points.

    Args:
        msg (PointCloud2): ros2 point cloud message

    Returns:
        list: list of points [[x, y, z], ...]
    """
    pts = []
    for i in range(msg.width):
        offset = msg.point_step * i
        x, y, z = struct.unpack("fff", msg.data[offset : offset + 12])
        pts.append([x, y, z])
    return pts


def generate_static_transform_msg():
    """Generate static transform message from base_link to map.

    Returns:
        TransformStamped: static transform message
    """
    msg = TransformStamped()
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()
    msg.header.frame_id = "map"
    msg.child_frame_id = "base_link"
    msg.transform.translation.x = 0.0
    msg.transform.translation.y = 0.0
    msg.transform.translation.z = 0.0
    msg.transform.rotation.x = 0.0
    msg.transform.rotation.y = 0.0
    msg.transform.rotation.z = 0.0
    msg.transform.rotation.w = 1.0
    return msg



# Geometry probe: a wall 10 m ahead must give occupied cells at the wall, free cells between the
# sensor and the wall, and unknown cells behind it and where no ray passes.
class TestGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_wall(self):
        node = rclpy.create_node("ogm_geometry_probe")
        br = StaticTransformBroadcaster(node)
        br.sendTransform(generate_static_transform_msg())
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)
        buf = []
        node.create_subscription(OccupancyGrid, "/occupancy_grid", lambda m: buf.append(m), qos_profile=qos)
        pub_raw = node.create_publisher(PointCloud2, INPUT_TOPIC_RAW, 10)
        pub_obs = node.create_publisher(PointCloud2, INPUT_TOPIC_OBSTACLE, 10)
        time.sleep(3)
        wall = [[10.1, y * 0.05, 0.5] for y in range(-60, 61)]
        for k in range(5):
            m = get_pointcloud_msg(wall)
            pub_raw.publish(m)
            pub_obs.publish(m)
            rclpy.spin_once(node, timeout_sec=2.0)
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.5)
        self.assertGreater(len(buf), 0)
        g = buf[-1]
        res, ox, oy, w = g.info.resolution, g.info.origin.position.x, g.info.origin.position.y, g.info.width

        def cell(x, y):
            return g.data[int((y - oy) / res) * w + int((x - ox) / res)]

        free = [cell(x * 0.5 + 0.25, 0.25) for x in range(4, 18)]  # 2 m .. 9 m ahead
        occ = [cell(10.25, y * 0.5 + 0.25) for y in range(-4, 4)]
        behind = [cell(x * 0.5 + 0.25, 0.25) for x in range(24, 40)]  # 12 m .. 20 m
        side = [cell(0.25, y * 0.5 + 0.25) for y in range(20, 40)]  # 10 m .. 20 m to the left
        print("grid", w, g.info.height, res, ox, oy, "frames", len(buf))
        print("free", free)
        print("occupied", occ)
        print("behind", behind)
        print("side", side)
        self.assertTrue(all(v < 50 for v in free), free)
        self.assertTrue(all(v > 50 for v in occ), occ)
        self.assertTrue(all(v == behind[0] for v in behind) and min(free) < behind[0] < min(occ), behind)
        self.assertTrue(all(v == side[0] for v in side), side)
        node.destroy_node()
