#!/usr/bin/env python3
import sys
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import time
from collections import deque

from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from kiss_icp.registration import Registration
from kiss_icp.mapping import VoxelHashMap
from kiss_icp.voxelization import voxel_down_sample


def pointcloud2_to_numpy(msg):
    field_map = {f.name: f for f in msg.fields}
    if not all(k in field_map for k in ('x', 'y', 'z')):
        return np.empty((0, 3))

    point_step = msg.point_step
    num_points = msg.width * msg.height
    if num_points == 0:
        return np.empty((0, 3))

    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(num_points, point_step)

    ox = field_map['x'].offset
    oy = field_map['y'].offset
    oz = field_map['z'].offset

    x = raw[:, ox:ox+4].view(np.float32).reshape(-1)
    y = raw[:, oy:oy+4].view(np.float32).reshape(-1)
    z = raw[:, oz:oz+4].view(np.float32).reshape(-1)

    points = np.column_stack([x, y, z]).astype(np.float64)
    valid = np.isfinite(points).all(axis=1)
    return points[valid]


def numpy_to_pointcloud2(points, frame_id, stamp):
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.height = 1
    msg.width = len(points)
    msg.row_step = 12 * len(points)
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    return msg


def rot_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


class KissIcpSlamNode(Node):

    def __init__(self):
        super().__init__('kiss_icp_slam_node')

        self.voxel_size = 0.1
        self.max_range  = 10.0
        self.min_range  = 0.3

        self.local_map = VoxelHashMap(
            voxel_size=self.voxel_size,
            max_distance=self.max_range,
            max_points_per_voxel=20
        )
        self.icp = Registration(
            max_num_iterations=500,
            convergence_criterion=0.0001
        )

        self.pose = np.eye(4)
        self.scan_count = 0
        self.global_map_chunks = []
        self.trajectory = []
        self.proc_times = deque(maxlen=50)
        self.last_log = time.time()

        self.tf_broadcaster = TransformBroadcaster(self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub = self.create_subscription(
            PointCloud2, '/pointcloud2', self._cb, sensor_qos
        )

        self.odom_pub      = self.create_publisher(Odometry,      '/kiss_icp/odometry',   10)
        self.local_pub     = self.create_publisher(PointCloud2,   '/kiss_icp/local_map',  sensor_qos)
        self.global_pub    = self.create_publisher(PointCloud2,   '/kiss_icp/global_map', sensor_qos)
        self.path_pub      = self.create_publisher(Path,          '/kiss_icp/path',       10)

        self.create_timer(5.0, self._publish_global_map)

        self.get_logger().info('KISS-ICP SLAM node started')
        self.get_logger().info(f'voxel_size={self.voxel_size} min_range={self.min_range} max_range={self.max_range}')

    def _cb(self, msg):
        t0 = time.time()

        points = pointcloud2_to_numpy(msg)
        if len(points) < 10:
            return

        dist = np.linalg.norm(points[:, :2], axis=1)
        mask = (dist >= self.min_range) & (dist <= self.max_range)
        points = points[mask]
        if len(points) < 10:
            return

        frame = voxel_down_sample(points, self.voxel_size * 0.5)

        if self.local_map.empty():
            self.local_map.add_points(frame)
            self.global_map_chunks.append(frame.copy())
            self.trajectory.append(self.pose.copy())
            self.scan_count += 1
            self.get_logger().info(f'First scan: {len(frame)} points added to map')
            return

        self.pose = self.icp.align_points_to_map(
            points=frame,
            voxel_map=self.local_map,
            initial_guess=self.pose,
            max_correspondance_distance=self.voxel_size * 3.0,
            kernel=self.voxel_size * 0.5
        )

        self.local_map.update(frame, self.pose)
        self.local_map.remove_far_away_points(self.pose[:3, 3])

        R = self.pose[:3, :3]
        t = self.pose[:3, 3]
        points_world = (R @ frame.T).T + t
        self.global_map_chunks.append(points_world)
        self.trajectory.append(self.pose.copy())

        elapsed = (time.time() - t0) * 1000.0
        self.proc_times.append(elapsed)
        self.scan_count += 1

        stamp = msg.header.stamp
        self._publish_odom(stamp)
        self._publish_tf(stamp)
        self._publish_local_map(stamp)
        self._publish_path(stamp)

        if time.time() - self.last_log > 5.0:
            pos = self.pose[:3, 3]
            avg = np.mean(self.proc_times)
            total = sum(len(c) for c in self.global_map_chunks)
            self.get_logger().info(
                f'scan={self.scan_count} icp={avg:.1f}ms '
                f'pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) '
                f'map_pts={total}'
            )
            self.last_log = time.time()

    def _publish_odom(self, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = float(self.pose[0, 3])
        msg.pose.pose.position.y = float(self.pose[1, 3])
        msg.pose.pose.position.z = float(self.pose[2, 3])
        q = rot_to_quat(self.pose[:3, :3])
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])
        self.odom_pub.publish(msg)

    def _publish_tf(self, stamp):
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link_slam'
        tf.transform.translation.x = float(self.pose[0, 3])
        tf.transform.translation.y = float(self.pose[1, 3])
        tf.transform.translation.z = float(self.pose[2, 3])
        q = rot_to_quat(self.pose[:3, :3])
        tf.transform.rotation.x = float(q[0])
        tf.transform.rotation.y = float(q[1])
        tf.transform.rotation.z = float(q[2])
        tf.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(tf)

    def _publish_local_map(self, stamp):
        if not self.local_pub.get_subscription_count():
            return
        pts = self.local_map.point_cloud()
        if len(pts) == 0:
            return
        self.local_pub.publish(numpy_to_pointcloud2(pts, 'odom', stamp))

    def _publish_path(self, stamp):
        if not self.path_pub.get_subscription_count():
            return
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = 'odom'
        for p in self.trajectory[-500:]:
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = 'odom'
            ps.pose.position.x = float(p[0, 3])
            ps.pose.position.y = float(p[1, 3])
            ps.pose.position.z = float(p[2, 3])
            q = rot_to_quat(p[:3, :3])
            ps.pose.orientation.x = float(q[0])
            ps.pose.orientation.y = float(q[1])
            ps.pose.orientation.z = float(q[2])
            ps.pose.orientation.w = float(q[3])
            path.poses.append(ps)
        self.path_pub.publish(path)

    def _publish_global_map(self):
        if not self.global_pub.get_subscription_count():
            return
        if not self.global_map_chunks:
            return
        all_pts = np.vstack(self.global_map_chunks)
        downsampled = voxel_down_sample(all_pts, self.voxel_size * 2.0)
        stamp = self.get_clock().now().to_msg()
        self.global_pub.publish(numpy_to_pointcloud2(downsampled, 'odom', stamp))
        self.get_logger().info(f'Global map published: {len(downsampled)} pts')

    def get_global_map(self):
        if not self.global_map_chunks:
            return np.empty((0, 3))
        all_pts = np.vstack(self.global_map_chunks)
        return voxel_down_sample(all_pts, self.voxel_size)


def main(args=None):
    rclpy.init(args=args)
    node = KissIcpSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping...')
        global_map = node.get_global_map()
        if len(global_map) > 0:
            import os
            os.makedirs('/workspace/humanoid-motion-planning/maps', exist_ok=True)
            path = '/workspace/humanoid-motion-planning/maps/slam_map.npy'
            np.save(path, global_map)
            node.get_logger().info(f'Map saved: {path} ({len(global_map)} pts)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
