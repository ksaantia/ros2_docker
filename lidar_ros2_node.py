import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Twist
import sensor_msgs.msg as sensor_msgs
import tf2_ros
import numpy as np
import mujoco
import threading
import time
import math
from lidar_raycaster import LidarRaycaster

XML_PATH    = "test_corridor.xml"
SITE_NAME   = "lidar_site"
PUBLISH_HZ  = 10.0
SIM_STEP_HZ = 200.0


def numpy_to_pointcloud2(points, frame_id, stamp):
    fields = [
        sensor_msgs.PointField(name='x', offset=0,  datatype=sensor_msgs.PointField.FLOAT32, count=1),
        sensor_msgs.PointField(name='y', offset=4,  datatype=sensor_msgs.PointField.FLOAT32, count=1),
        sensor_msgs.PointField(name='z', offset=8,  datatype=sensor_msgs.PointField.FLOAT32, count=1),
    ]
    msg = sensor_msgs.PointCloud2()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height       = 1
    msg.width        = len(points)
    msg.fields       = fields
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = 12 * len(points)
    msg.is_dense     = True
    msg.data         = points.tobytes()
    return msg


class LidarNode(Node):

    def __init__(self):
        super().__init__('mujoco_lidar_node')

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.lidar = LidarRaycaster(
            self.model, self.data,
            site_name=SITE_NAME,
            exclude_body=None,
        )

        self.sim_lock = threading.Lock()

        self.base_x   = 0.0
        self.base_y   = 0.0
        self.base_yaw = 0.0
        self.cmd_vx   = 0.0
        self.cmd_wz   = 0.0

        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'lidar_body'
        )

        self.pc_pub = self.create_publisher(sensor_msgs.PointCloud2, '/pointcloud2', 10)
        self.tf_broadcaster        = tf2_ros.TransformBroadcaster(self)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._publish_static_tf()

        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_callback, 10)
        self.create_timer(1.0 / PUBLISH_HZ, self._timer_callback)

        self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.sim_thread.start()

        self.get_logger().info("Нода запущена. /pointcloud2 на {:.0f} Гц".format(PUBLISH_HZ))
        self.get_logger().info("Телеоп: i=вперёд  ,=назад  j=влево  l=вправо  k=стоп")

    def _publish_static_tf(self):
        now = self.get_clock().now().to_msg()
        transforms = []

        t1 = TransformStamped()
        t1.header.stamp    = now
        t1.header.frame_id = 'map'
        t1.child_frame_id  = 'odom'
        t1.transform.rotation.w = 1.0
        transforms.append(t1)

        t2 = TransformStamped()
        t2.header.stamp    = now
        t2.header.frame_id = 'base_link'
        t2.child_frame_id  = 'lidar_link'
        t2.transform.translation.z = 1.357
        t2.transform.rotation.w = 1.0
        transforms.append(t2)

        self.static_tf_broadcaster.sendTransform(transforms)

    def _cmd_vel_callback(self, msg):
        self.cmd_vx = msg.linear.x
        self.cmd_wz = msg.angular.z

    def _sim_loop(self):
        dt = 1.0 / SIM_STEP_HZ
        while True:
            t0 = time.perf_counter()
            with self.sim_lock:
                self.base_yaw += self.cmd_wz * dt
                self.base_x   += self.cmd_vx * math.cos(self.base_yaw) * dt
                self.base_y   += self.cmd_vx * math.sin(self.base_yaw) * dt

                self.model.body_pos[self.body_id][0] = self.base_x
                self.model.body_pos[self.body_id][1] = self.base_y
                mujoco.mj_forward(self.model, self.data)

            elapsed = time.perf_counter() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _timer_callback(self):
        with self.sim_lock:
            points = self.lidar.scan()
            x   = self.base_x
            y   = self.base_y
            yaw = self.base_yaw

        stamp = self.get_clock().now().to_msg()

        pc_msg = numpy_to_pointcloud2(points, frame_id='lidar_link', stamp=stamp)
        self.pc_pub.publish(pc_msg)

        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_link'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info(
            "points={} x={:.2f} y={:.2f} yaw={:.2f}".format(len(points), x, y, yaw),
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
