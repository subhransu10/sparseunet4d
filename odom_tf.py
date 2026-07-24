import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class OdomTF(Node):
    def __init__(self):
        super().__init__("odom_tf_broadcaster")
        self.br = TransformBroadcaster(self)
        self.create_subscription(Odometry, "/platform/odom/filtered", self.cb, 10)
    def cb(self, m):
        t = TransformStamped()
        t.header = m.header                     # frame_id = odom
        t.child_frame_id = m.child_frame_id     # base_link
        t.transform.translation.x = m.pose.pose.position.x
        t.transform.translation.y = m.pose.pose.position.y
        t.transform.translation.z = m.pose.pose.position.z
        t.transform.rotation = m.pose.pose.orientation
        rclpy.logging.get_logger("odom_tf").info("broadcasting odom->base_link", once=True)
        self.br.sendTransform(t)

rclpy.init(); rclpy.spin(OdomTF()); rclpy.shutdown()