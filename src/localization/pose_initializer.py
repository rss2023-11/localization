#!/usr/bin/env python

import rospy
import numpy as np

import tf
from tf.transformations import euler_from_quaternion

from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Point, Quaternion

class PoseInitializer:
    """
    Rosnode for handling simulated cone. Listens for clicked point
    in rviz and publishes a marker. Publishes position of cone
    relative to robot for Parking Controller to park in front of.
    """
    def __init__(self):
        self.map_frame = rospy.get_param("~map_frame")

        # Subscribe to clicked point messages from rviz    
        rospy.Subscriber("/clicked_point", 
            PointStamped, self.clicked_callback)
        self.message_x = None
        self.message_y = None
        self.message_frame = "map"

        self.initialpose_pub = rospy.Publisher("/initialpose", 
            PoseWithCovarianceStamped, queue_size=1)
        self.tf_listener = tf.TransformListener()

    def clicked_callback(self, msg):
        # Store clicked point in the map frame
        msg_frame_pos, msg_frame_quat = self.tf_listener.lookupTransform(
               self.message_frame, msg.header.frame_id, rospy.Time(0))
        
        # message to publish onwards
        message = PoseWithCovarianceStamped()
        message.header.frame_id = self.map_frame
        x, y, z = msg_frame_pos
        message.pose.pose.position = Point(x=x, y=y, z=z)
        x, y, z, w = msg_frame_quat
        message.pose.pose.orientation = Quaternion(x=x, y=y, z=z, w=w)

        # TODO: Put these constants in a param file
        X_STD = 1.0 # uncertainty of the x position in meters
        Y_STD = 1.0 # uncertainty of the y position in meters
        THETA_STD = 2 * np.pi # uncertainty of the angle in radians
        message.pose.covariance = np.array([
            [X_STD ** 2, 0, 0, 0, 0, 0],
            [0, Y_STD ** 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, THETA_STD ** 2],
        ]).ravel()

        self.initialpose_pub.publish(message)
        print("Sent the message to initialize the pose!")

if __name__ == '__main__':
    try:
        rospy.init_node('SimMarker', anonymous=True)
        PoseInitializer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass