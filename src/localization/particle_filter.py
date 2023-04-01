#!/usr/bin/env python2

import rospy
import numpy as np
from tf.transformations import euler_from_quaternion
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped


class ParticleFilter:
    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self._laser_scan_callback, # TODO: Fill this in
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self._odom_callback, # TODO: Fill this in
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self._initial_pose_callback,
                                          queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        self.num_particles = rospy.get_param("~num_particles", 200)
        self.particles = None

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def _initial_pose_callback(self, stamped_pose):
        """
        Given an initial pose `pose` of type PoseWithCovarianceStamped, initialize our
        pose estimate.
        """
        # Extract the pose and covariance matrix
        position = stamped_pose.pose.pose.position
        xy_position = position[:2]
        orientation = stamped_pose.pose.pose.orientation
        z_rotation = euler_from_quaternion(orientation)[2]

        mean_position =  np.array([xy_position[0], xy_position[1], z_rotation])
        covariance = np.array(stamped_pose.pose.covariance).reshape((6, 6))
        # Only keep the relevant rows: the ones corresponding to xy position and rotation about the z axis
        covariance = covariance[(0, 1, 5), (0, 1, 5)]

        # Create the particles
        self.particles = np.random.multivariate_normal(mean_position, covariance, (self.num_particles,))

    def _laser_scan_callback(self, laser_scan):
        """
        Given a `laser_scan` of type LaserScan, update our particles to match what we see better
        """
        if self.particles is None:
            raise Exception("Particles not initialized. Provide an initial pose through the /initialpose topic before using the laser callback")
        pass

    def _odom_callback(self, odometry):
        """
        Given odometry information `odometry`, update our particles to track that we just moved
        """
        if self.particles is None:
            raise Exception("Particles not initialized. Provide an initial pose through the /initialpose topic before using the odometry callback")
        pass

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
