#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, Pose, Position, Quaternion
from tf.transformations import euler_from_quaternion

import numpy as np
import math


class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")
        self.map_frame = rospy.get_param("~map_frame")
        self.num_particles = rospy.get_param("~num_particles")

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
                                          self.sensor_update,
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odometry_update,
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.set_initial_pose,
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

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.particles = None

    def initialize_particles(self, initial_pose: PoseWithCovarianceStamped):
        '''
        Initialize all particles as particle from RViz input.
        '''
        # extract coordinates of particle
        x = initial_pose.pose.pose.position[0]
        y = initial_pose.pose.pose.position[1]
        yaw = euler_from_quaternion(initial_pose.pose.pose.orientation)[2]
        position = np.array([[x, y, yaw]])
        self.particles = np.repeat(position, self.num_particles, axis=0)

    def odometry_update(self, odometry_msg: Odometry):
        '''
        Callback for motino model which updates particle positions and publishes average position.
        '''
        # extract change in position and angle (labeled twist) from published Odometry message 
        odometry = odometry_msg.twist.twist.linear
        # update particles with new odometry information
        self.particles = self.motion_model.evaluate(self.particles, odometry)
        # publish particle position
        self.publish_average_particle()
        
    def sensor_update(self, laser_scan: LaserScan):
        '''
        Callback for sensor model which determines particle likelihoods and publishes average position.
        '''
        # determine particle likelihoods with sensor model
        likelihoods = self.sensor_model.evaluate(self.particles, laser_scan)
        # resample particles based on likelihoods
        self.particles = np.random.choice(self.particles, self.particles.shape[0], likelihoods)
        # publish particle position
        self.publish_average_particle()

    def publish_average_particle(self):
        '''
        Publish average particle position.
        '''
        position = Odometry(
            header = Header(frame_id = self.map_frame),
            child_frame_id = self.particle_filter_frame,
            pose = PoseWithCovariance(pose = Pose(
                position = Position(),
                orientation = Quaternion()
            ))
        )

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
