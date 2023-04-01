#!/usr/bin/env python2

import rospy
import numpy as np
from tf.transformations import euler_from_quaternion
from sensor_model import SensorModel
from motion_model import MotionModel

from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, Pose, Point, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler

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
                                          self.initialize_particles,
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
        self.particles = None

    def initialize_particles(self, initial_pose):
        '''
        Initialize all particles as particle from RViz input.

        args:
            initial_pose: An object of type PoseWithCovarianceStamped representing the pose to initialize using
        '''
        # extract coordinates of particle
        x = initial_pose.pose.pose.position[0]
        y = initial_pose.pose.pose.position[1]
        yaw = euler_from_quaternion(initial_pose.pose.pose.orientation)[2]
        mean_position = np.array([[x, y, yaw]])

        covariance = np.array(initial_pose.pose.covariance).reshape((6, 6))
        # Only keep the relevant rows: the ones corresponding to xy position and rotation about the z axis
        covariance = covariance[(0, 1, 5), (0, 1, 5)]

        # Create the particles
        self.particles = np.random.multivariate_normal(mean_position, covariance, (self.num_particles,))

    def odometry_update(self, odometry_msg):
        '''
        Callback for motino model which updates particle positions and publishes average position.

        args:
            odometry_msg: An object of type Odometry representing the odometry message
        '''
        if self.particles is None:
            raise Exception("Particles not initialized. Provide an initial pose through the /initialpose topic before using the odometry callback")
        pass
        # extract change in position and angle (labeled twist) from published Odometry message 
        odometry = odometry_msg.twist.twist.linear
        # update particles with new odometry information
        self.particles = self.motion_model.evaluate(self.particles, odometry)
        # publish particle position
        self.publish_average_particle()
        
    def sensor_update(self, laser_scan):
        '''
        Callback for sensor model which determines particle likelihoods and publishes average position.

        args:
            laser_scan: An object of type LaserScan representing the LaserScan message
        '''
        if self.particles is None:
            raise Exception("Particles not initialized. Provide an initial pose through the /initialpose topic before using the odometry callback")
        pass
        # determine particle likelihoods with sensor model
        likelihoods = self.sensor_model.evaluate(self.particles, laser_scan)
        # resample particles based on likelihoods
        self.particles = np.random.choice(self.particles, self.particles.shape[0], likelihoods)
        # publish particle position
        self.publish_average_particle()

    def publish_average_particle(self):
        '''
        Publish average particle position.

        Using sqrt(M) random particles compared with sqrt(M) other random particles
        and choosing "best" particle as one from first group with min median dist
        from other group.
        '''
        N = int(self.particles.shape[0] ** 0.5)
        base_particles = np.random.choice(self.particles, N)
        other_particles = np.random.choice(self.particles, (N, N,))
        distances = ((base_particles - other_particles) ** 2).sum(axis=-1)
        median_distances = distances.median(axis=0)
        best_index = np.argmax(median_distances)
        best_particle = base_particles[best_index]

        position = Odometry(
            header = Header(frame_id = self.map_frame),
            child_frame_id = self.particle_filter_frame,
            pose = PoseWithCovariance(pose = Pose(
                position = Point(x=best_particle[0], y=best_particle[1], z=0),
                orientation = Quaternion(quaternion_from_euler(0, 0, best_particle[2]))
            ))
        )
        self.odom_pub.publish(position)

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
