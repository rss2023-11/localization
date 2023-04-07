#!/usr/bin/env python2

import rospy
import numpy as np
from scipy.interpolate import interp1d
import scipy.special

from tf.transformations import euler_from_quaternion
from sensor_model import SensorModel
from motion_model import MotionModel

from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, PoseStamped, Pose, Point, Quaternion, PoseArray, Vector3
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import math


class ParticleFilter:
    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame", "base_link_pf")
        self.map_frame = rospy.get_param("~map_frame", "/map")
        self.num_particles = rospy.get_param("~num_particles", 200)
        self.particles = None
        self.log_weights = None

        # Initialize publishers/subscribers
        #
        #  Important Note #1: It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and not use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.sensor_update,
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odometry_update,
                                          queue_size=1)

        #  Important Note #2: You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.initialize_particles,
                                          queue_size=1)

        #  Important Note #3: You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        self.particles_pub  = rospy.Publisher("/pf/pose/particles", PoseArray, queue_size=1)

        self.slime_pub = rospy.Publisher("/slime", Path, queue_size=20)

        
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        self.num_particles = rospy.get_param("~num_particles", 200)

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self._last_odometry_update_time = None
        self.trail = Path()
        self.trail.header.frame_id = self.map_frame
        self.trail.header.stamp = rospy.Time.now()
        self.count = 0

        ## Used for interpolating our estimate of the average to smooth it out
        self._past_averages = None # Past five averages

    def initialize_particles(self, initial_pose):
        '''
        Initialize all particles as particle from RViz input.

        args:
            initial_pose: An object of type PoseWithCovarianceStamped representing the pose to initialize using
        '''
        # extract coordinates of particle
        x = initial_pose.pose.pose.position.x
        y = initial_pose.pose.pose.position.y

        # use offsets for testing
        x_offset = rospy.get_param("~x_offset", 0)
        y_offset = rospy.get_param("~y_offset", 0)

        orientation = initial_pose.pose.pose.orientation
        quat = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]
        yaw = euler_from_quaternion(quat)[2]
        mean_position = np.array([x + x_offset, y + y_offset, yaw])

        covariance = np.identity(3) * rospy.get_param("~position_variance", 1)
        covariance[-1, -1] = rospy.get_param("~angle_variance", (math.pi/8)**2)
        print(covariance)

        # Create the particles
        self.particles = np.random.multivariate_normal(mean_position, covariance, (self.num_particles,))
        self.log_weights = np.ones((self.num_particles,)) / np.log(self.num_particles)

        # Reset the odometry update time
        self._last_odometry_update_time = None

        # Set the moving average
        self._past_averages = [mean_position,] * 5

    def odometry_update(self, odometry_msg):
        '''
        Callback for motino model which updates particle positions and publishes average position.

        args:
            odometry_msg: An object of type Odometry representing the odometry message
        '''
        if self.particles is None:
            return
            #raise Exception("Particles not initialized. Provide an initial pose through the /initialpose topic before using the odometry callback")

        if self._last_odometry_update_time is None:
            self._last_odometry_update_time = rospy.get_time()
            return
        time = rospy.get_time()
        dt = time - self._last_odometry_update_time
        self._last_odometry_update_time = time

        # extract change in position and angle (labeled twist) from published Odometry message 
        #print(odometry_msg)
        translation = odometry_msg.twist.twist.linear
        rotation = odometry_msg.twist.twist.angular
        odometry = np.array([translation.x, translation.y, rotation.z]) * dt
        #print(odometry)
        # update particles with new odometry information
        self.particles = self.motion_model.evaluate(self.particles, odometry)
        # publish particle position
        self.publish_average_particle()
        self.publish_particles()
        
    def sensor_update(self, laser_scan):
        '''
        Callback for sensor model which determines particle likelihoods and publishes average position.

        args:
            laser_scan: An object of type LaserScan representing the LaserScan message
        '''
        if self.particles is None:
            return
            #raise Exception("Particles not initialized. Provide an initial pose through the /initialpose topic before using the odometry callback")

        # determine particle likelihoods with sensor model
        downsampled = self.sensor_model.downsample(laser_scan)
        likelihoods = self.sensor_model.evaluate(self.particles, downsampled)

        self._update_particles_with_likelihoods(likelihoods)

        # publish particle position
        self.publish_average_particle()
        self.publish_particles()

    def publish_average_particle(self):
        '''
        Publish average particle position and add to slime trail.
        '''
        mean_particle = self._get_mean_pose() 
        quaternion = quaternion_from_euler(0, 0, mean_particle[2])
        average_pose = Pose(
            position=Point(x=mean_particle[0], y=mean_particle[1], z=0),
            orientation=Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
        )
        position = Odometry(
            header = Header(frame_id = self.map_frame, stamp = rospy.Time.now()),
            child_frame_id = self.particle_filter_frame,
            pose = PoseWithCovariance(pose=average_pose)
        )
        self.odom_pub.publish(position)

        poseStamped = PoseStamped(
            header = Header(frame_id = self.map_frame, stamp = rospy.Time.now()),
            pose = average_pose
        )
        self.trail.poses.append(poseStamped)
        self.slime_pub.publish(self.trail)
        # rospy.loginfo("Published {} points for slime trail.".format(len(self.trail.poses)))

    def publish_particles(self):
        '''
        Publish a PoseArray of all of our particles
        '''

        poses = [self._get_pose(particle) for particle in self.particles]
        pose_array = PoseArray(
            header=Header(frame_id=self.map_frame),
            poses=poses
        )
        self.particles_pub.publish(pose_array)

    def _get_pose(self, particle):
        '''
        Given a particle in our list of particles, return a Pose object for it
        '''
        quaternion = quaternion_from_euler(0, 0, particle[2])
        return Pose(
            position=Point(particle[0], particle[1], 0),
            orientation=Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
        )

    def _get_mean_pose(self):
        angles = self.particles[:, 2]
        angle_positions = np.vstack([np.cos(angles), np.sin(angles)])
        mean_angle_position = np.mean(angle_positions, axis=1)
        mean_angle = math.atan2(mean_angle_position[1], mean_angle_position[0])
        mean_position = np.mean(self.particles[:, 0:2], axis=0)
        mean_pose = np.array([mean_position[0], mean_position[1], mean_angle])


        self._past_averages = self._past_averages[1:] + [mean_pose]
        interpolator =  interp1d([0, 1, 2, 3, 4], self._past_averages, kind='cubic', axis=0)
        interpolated_point = interpolator(4)

        return interpolated_point
    
    def _update_particles_with_likelihoods(self, likelihoods):
        '''
        Given a 1d array of likelihoods with length equal to num_particles,
        update the weights of the particles with these likelihoods and resample
        if the entropy is too low.
        '''

        # Multiply all the weights with their likelihoods (or, rather, add the log of the weights
        # with the log of their likelihoods)
        log_likelihoods = np.log(likelihoods)
        self.log_weights += log_likelihoods
        self.log_weights -= scipy.special.logsumexp(self.log_weights) # Renormalize so the sum of weights is 1

        # Calculate the entropy, and if it is less than half that of a uniform distribution resample to
        # even out our weights
        entropy = -np.sum(np.exp(self.log_weights) * self.log_weights)
        uniform_entropy = math.log(self.num_particles)
        if entropy < uniform_entropy / 2:
            # Resample based on weights:
            self.particles = self._sample_particles(
                self.particles.shape[0],
                np.exp(self.log_weights)
            )
            self.log_weights = np.ones((self.num_particles,)) / np.log(self.num_particles)

    def _sample_particles(self, shape, likelihoods=None):
        '''
        Sample from our particles and return an array of the given shape
        '''
        if likelihoods is None:
            idx = np.random.randint(self.particles.shape[0], size=shape)
        else:
            idx = np.random.choice(list(range(self.particles.shape[0])), size=shape, p=likelihoods)

        return self.particles[idx]


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
