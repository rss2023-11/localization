#!/usr/bin/env python2

import rospy
import numpy as np
from scipy.interpolate import interp1d
import scipy.special

from sensor_model import SensorModel
from motion_model import MotionModel

from std_msgs.msg import Header, ColorRGBA, Float32
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, PoseStamped, Pose, Point, Quaternion, PoseArray, Vector3, TransformStamped

import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray
from threading import Lock

import numpy as np
import math


class ParticleFilter:
    def __init__(self):
        # Get parameters
        self.num_particles = rospy.get_param("~num_particles", 200)
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame", "base_link")
        self.map_frame = rospy.get_param("~map_frame", "/map")
        self.num_particles = rospy.get_param("~num_particles", 200)
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")


        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        self.particles_pub  = rospy.Publisher("/pf/pose/particles", PoseArray, queue_size=1)

        self.slime_pub = rospy.Publisher("/slime", Path, queue_size=20)
        # To publish std. dev. of particles
        self.stddev_pub = rospy.Publisher("std_dev", Float32, queue_size=1)

        self.broadcaster = tf.TransformBroadcaster()
        
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
        self._last_odometry_update_time = None
        self.trail = Path()
        self.trail.header.frame_id = self.map_frame
        self.trail.header.stamp = rospy.Time.now()
        self.count = 0
        self.mutex = Lock()

        init_pose = Pose(
            position=Point(x=-19.73, y=3.60, z=0),
            orientation=Quaternion(x=0, y=0, z=-0.8125, w=0.5829)
        )
        pose_stamped = PoseWithCovarianceStamped(
            pose=PoseWithCovariance(pose=init_pose), 
            header = Header(frame_id = self.map_frame, stamp = rospy.Time.now())
        )
        self.particles = self.initialize_particles(pose_stamped)
        self.log_weights = None
        ## Used for interpolating our estimate of the average to smooth it out
        self._past_averages = None # Past five averages


        # Initialize publishers/subscribers
        #
        #  Important Note #1: It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and not use the pose component.
        #  Important Note #2: You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        
        #  Important Note #3: You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.sensor_update,
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odometry_update,
                                          queue_size=1)
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.initialize_particles,
                                          queue_size=1)

    def initialize_particles(self, initial_pose):
        '''
        Initialize all particles as particle from RViz input.

        args:
            initial_pose: An object of type PoseWithCovarianceStamped representing the pose to initialize using
        '''
        self.mutex.acquire()
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
        # print(covariance)

        # Create the particles
        self.particles = np.random.multivariate_normal(mean_position, covariance, (self.num_particles,))
        self.log_weights = np.ones((self.num_particles,)) / np.log(self.num_particles)

        # Reset slime trail
        self.trail = Path()
        self.trail.header.frame_id = self.map_frame
        self.trail.header.stamp = rospy.Time.now()

        # Reset the odometry update time
        self._last_odometry_update_time = None

        # Set the moving average
        self._past_averages = [mean_position,] * 5
        self.mutex.release()


    def odometry_update(self, odometry_msg):
        '''
        Callback for motion model which updates particle positions and publishes average position.

        args:
            odometry_msg: An object of type Odometry representing the odometry message
        '''
        if self.particles is None:
            return
        # rospy.loginfo("ODOMETRY UPDATING")
        self.mutex.acquire()
        # update particles with new odometry information
        self.motion_model.evaluate(self.particles, odometry_msg)
        # publish particle position
        self.publish_average_particle()
        self.publish_particles()
        self.mutex.release()

    def sensor_update(self, laser_scan):
        '''
        Callback for sensor model which determines particle likelihoods and publishes average position.

        args:
            laser_scan: An object of type LaserScan representing the LaserScan message
        '''
        if self.particles is None:
            return
            #raise Exception("Particles not initialized. Provide an initial pose through the /initialpose topic before using the odometry callback")

        self.mutex.acquire()
        # determine particle likelihoods with sensor model
        downsampled = self.sensor_model.downsample(laser_scan)
        likelihoods = self.sensor_model.evaluate(self.particles, downsampled)

        self._update_particles_with_likelihoods(likelihoods)

        # publish particle position
        self.publish_average_particle()
        self.publish_particles()
        
        self.mutex.release()

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
        # publish average particle position
        # rospy.loginfo("CURRENT POSITION")
        # rospy.loginfo(position.pose.pose)
        
        self.odom_pub.publish(position)

        # publish std. dev. publishing
        std_devs = np.std(self.particles, axis=0)
        std_dev_xy = math.sqrt(std_devs[0]**2 + std_devs[1]**2)
        self.stddev_pub.publish(std_dev_xy)

        # publish trail for visualization
        poseStamped = PoseStamped(
            header = Header(frame_id = self.map_frame, stamp = rospy.Time.now()),
            pose = average_pose
        )
        self.trail.poses.append(poseStamped)
        self.slime_pub.publish(self.trail)

        # Publish the transform
        # Create a TransformStamped message
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.particle_filter_frame
        transform.transform.translation.x = mean_particle[0]
        transform.transform.translation.y = mean_particle[1]
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        self.broadcaster.sendTransform(
            (transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z),
            (transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w),
            transform.header.stamp,
            transform.child_frame_id,
            transform.header.frame_id
        )

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
        mean_angle_position = np.average(angle_positions, axis=1, weights=np.exp(self.log_weights))
        mean_angle = math.atan2(mean_angle_position[1], mean_angle_position[0])
        mean_position = np.average(self.particles[:, 0:2], axis=0, weights=np.exp(self.log_weights))
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
                np.exp(self.log_weights),
                noise_covariance=[[0.05 ** 2, 0, 0],
                                  [0, 0.02 ** 2, 0],
                                  [0, 0, 0.05 ** 2]]
            )
            self.log_weights = np.ones((self.num_particles,)) / np.log(self.num_particles)

    def _sample_particles(self, shape, likelihoods=None, noise_covariance=None):
        '''
        Sample from our particles and return an array of the given shape
        '''
        if likelihoods is None:
            idx = np.random.randint(self.particles.shape[0], size=shape)
        else:
            idx = np.random.choice(list(range(self.particles.shape[0])), size=shape, p=likelihoods)

        new_particles = self.particles[idx]
        if noise_covariance is not None:
            new_particles += np.random.multivariate_normal([0.0, 0.0, 0.0], noise_covariance, size=shape)

        return new_particles


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
