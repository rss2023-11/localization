from math import *
import numpy as np
import rospy

class MotionModel:

    def __init__(self):
        self.deterministic = rospy.get_param("~deterministic")
        self._last_odometry_update_time = None

    def evaluate(self, particles, odometry_msg):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            None (It modifies the particles array in place)
        """
        #test git functions
        if self._last_odometry_update_time is None:
            self._last_odometry_update_time = rospy.get_time()
        time = rospy.get_time()
        dt = time - self._last_odometry_update_time
        self._last_odometry_update_time = time

        translation = odometry_msg.twist.twist.linear
        rotation = odometry_msg.twist.twist.angular
        covariance = np.array(odometry_msg.twist.covariance).reshape((6, 6))
        covariance = covariance[np.ix_((0, 1, 5), (0, 1, 5))]
        odometry_mean = np.array([translation.x, translation.y, rotation.z])

        output = np.zeros((particles.shape[0], 3))

        random_odometries = np.random.multivariate_normal(odometry_mean, covariance, size=(len(particles),))

        if not self.deterministic:
            random_odometries += np.random.multivariate_normal([0.0, 0.0, 0.0], [[0.5 ** 2, 0, 0],
                                                                                 [0, 0.2 ** 2, 0],
                                                                                 [0, 0, 0.5 ** 2]], size=(len(particles),))
        #print(random_odometries)
        random_odometries *= dt
        

        thetas = particles[:, 2]
        # An N x 2 x 2 array of rotation matrices
        rot_matrices = np.array([[np.cos(thetas), -np.sin(thetas)],
                                 [np.sin(thetas), np.cos(thetas)]]).transpose(2, 0, 1)
        particles[:, :2] += np.einsum('ijk,ik->ij', rot_matrices, random_odometries[:, :2])
        particles[:, 2] += random_odometries[:, 2]