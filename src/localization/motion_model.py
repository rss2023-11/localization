from math import *
import numpy as np
import rospy

class MotionModel:

    def __init__(self):

        ####################################
        # Do any precomputation for the motion
        # model here.

        # TODO
        self.deterministic = rospy.get_param("~deterministic")

        ####################################

    def evaluate(self, particles, odometry):
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
            particles: An updated matrix of the
                same size
        """

        output = np.zeros((particles.shape[0], 3))

        for r in range(particles.shape[0]):
            r_theta = particles[r][2]
            # print(r_theta)
            row = particles[r].reshape((3,1))
            # print(row)
            #transformation matrix
            trans = np.array([[cos(r_theta), -sin(r_theta), 0],
                              [sin(r_theta), cos(r_theta), 0],
                              [0,0,1]])
            # print(trans, odometry.reshape((3,1)))
            trans = np.matmul(trans, odometry.reshape((3,1)))
            # print(trans)
            #apply the transformation
            new_particle = np.add(row, trans)
            # print(new_particle)
            # add noise
            if not self.deterministic:
                x_error = np.random.normal(0.0, 0.05)
                y_error = np.random.normal(0.0, 0.02)
                theta_error = np.random.normal(0.0, 0.05)

                noise = np.array([x_error, y_error, theta_error]).reshape((3,1))
                new_particle = np.add(noise, new_particle)

            output[r] = new_particle.reshape((1,3))

        return output
