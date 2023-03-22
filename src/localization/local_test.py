import numpy as np
import math
from math import *

# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

class SensorModel:

    def __init__(self):
        # Fetch parameters

        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 5
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = None
        self.precompute_sensor_model()

    def p(self, d, z):
        zmax = 10.0
        sigma = 0.5
        e = 0.1
        phit = 0 if z<0 or z>zmax else 1/(2*math.pi * sigma**2)**0.5 * math.e**(-((z-d)**2/(2*sigma**2)))
        pshort = 0 if z<1 or z>d else (1-z/d)*2/d
        pmax = 0 if z<(zmax-e) or z>zmax else 1.0/e
        prand = 0 if z<0 or z>zmax else 1.0/zmax

        return self.alpha_hit*phit + self.alpha_short*pshort + self.alpha_max*pmax + self.alpha_rand*prand



    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        # self.sensor_model_table = np.zeros((self.table_width, self.table_width))

        # for d in range(self.table_width):
        #     for z in range(self.table_width):
        #         self.sensor_model_table[d][z] = self.p(d, z)
        # #normalize hits term array
        # norms=np.linalg.norm(hits_terms, axis=0)
        # hits_terms=np.divide(hits_terms,norms)
        # self.sensor_model_table=np.add(self.sensor_model_table,hits_terms)
        
        # #normalize whole table
        # table_norms=np.linalg.norm(self.sensor_model_table, axis=0)
        # self.sensor_model_table=np.divide(self.sensor_model_table, table_norms)
        # print(self.sensor_model_table)

test = SensorModel()
# test.precompute_sensor_model()

class MotionModel:

    def __init__(self):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        self.deterministic = False

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
        
        ####################################
        
        
        output=np.zeros((particles.shape[0], 3))
        
        for r in range(particles.shape[0]):
            r_theta=particles[r][2]
            # print(r_theta)
            row=particles[r].reshape((3,1))
            # print(row)
            #transformation matrix
            trans = np.array([[cos(r_theta), -sin(r_theta), 0],
                              [sin(r_theta), cos(r_theta), 0],
                              [0,0,1]])
            # print(trans, odometry.reshape((3,1)))
            trans=np.matmul(trans, odometry.reshape((3,1)))
            # print(trans)
            #apply the transformation
            new_particle=np.add(row, trans)
            # print(new_particle)
            #add noise
            if self.deterministic==False:
                x_error=np.random.normal(0.0, 0.05)
                y_error=np.random.normal(0.0, 0.02)
                theta_error=np.random.normal(0.0, 0.05)
                
                noise=np.array([x_error, y_error, theta_error]).reshape((3,1))
                new_particle=np.add(noise, new_particle)
                
            output[r] = new_particle.reshape((1,3))
            
        return output
    
        ####################################

particles = np.array([[1, 2, 0.5],
                      [1,2,4]]) #2*3
odometry = np.array([1,1,1]) #1*3
test_motion = MotionModel()
print(test_motion.evaluate(particles, odometry))
