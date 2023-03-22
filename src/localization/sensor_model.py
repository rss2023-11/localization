import numpy as np
from localization.scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

import rospy
import math
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")

        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0
        self.z_max = 200 #I assume it's 200 bc that's the max possible distance we get?

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = np.zeros((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization) 

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)

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
        #fxns for calculating different terms of the probability   
        def p_hit(zi, d):
            if zi>=0 and zi<=d:
                output=self.alpha_hit*1/(math.sqrt(2*math.pi*self.sigma_hit**2))*np.exp(-(zi-d)**2/(2*self.sigma_hit**2))
                return output
            else:
                return 0
            
        def p_short(zi, d):
            if zi>=0 and zi<=d and d!=0:
                output=(2/d)*(1-(zi/d))
                return output
            else:
                return 0
        
        def p_max(zi, d):
            if zi==self.z_max:
                return 1
            else:
                return 0
            
        def p_rand(zi, d):
            if zi>=0 and zi<=self.z_max:
                return 1/self.z_max
            else:
                return 0
        
        hits_terms=np.zeros((self.table_width, self.table_width))
        
        #fill a hits term array to be normalized, as well as the actual table
        for d in range(0,self.table_width):
            for zi in range(0,self.table_width):
                hit_term=p_hit(zi,d)
                short_term=p_short(zi,d)
                max_term=p_max(zi,d)
                rand_term=p_rand(zi,d)
                
                hits_terms[d][zi]=hit_term
                self.sensor_model_table[d][zi]=short_term+max_term+rand_term
                
        #normalize hits term array
        norms=np.linalg.norm(hits_terms, axis=0)
        hits_terms=np.divide(hits_terms,norms)
        self.sensor_model_table=np.add(self.sensor_model_table,hits_terms)
        
        #normalize whole table
        table_norms=np.linalg.norm(self.sensor_model_table, axis=0)
        self.sensor_model_table=np.divide(self.sensor_model_table, table_norms)

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans = self.scan_sim.scan(particles)

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
