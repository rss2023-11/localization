import numpy as np
from localization.scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

import rospy
import math
import tf
from scipy import interpolate
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale")

        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0
        self.z_max = 200.0 #I assume it's 200 bc that's the max possible distance we get?

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
            unscaled = (zi >= 0) * (zi <= self.z_max) * np.exp(-(zi - d) ** 2 / (2.0 * self.sigma_hit ** 2))
            return unscaled
            
        def p_short(zi, d):
            if d == 0:
                return np.zeros_like(zi)
            return (zi >= 0) * (zi <= d) * 2.0 / (d) * (1 - zi / d)
        
        def p_max(zi, d):
            return 1.0 * (zi == self.z_max)
            
        def p_rand(zi, d):
            return np.ones_like(zi.shape) / (self.z_max)
        
        #fill a hits term array to be normalized, as well as the actual table
        for d in range(self.table_width):
            zi = np.array(range(self.table_width), dtype=float)
            d_float = float(d)
            hit_term = p_hit(zi, d_float)
            hit_term /= np.sum(hit_term)

            short_term = p_short(zi, d_float)
            max_term = p_max(zi, d_float)
            rand_term = p_rand(zi, d_float)

            total = (self.alpha_hit * hit_term +
                     self.alpha_short  * short_term +
                     self.alpha_max * max_term +
                     self.alpha_rand * rand_term)
            total /= np.sum(total)
            self.sensor_model_table[:, d] = total

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

        # Make sure we were passed the correct number of observations
        assert len(observation) == self.num_beams_per_particle

        ####################################
        # TODO
        # Evaluate the sensor model here!
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 
        scans = self.scan_sim.scan(particles)

        scale_factor = self.map_resolution * self.lidar_scale_to_map_scale
        scans /= scale_factor
        observation /= scale_factor
        
        # Convert to integers and clip. The out=... is done to make it in place (saving speed & memory)
        scans = scans.clip(0, self.z_max).round().astype(int)
        observation = observation.clip(0, self.z_max).round().astype(int)

        probs = self.sensor_model_table[observation, scans]
        log_probs = np.log(probs).sum(axis=1) / 2.2 # Squashing factor is 2.2
        return np.exp(log_probs)

        ####################################

    def downsample(self, lidar_scan, num_samples=None, angle_min=None, angle_max=None):
        """
        Given a lidar scan, return a vector of lidar data corresponding to the given number of samples equally
        distributed from `angle_min` to `angle_max` (inclusive). By default, angle_min and angle_max are
        chosen to encompass the entire range of the lidar scan.
        """
        num_samples = num_samples if num_samples is not None else self.num_beams_per_particle
        angle_min = angle_min if angle_min is not None else lidar_scan.angle_min
        angle_max = angle_max if angle_max is not None else lidar_scan.angle_max
        desired_angles = np.linspace(angle_min, angle_max, num_samples)
        
        ranges = np.array(lidar_scan.ranges)
        N = len(ranges)
        angles = np.linspace(
            lidar_scan.angle_min,
            lidar_scan.angle_max,
            N,
        )
        spline = interpolate.splrep(angles, ranges)
        desired_ranges = interpolate.splev(desired_angles, spline)
        return desired_ranges
        

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)
        self.map_resolution = map_msg.info.resolution

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