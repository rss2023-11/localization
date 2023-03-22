import math
import numpy as np
import rospy

class MotionModel:

    def __init__(self):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

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
        
        ####################################
        # TODO

        #strategy: 
        #apply odometry information to each particle.
        #get the new particle.
        #add noise to the particle measurement somehow?
        #return the list of all the new particles 
        
        od_x=odometry[0]
        od_y=odometry[1]
        od_theta=odometry[2]
        
        output=[]
        print(particles.shape())

        for r in particles:
            r_theta=r[2]
            
            row=np.array(r).reshape((3,1))
            
            #transformation matrix
            trans=np.array([od_x*math.cos(r_theta)-od_y*math.sin(r_theta), od_x*math.sin(r_theta)+od_y*math.cos(r_theta), od_theta]).reshape((3,1))
            print(trans.shape())
            #apply the transformation
            new_particle=np.add(row, trans)
            
            #add noise
            if self.deterministic==False:
                x_error=np.random.normal(0.0, 0.05)
                y_error=np.random.normal(0.0, 0.02)
                theta_error=np.random.normal(0.0, 0.05)
                
                noise=np.array([x_error, y_error, theta_error]).reshape((3,1))
                new_particle=np.add(noise, new_particle)
                
            output.append(list(new_particle))
            
        return output
    
        ####################################

#tester not working--ask about this in OH. "no module named localization.motion_model"