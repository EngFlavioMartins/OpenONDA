import numpy as np
import scipy
import scipy.special


# =========================================================== #

class vortex_filament_model:
      """
      Class vortex filament: Based on Lamb-Oseen vortex model.
      
      Determines the 3 velocity components associated to a classical Lamb-Oseen vortex (see https://www2.ts.ctw.utwente.nl/rob/TS_publications/PDF/R.Steijl.pdf and https://en.wikipedia.org/wiki/Lamb%E2%80%93Oseen_vortex)
      
      Attributes:
      -----------
            center: (3,) numpy array
                  The coordinates of the starting of the center of the vortex filament, m
            strength: float
                  circulation of the vortex filamente, m2/s
            nu: float
                 kinematic viscosity, m/s2
            time: float
                  flow time, s
      
      """
      
      center: np.ndarray
      strength: float
      nu: float
      time: float
      
      
      def __init__(self, center: np.ndarray, strength: float, nu: float, time: float):
            
            self.strength = strength
            self.nu = nu
            self.time = time
            self.center = center
            
            
      def update_vortex_time(self, dt: float):
            """ 
            Parameters:
            -----------
                  dt: float
                        time-step since last update, s
            Returns: 
            -----------
                  -----------
            """
            self.time += dt
            
            #print("\n>>>This time: {} s".format(self.time))
            
            
      def update_core_location(self, dt: float, Uinf: np.ndarray):
            """ 
            Update the location, m, of the vortex core using simple Euler scheme
            """
            
            dx = Uinf[0]*dt
            dy = Uinf[1]*dt
            dz = Uinf[2]*dt
            self.center += np.array([dx,dy,dz], dtype=np.float64)
            
            information = "\nVortex core at: ({:.3f}, {:.3f}, {:.3f}) m\n".format(self.center[0], self.center[1], self.center[2])
            
            print(information)
                      
                             
      def get_induced_velocity(self, points: np.ndarray):
            """
            Solve the vortex-induced velocity sing Biot-Savart + Lamb-Oseen vortex model.
            
            Parameters:
            -----------
                  points: (numPoints,3) ndarray numpy array 
                        x, y, z - coordinates where the induced velocity should be calcualted at, m
            
            Returns:
            -----------
                  vx, vy, vz: (numPoints,) ndarray numpy array 
                        Velocity componenets at the points location, m/s
                  p: (numPoints,) ndarray numpy array 
                        Pressure over density, m2/s2
            """
                        
            x0 = points[:,0] - self.center[0]
            y0 = points[:,1] - self.center[1]
                        
            r_norm2 = x0**2 + y0**2
                        
            g = 1 - np.exp( -r_norm2 / (4 * self.nu * self.time) ) 
            K = self.strength * g / (2 * np.pi * r_norm2)
                        
            vx =  K * y0
            vy = -K * x0
            vz =  np.zeros_like(vx)  
            p  = -0.5 * ( vx**2 + vy**2 + vz**2 )
                        
            return vx, vy, vz, p
                  
# =========================================================== #


class vortex_ring_model:
      """
      Class vortex ring: Based on Lamb-Oseen vortex model.
      
      Based on the Vortex Ring model from (Yoon and Heister, 2004)
      Modified with smoothing kernel by Flavio Martins (1-exp)
      
      Attributes:
      -----------
            strength: float
                  circulation of the vortex filamente, m2/s
            nu: float
                 kinematic viscosity, m/s2
            stroke_time: float
                  The characteristic time of formation of the vortex ring, s
            center: (3,) numpy ndarray
                  Location vector of the vortex's core, m
            speed: (3,) numpy ndarray
                  Velocity vector of the vortex's core, m/s
            radius: float
                  The vortex ring radius, m
                  
      Returns:
      ----------
            vortexRing class object with calculated core thickness (the toroidal diameter of the vortex ring, in meters)
      """
      
      
      strength: float
      nu: float
      stroke_time: float
      center: np.ndarray
      speed: np.ndarray
      radius: float
      
      time: float
      thickness: float
      
      
      def __init__(self, strength: float, nu: float, stroke_time: float, center: np.ndarray, speed: np.ndarray, radius: float):
            
            self.strength = strength
            self.nu = nu
            self.center = center
            self.speed = speed
            self.radius = radius
            self.stroke_time = stroke_time
            self.time = 0.0
            self.thickness = np.sqrt(4 * nu * stroke_time) # m


      def update_vortex_ring_state(self, dt: float, Uinf: np.ndarray):
            """ 
            Update the vortex time, s, and the thickness of the vortex ring, m:
            
            Update also the self-induced translational speed of the vortex based on the work of (Yoon and Heister, 2004) 
            
            Parameters:
            -----------
                  dt: float
                        time-step since last update, s
            Returns: 
            -----------
                  -----------
            """
            
            # Update the flow time:
            self.time += dt
            
            # Update the vortex' thickness:
            self.thickness = np.sqrt(4*self.nu*(self.stroke_time + self.time))
            
            Decay = np.log( 8*self.radius / self.thickness) - 1/2
      
            # Update the vortex self-induced velocity:
            self.speed[0] = self.strength * Decay / (4 * np.pi * self.radius)
      
            # Update the vortex core x-location using a simple Euler scheme:
            self.center += np.array([ (self.speed[0] + Uinf[0])*dt, 0, 0], dtype=np.float64)
            
            information = "Vortex ring core at: ({:.3f}, {:.3f}, {:.3f}) m\n".format(self.center[0], self.center[1], self.center[2])
            
            print(information)


      def get_induced_velocity(self, points: np.ndarray):
            """
            Compute the total velocity induced by a vortex ring on a 3-dimendional space defined by points
            
            The vortex ring plan is always parallel to the x-direction.
            
            Based on the Vortex Ring model from Yoon and Heister, 2004
            Modified with smoothing kernel by Flavio Martins (1-exp)
            
            Parameters:
            -----------
                  points: (numPoints,3) ndarray numpy array 
                        x, y, z - coordinates where the induced velocity should be calcualted at, m
            
            Returns:
            -----------
                  vx, vy, vz: (numPoints,) ndarray numpy array 
                        Velocity componenets at the points location, m/s
                  p: (numPoints,) ndarray numpy array 
                        Pressure over density, m2/s2
            """
            
            # Calculate the radius of the point:
            r = np.sqrt(points[:,1]**2 + points[:,2]**2)
            
            small_value = 5e-2
            
            dx = points[:,0]-self.center[0]
            dx = np.where( (dx>=0) & (dx<small_value), small_value, dx ) 
            dx = np.where( (dx<0) & (dx>-small_value), -small_value, dx )
            

            # Calculate common terms for efficiency
            A = dx**2 + r**2 + self.radius**2
            B = -2 * r * self.radius
            
            a = np.sqrt((r+self.radius)**2 + dx**2)
            m = 4 * r * self.radius / a**2
            
            K = scipy.special.ellipk(m)
            E = scipy.special.ellipe(m)
            
            I1 = (4/a) * K
            I2 = (4/a**3) * E / (1-m)

            # x-velocity:
            vx = (self.strength/(4*np.pi)) * self.radius * ((self.radius + r * A/B) * I2 - r*I1/B) 
            
            vr = (self.strength/(4*np.pi)) * self.radius * dx/B * (I1 - A*I2) 

            theta = np.arctan2(points[:,1], points[:,2])

            # y- and z-velocities:
            vy = vr * np.sin(theta) 
            vz = vr * np.cos(theta) 
            
            
            # Pressure over density, m2/s2
            p = -0.5 * (vx**2 + vy**2 + vz**2)
                  
                  
            return vx, vy, vz, p
      
      
      
# =========================================================== #


class inflating_dipole_model:
      """
      This class simulates an inflating dipole flow that exchanges numpy arrays with pHyFlow's Eulerian module.
      This dipole is orientated along the (x,y)-plane
      
      Attributes:
      -----------
            center: (3,1) numpy array
                  The coordinates of the starting of the center of doublet, m
            U: float
                 The characteristic velocity of the dipole, m/s
            L: float
                  The characteristic length-scale of the dipole, m
            t: float
                  The current flow time, s
            nu: float
                  Kinematic viscosity of the fluid, m2/s
      """
      
      center: np.ndarray
      U: float
      L: float
      nu: float
      t: float
      L0: float
      U0: float
      
      def __init__(self, center: np.ndarray, U: float, L: float, nu: float):
            self.center = center
            self.U = U
            self.L = L
            self.nu = nu

            # Initial conditions:
            self.t = 0  # vortex created at t=0
            self.L0 = L # at the time of creation (t=0) L0 = L
            self.U0 = U # at the time of creation (t=0) L0 = L

            information1 = "\nDipole created: \nU:{:.3f} m/s, L:{:.3f} m, nu:{:.3f} m2/s\n".format(self.U, self.L, self.nu)

            information2 = "U0:{:.3f} m/s, L0:{:.3f} m\n".format(self.U0, self.L0)

            print(information1+information2)


      def update_dipole_state(self, dt: float, Uinf: np.ndarray):
            """ 
            Update the location, m, of the vortex core using simple Euler scheme.
            
            Attributes:
            -----------
                  Uinf: (3,1) numpy array
                        Free-stream velocity, m/s
                  dt: float
                        Time-step size, s
            """
            self.center += np.array([Uinf[0]*dt, Uinf[1]*dt, Uinf[2]*dt], dtype=np.float64)
            self.t += dt
            self.L = np.sqrt( self.L0**2 + 4 * self.nu * self.t )
            self.U = self.U0 * (self.L0/self.L)**2

            information1 = "\nDipole center at: ({:.3f}, {:.3f}, {:.3f}) m\n".format(self.center[0], self.center[1], self.center[2])

            information2 = "Dipole properties U:{:.3f} m/s, L:{:.3f} m\n".format(self.U, self.L)

            print(information1+information2)
                      
                             
      def get_induced_velocity(self, points: np.ndarray, Uinf: np.ndarray):
            """
            Solve 3D Dipole equation.
            
            Parameters:
            -----------
                  points: (numPoints,3) ndarray numpy array 
                        x, y, z - coordinates where the induced velocity should be calcualted at, m
                  Uinf: (1,3) ndarray numpy array 
                        x, y, z - velocity components of the background velocity
            
            Returns:
            -----------
                  vx, vy, vz: (numPoints,) ndarray numpy array 
                        Velocity componenets at the points location, m/s
                  p: (numPoints,) ndarray numpy array 
                        Pressure over density, m2/s2
            """
                        
            x0 = points[:,0] - self.center[0]
            y0 = points[:,1] - self.center[1]
            z0 = points[:,2] - self.center[2]

            factor = self.U * np.exp( - (x0**2 + y0**2 + z0**2) / (2*self.L**2) ) / self.L
                        

            vx = - factor * y0 + Uinf[0]
            vy =   factor * x0 + Uinf[1]
            vz =   np.zeros_like(vx, dtype=np.float64) + Uinf[2]
            p  = - 0.5 * ( vx**2 + vy**2 + vz**2 )
                        
            return vx, vy, vz, p
      
      
class doublet_flow_model:
      """
      This class simulates a doublet flow that exchanges numpy arrays with pHyFlow's Eulerian module
      
      Attributes:
      -----------
            center: (3,) numpy array
                  The coordinates of the starting of the center of doublet, m
            dx, dy, dz: float
                 The unitary vectors describing the direction of the doublet flow
            kappa: float
                  represents the strength of the doublet., m2/s
      """
      
      kappa: float
      center: np.ndarray
      dx: float
      dy: float
      dz: float
      
      def __init__(self, center: np.ndarray, direction: np.ndarray, kappa: float):
            self.kappa = kappa
            self.center = center
            self.dx = direction[0] / np.linalg.norm(direction)
            self.dy = direction[1] / np.linalg.norm(direction)
            self.dz = direction[2] / np.linalg.norm(direction)
            
            
      def update_core_location(self, dt: float, Uinf: np.ndarray):
            """ 
            Update the location, m, of the vortex core using simple Euler scheme
            
            Attributes:
            -----------
                  Uinf: (3,1) numpy array
                        Free-stream velocity, m/s
                  dt: float
                        Time-step size, s
            """
            self.center += np.array([Uinf[0]*dt, Uinf[1]*dt, Uinf[2]*dt], dtype=np.float64)
            
            information = "\nDoublet center at: ({:.3f}, {:.3f}, {:.3f}) m\n".format(self.center[0], self.center[1], self.center[2])
            
            print(information)
                      
                             
      def get_induced_velocity(self, points: np.ndarray, Uinf: np.ndarray):
            """
            Solve 3D Doublet equation.
            
            Parameters:
            -----------
                  points: (numPoints,3) ndarray numpy array 
                        x, y, z - coordinates where the induced velocity should be calcualted at, m
                  Uinf: (1,3) ndarray numpy array 
                        x, y, z - velocity components of the background velocity
            
            Returns:
            -----------
                  vx, vy, vz: (numPoints,) ndarray numpy array 
                        Velocity componenets at the points location, m/s
                  p: (numPoints,) ndarray numpy array 
                        Pressure over density, m2/s2
            """
                        
            x0 = points[:,0] - self.center[0]
            y0 = points[:,1] - self.center[1]
            z0 = points[:,2] - self.center[2]
                        
            r_squared = x0**2 + y0**2 + z0**2
            
            safe_division = - np.divide(self.kappa / (4 * np.pi), r_squared**2.5, out=np.zeros_like(r_squared), where=r_squared!=0)   
              
            vx = safe_division * ( r_squared*self.dx - 3*x0*( self.dx*x0 + self.dy*y0 + self.dz*z0 ) ) + Uinf[0]
            vy = safe_division * ( r_squared*self.dy - 3*y0*( self.dx*x0 + self.dy*y0 + self.dz*z0 ) ) + Uinf[1]
            vz = safe_division * ( r_squared*self.dz - 3*z0*( self.dx*x0 + self.dy*y0 + self.dz*z0 ) ) + Uinf[2]
            p  = -0.5 * ( vx**2 + vy**2 + vz**2 )
                        
            return vx, vy, vz, p