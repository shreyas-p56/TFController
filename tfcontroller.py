import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class ptg:
    #periodic trajectory generation

    def __init__(self, A, f, delta, d):
        self.A = A
        self.f = f
        self.delta = delta
        self.d = d
        self.trajectory()

    def pos_c(self,t):
        A = self.A
        f = self.f
        delta = self.delta
        d = self.d
        
        s_c = d + np.array([A[0]*np.cos(f[0]*t + delta[0]), A[1]*np.cos(f[1]*t + delta[1]), A[2]*np.cos(f[2]*t + delta[2])]).transpose()
        return s_c
    
    def vel_c(self,t):
        A = self.A
        f = self.f
        delta = self.delta
        
        s_dot_c = [-f[0]*A[0]*np.sin(f[0]*t + delta[0]), -f[1]*A[1]*np.sin(f[1]*t + delta[1]), -f[2]*A[2]*np.sin(f[2]*t + delta[2])]
        return s_dot_c

    def acc_c(self,t):
        A = self.A
        f = self.f
        delta = self.delta
        
        s_ddot_c = [-f[0]*f[0]*A[0]*np.cos(f[0]*t + delta[0]), -f[1]*f[1]*A[1]*np.cos(f[1]*t + delta[1]), -f[2]*f[2]*A[2]*np.cos(f[2]*t + delta[2])]
        return s_ddot_c

    def psi_c(self,t):
        #vel_v = self.vel_c(t)
        #return np.arctan2(vel_v[1],vel_v[0])
        return 0

    def trajectory(self):
        total_time = 20.0
        dt = 0.01
        self.dt = dt
        iter = int(total_time/dt)
        t_arr = np.linspace(0.0, total_time, iter)
        
        x_path = []
        y_path = []
        z_path = []
        psi_path = []

        x_dot_path = []
        y_dot_path = []
        z_dot_path = []

        x_ddot_path = []
        y_ddot_path = []
        z_ddot_path = []

        for i in range(iter):
            s = self.pos_c(t_arr[i])
            psi = self.psi_c(t_arr[i])
            s_dot = self.vel_c(t_arr[i])
            s_ddot = self.acc_c(t_arr[i])
            
            x_path.append(s[0])
            y_path.append(s[1])
            z_path.append(s[2])
            psi_path.append(psi)

            x_dot_path.append(s_dot[0])
            y_dot_path.append(s_dot[1])
            z_dot_path.append(s_dot[2])

            x_ddot_path.append(s_ddot[0])
            y_ddot_path.append(s_ddot[1])
            z_ddot_path.append(s_ddot[2])
            
        
        self.x_path = x_path
        self.y_path = y_path
        self.z_path = z_path
        self.psi_path = psi_path

        self.x_dot_path = x_dot_path
        self.y_dot_path = y_dot_path
        self.z_dot_path = z_dot_path

        self.x_ddot_path = x_ddot_path
        self.y_ddot_path = y_ddot_path
        self.z_ddot_path = z_ddot_path


class tfc:
    #trajectory-following controller

    def __init__(self, kp_pos, kd_pos, t_rpx, t_rpy, kp_yaw, kp_p, kp_q, kp_r, dt):
        
        self.g = np.array([0, 0, -9.8]).transpose()
        
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.t_rpx = t_rpx
        self.t_rpy = t_rpy
        self.kp_yaw = kp_yaw
        self.kp_p = kp_p
        self.kp_q = kp_q
        self.kp_r = kp_r
        self.dt = dt
        
    def controller(self, s, s_dot, s_c, s_dot_c, s_ddot_c, psi_c, omega, rot_mat):

        self.s = np.copy(s)
        self.s_dot = np.copy(s_dot)
        self.rot_mat = np.copy(rot_mat)
        self.omega = np.copy(omega)

        self.theta = np.arcsin(-self.rot_mat[2,0])
        self.phi = np.arcsin(self.rot_mat[2,1]/np.cos(self.theta))
        self.psi = np.arctan2(self.rot_mat[1,0]/np.cos(self.theta), self.rot_mat[0,0]/np.cos(self.theta))
        
        bx = self.rot_mat[0,2]
        by = self.rot_mat[1,2]
        bz = self.rot_mat[2,2]

        #'_c' indicates 'commanded', we get it from the trajectory generator
        self.s_c = np.copy(s_c)
        self.s_dot_c = np.copy(s_dot_c)
        self.s_ddot_c = np.copy(s_ddot_c)
        self.psi_c = psi_c
        
        #position controller
        
        self.acc = np.dot(self.kp_pos, self.s_c - self.s) + np.dot(self.kd_pos, self.s_dot_c - self.s_dot) + self.s_ddot_c

        #TFC output command c
        c = (self.acc[2]-self.g[2])/bz
        self.c = np.array([0, 0, c]).transpose()
        
        self.bx_c = self.acc[0]/c
        self.by_c = self.acc[1]/c
    
        #attitude controller
        
        bx_dot_c = -(bx - self.bx_c)/self.t_rpx
        by_dot_c = -(by - self.by_c)/self.t_rpy
        #print("For x:", bx_dot_c, s_dot_c[0], self.theta, s_dot[0])
        #print("For y:", by_dot_c, s_dot_c[1], self.phi, s_dot[1])
        b_dot = np.array([bx_dot_c, by_dot_c]).transpose()
        rot_pq = np.array([[self.rot_mat[1,0], -self.rot_mat[0,0]], [self.rot_mat[1,1], -self.rot_mat[0,1]]])
        
        if (self.psi_c - self.psi > np.pi):
            r_world = self.kp_yaw*(self.psi_c - self.psi -2*np.pi)
        elif (self.psi_c - self.psi < -np.pi):
            r_world = self.kp_yaw*(self.psi_c - self.psi + 2*np.pi)
        else:
            r_world = self.kp_yaw*(self.psi_c - self.psi)

        pq_c = (np.dot(rot_pq, b_dot))/bz
        r_c = bz*r_world
        #print(bz)
        #TFC output commands p_c, q_c, r_c
        self.omega_c = np.array([pq_c[0], pq_c[1], r_c]).transpose()

        self.update()

    def update(self):
        #updates the state of the drone
        '''
        self.s_dot_u = self.s_dot + self.acc*self.dt
        self.s_u = self.s + self.s_dot_u*self.dt
        '''

        psi = self.psi
        rot_mat = np.copy(self.rot_mat)
        omega = np.copy(self.omega)

        z_dot = self.s_dot[2] + self.acc[2]*self.dt

        for i in range(5):

            omega_dot = np.array([self.kp_p*(self.omega_c[0]-omega[0]), self.kp_q*(self.omega_c[1]-omega[1]), self.kp_r*(self.omega_c[2]-omega[2])]).transpose()
            omega = omega + omega_dot*self.dt/5
            #print(self.omega_c)
        
            theta = np.arcsin(-rot_mat[2,0])
            phi = np.arcsin(rot_mat[2,1]/np.cos(theta))
            psi = np.arctan2(rot_mat[1,0]/np.cos(theta), rot_mat[0,0]/np.cos(theta))

            mat_t = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
            omega_w = np.dot(mat_t,omega)
            #print(omega_w)
            #print(mat_t)

            
            if (phi + omega_w[0]*self.dt/5 < 0):
               phi = max(-1.55, phi + omega_w[0]*self.dt/5)
            else:
               phi = min(1.55, phi + omega_w[0]*self.dt/5)
        
            if (theta + omega_w[1]*self.dt/5 < 0):
                theta = max(-1.55, theta + omega_w[1]*self.dt/5)
            else:
                theta = min(1.55, theta + omega_w[1]*self.dt/5)
            

            #phi += omega_w[0]*self.dt/5
            #theta += omega_w[1]*self.dt/5
            psi += omega_w[2]*self.dt/5
        
            while (psi < -np.pi):
                psi += 2*np.pi
        
            while (psi >= np.pi):
                psi -= 2*np.pi

            #print("\t",[phi, theta, psi])

            rot_psi = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
            rot_theta = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            rot_phi = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])

            rot_theta_phi = np.dot(rot_theta, rot_phi)
            rot_mat = np.dot(rot_psi, rot_theta_phi) 
        
            '''
            alpha = np.array([[0, -omega_dot[2], omega_dot[1]], [omega_dot[2], 0, -omega_dot[0]], [-omega_dot[1], omega_dot[0], 0]])
            omega_dt = self.omega + 0.5*omega_dot*self.dt + np.dot(alpha,self.omega)*(self.dt**2)/12
            omega_dt_hat = np.array([[0, -omega_dt[2], omega_dt[1]], [omega_dt[2], 0, -omega_dt[0]], [-omega_dt[1], omega_dt[0], 0]])

            self.rot_mat_u = expm(omega_dt_hat*self.dt)

            theta_u = np.arcsin(-self.rot_mat_u[2,0])
            self.psi_u = np.arctan2(self.rot_mat_u[1,0]/np.cos(theta_u), self.rot_mat_u[0,0]/np.cos(theta_u))
            '''

        #print("\n")

        self.psi_u = psi
        self.omega_u = np.copy(omega)
        self.rot_mat_u = np.copy(rot_mat)

        self.bx_err = self.bx_c - self.rot_mat_u[0,2]
        self.by_err = self.by_c - self.rot_mat_u[1,2]

        acc = self.g + np.dot(self.rot_mat_u, self.c)
        self.s_dot_u = np.array([self.s_dot[0] + acc[0]*self.dt, self.s_dot[1] + acc[1]*self.dt, z_dot]).transpose()
        self.s_u = self.s + 0.5*(self.s_dot + self.s_dot_u)*self.dt


if __name__ == "__main__":
    
    #periodic-trajectory generation parameters
    A = [0, 1, 1]
    f = [0, np.pi/2, np.pi]
    delta = [0, 0, np.pi/2]
    d = np.array([0, 0, 0]).transpose()

    Ptg = ptg(A, f, delta, d)
    
    x_des = Ptg.x_path
    y_des = Ptg.y_path
    z_des = Ptg.z_path
    psi_des = Ptg.psi_path

    x_dot_des = Ptg.x_dot_path
    y_dot_des = Ptg.y_dot_path
    z_dot_des = Ptg.z_dot_path

    x_ddot_des = Ptg.x_ddot_path
    y_ddot_des = Ptg.y_ddot_path
    z_ddot_des = Ptg.z_ddot_path

    dt = Ptg.dt
    iter = len(x_des)
    
    #trajectory-following controller parameters
    kp_pos = np.diag(np.array([144,144,169]))
    kd_pos = np.diag(np.array([24,24,26]))
    t_rpx = 0.1
    t_rpy = 0.1
    kp_yaw = 100
    kp_p = 80
    kp_q = 80
    kp_r = 103

    Tfc = tfc(kp_pos, kd_pos, t_rpx, t_rpy, kp_yaw, kp_p, kp_q, kp_r, dt)

    x_path = [0]
    y_path = [0]
    z_path = [0]
    psi_path = [0]

    #r_err = [((x_des[0]-x_path[0])**2 + (y_des[0]-y_path[0])**2 + (z_des[0]-z_path[0])**2)**0.5]
    bx_err = [0]
    by_err = [0]
    psi_err = [0]

    #x_err = [0]
    #y_err = [0]
    #z_err = [0]

    time = [0]

    s_dot = np.array([0, 0, 0]).transpose()
    omega = np.array([0, 0, 0]).transpose()
    rot_mat = np.identity(3)

    for i in range(iter):
        s = np.array([x_path[-1], y_path[-1], z_path[-1]]).transpose()
        
        psi_c = psi_des[i]
        s_c = np.array([x_des[i], y_des[i], z_des[i]]).transpose()
        s_dot_c = np.array([x_dot_des[i], y_dot_des[i], z_dot_des[i]]).transpose()
        s_ddot_c = np.array([x_ddot_des[i], y_ddot_des[i], z_ddot_des[i]]).transpose()

        Tfc.controller(s, s_dot, s_c, s_dot_c, s_ddot_c, psi_c, omega, rot_mat)
        
        s = np.copy(Tfc.s_u)
        x_path.append(s[0])
        y_path.append(s[1])
        z_path.append(s[2])
        psi_path.append(Tfc.psi_u)

        #r_err.append(((x_des[i]-s[0])**2 + (y_des[i]-s[1])**2 + (z_des[i]-s[2])**2)**0.5)
        bx_err.append(Tfc.bx_err)
        by_err.append(Tfc.by_err)
        psi_err.append(psi_c-psi_path[-1])
        
        #x_err.append(x_des[i]-x_path[-1])
        #y_err.append(y_des[i]-y_path[-1])
        #z_err.append(z_des[i]-z_path[-1])
        
        time.append(time[-1]+dt)

        s_dot = np.copy(Tfc.s_dot_u)
        omega = np.copy(Tfc.omega_u)
        rot_mat = np.copy(Tfc.rot_mat_u)
    
    #trajectory plotting
    
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot3D(x_des, y_des, z_des, linestyle = '-', marker = '.', color = 'red')
    ax.plot3D(x_path, y_path, z_path, linestyle = '-', color = 'blue')

    ax.set_title('Flight path').set_fontsize(20)
    ax.set_xlabel('$x$').set_fontsize(20)
    ax.set_ylabel('$y$').set_fontsize(20)
    ax.set_zlabel('$z$').set_fontsize(20)
    plt.legend(['Planned path','Executed path'], fontsize = 14)

    plt.show()

    plt.plot(time, bx_err, c='r')
    plt.plot(time, by_err, '--b')
    plt.legend(["bx_err","by_err"])
    plt.show()

    plt.plot(time[1:], x_des, c='r', linewidth=3.0)
    plt.plot(time, x_path, '--b')
    plt.legend(["x_des","x_path"])
    plt.show()
    #plt.plot(time[1:], y_des)

    plt.plot(time[1:], y_des, c='r', linewidth=3.0)
    plt.plot(time, y_path, '--b')
    plt.legend(["y_des","y_path"])
    plt.show()

    plt.plot(time[1:], z_des, c='r', linewidth=3.0)
    plt.plot(time, z_path, '--b')
    plt.legend(["z_des","z_path"])
    plt.show()

    #plt.plot(time, psi_err)
    #plt.show()
