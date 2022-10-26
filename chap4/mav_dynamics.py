"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
part of mavPySim 
    - Beard & McLain, PUP, 2012
    - Update history:  
        12/20/2018 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import math

# load message types
from message_types.msg_state import MsgState

import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler


class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0]])   # (12)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize true_state message
        self.true_state = MsgState()

    ###################################
    # public functions
    def update(self, delta, wind):
        """
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        """
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        # north = state.item(0)
        # east = state.item(1)
        # down = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        pos_dot = Quaternion2Rotation(np.array([e0, e1, e2, e3])) @ np.array([u, v, w])
        north_dot = pos_dot[0]
        east_dot = pos_dot[1]
        down_dot = pos_dot[2]

        # position dynamics
        u_dot = r * v - q * w + fx / MAV.mass
        v_dot = p * w - r * u + fy / MAV.mass
        w_dot = q * u - p * v + fz / MAV.mass

        # rotational kinematics
        # skew = np.array([
        #     [0, -p, -q, -r],
        #     [p, 0, r, -q],
        #     [q, -r, 0, p],
        #     [r, q, -p, 0]])
        # e_dot = 0.5 * skew @ np.array([e0, e1, e2, e3]).T
        # e0_dot = e_dot.item(0)
        # e1_dot = e_dot.item(1)
        # e2_dot = e_dot.item(2)
        # e3_dot = e_dot.item(3)
        e0_dot = 0.5 * (-p*e1 - q*e2 - r*e3)
        e1_dot = 0.5 * ( p*e0 + r*e2 - q*e3)
        e2_dot = 0.5 * ( q*e0 - r*e1 + p*e3)
        e3_dot = 0.5 * ( r*e0 + q*e1 - p*e2)

        # rotatonal dynamics
        p_dot = MAV.gamma1*p*q - MAV.gamma2*q*r + MAV.gamma3*l + MAV.gamma4*n
        q_dot = MAV.gamma5*p*r - MAV.gamma6*(p**2 - r**2) + m / MAV.Jy
        r_dot = MAV.gamma7*p*q - MAV.gamma1*q*r + MAV.gamma4*l + MAV.gamma8*n

        # collect the derivative of the states
        x_dot = np.array([[north_dot, east_dot, down_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]
        # convert wind vector from world to body frame and add gust
        wind_body_frame = Quaternion2Rotation(self._state[6:10]) @ steady_state + gust
        # velocity vector relative to the airmass
        v_air = self._state[3:6] - wind_body_frame
        ur = v_air.item(0)
        vr = v_air.item(1)
        wr = v_air.item(2)
        # compute airspeed
        self._Va = np.linalg.norm(v_air) #np.sqrt(ur**2+vr**2+wr**2)
        # compute angle of attack
        if ur == 0:
            self._alpha = 0
        else:
            self._alpha = np.arctan(wr/ur)
        # compute sideslip angle
        if self._Va == 0:
            self._beta = 0
        else:
            self._beta = np.arcsin(vr/self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)
        # compute gravitaional forces
        fg = Quaternion2Rotation(self._state[6:10]).T @ np.array([[0], [0], [MAV.mass*MAV.gravity]])
        # compute Lift and Drag coefficients
        M = MAV.M
        alpha = self._alpha
        alpha0 = MAV.alpha0
        rho = MAV.rho
        Va = self._Va
        S = MAV.S_wing
        q = self._state.item(11)
        c = MAV.c
        pVa2S_2 = 0.5 * rho * (Va ** 2) * S
        c_2Va = c / (2 * Va)

        e_neg_M = math.exp(-M * (alpha - alpha0))
        e_pos_M = math.exp(M * (alpha + alpha0))

        sigma = (1 + e_neg_M + e_pos_M) / ((1 + e_neg_M) * (1 + e_pos_M))
        CL = (1 - sigma) * (MAV.C_L_0 + MAV.C_L_alpha * alpha) + sigma * (2 * np.sign(alpha) * (np.sin(alpha) ** 2) * np.cos(alpha))
        CD = MAV.C_D_p + ((MAV.C_L_0 + MAV.C_L_alpha * alpha) ** 2) / (np.pi * MAV.e * MAV.AR)

        F_lift = pVa2S_2 * (CL + MAV.C_L_q * c_2Va * q + MAV.C_L_delta_e * delta.elevator)
        F_drag = pVa2S_2 * (CD + MAV.C_D_q * c_2Va * q + MAV.C_D_delta_e * delta.elevator)
        #compute propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._alpha, delta.throttle)

        # compute longitudinal forces in body frame
        q_dynamic = 0.5*MAV.rho*(self._Va**2)*MAV.S_wing
        fx = -F_drag * np.cos(alpha) + F_lift * np.sin(alpha) + fg.item(0)
        fz = -F_drag * np.sin(alpha) - F_lift * np.cos(alpha) + fg.item(2)
        # compute lateral forces in body frame
        b_2Va = MAV.b / (2 * Va)
        beta = self.true_state.beta

        fy = pVa2S_2 * (MAV.C_Y_0 + MAV.C_Y_beta * beta + MAV.C_Y_p * b_2Va * p + MAV.C_Y_r * b_2Va * r + MAV.C_Y_delta_a * delta.aileron + MAV.C_Y_delta_r * delta.rudder) + fg.item(1)
        # compute logitudinal torque in body frame
        My = pVa2S_2 * c * (MAV.C_m_0 + MAV.C_m_alpha * alpha + MAV.C_m_q * c_2Va * q + MAV.C_m_delta_e * delta.elevator)

        # compute lateral torques in body frame
        Mx = pVa2S_2 * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta * beta + MAV.C_ell_p * b_2Va * p + MAV.C_ell_r * b_2Va * r + MAV.C_ell_delta_a * delta.aileron + MAV.C_ell_delta_r * delta.rudder)
        Mz = pVa2S_2 * MAV.b * (MAV.C_n_0 + MAV.C_n_beta * beta + MAV.C_n_p * b_2Va * p + MAV.C_n_r * b_2Va * r + MAV.C_n_delta_a * delta.aileron + MAV.C_n_delta_r * delta.rudder)
        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz
        print(self._Va)
        return np.array([[fx, fy, fz, Mx, My, Mz]]).T


    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller  (See addendum by McLain)
        # map delta_t throttle command(0 to 1) into motor input voltage
        V_in = delta_t * MAV.V_max

        a_1 = MAV.rho * MAV.D_prop ** 5 / ((2.0 * np.pi) ** 2) * MAV.C_Q0
        b_1 = MAV.rho * MAV.D_prop ** 4 / (2.0 * np.pi) * MAV.C_Q1 * self._Va + (MAV.KQ ** 2) / MAV.R_motor
        c_1 = MAV.rho * MAV.D_prop ** 3 * MAV.C_Q2 * self._Va ** 2 - MAV.KQ / MAV.R_motor * V_in + MAV.KQ * MAV.i0

        Omega_p = (-b_1 + np.sqrt(b_1 ** 2 - 4 * a_1 * c_1)) / (2.0 * a_1)

        # nondimensional advanced ratio J for propeller
        J = 2 * np.pi * self._Va / (Omega_p * MAV.D_prop)

        # Aerodynamic Coefficient Approximations
        CT = MAV.C_T2 * J ** 2 + MAV.C_T1 * J + MAV.C_T0
        CQ = MAV.C_Q2 * J ** 2 + MAV.C_Q1 * J + MAV.C_Q0

        # thrust and torque due to propeller
        thrust_prop = MAV.rho * MAV.D_prop ** 4 / (4 * np.pi ** 2) * Omega_p ** 2 * CT
        torque_prop = MAV.rho * MAV.D_prop ** 5 / (4 * np.pi ** 2) * Omega_p ** 2 * CQ
        return thrust_prop, torque_prop

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        pdot = Quaternion2Rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
