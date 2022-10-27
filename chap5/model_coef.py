import numpy as np
x_trim = np.array([[0.000000, 0.000000, 0.000000, 25.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]]).T
u_trim = np.array([[0.000000, 0.000000, 0.000000, 0.500000]]).T
Va_trim = 25.000000
alpha_trim = 0.000000
theta_trim = 0.000000
a_phi1 = 22.628851
a_phi2 = 130.883678
a_theta1 = 5.294738
a_theta2 = 99.947422
a_theta3 = -36.112390
a_V1 = 0.281683
a_V2 = 8.144472
a_V3 = 9.810000
A_lon = np.array([
    [-0.206724, 0.499725, -1.222177, -9.797440, -0.000000],
    [-0.560950, -4.463854, 24.370933, -0.496241, -0.000000],
    [0.200247, -3.992887, -5.294738, 0.000000, -0.000000],
    [0.000000, 0.000000, 0.999974, 0.000000, -0.000000],
    [0.050086, -0.998745, -0.000000, 24.999996, 0.000000]])
B_lon = np.array([
    [-0.138152, 8.144472],
    [-2.586197, 0.000000],
    [-36.112390, 0.000000],
    [0.000000, 0.000000],
    [-0.000000, -0.000000]])
A_lat = np.array([
    [-0.776772, 1.252151, -24.968623, 9.797686, 0.000000],
    [-3.866744, -22.628851, 10.905041, 0.000000, 0.000000],
    [0.783075, -0.115092, -1.227655, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.050149, 0.000000, 0.000000],
    [0.000000, -0.000000, 1.001256, 0.000000, 0.000000]])
B_lat = np.array([
    [1.486172, 3.764969],
    [130.883678, -1.796374],
    [5.011735, -24.881341],
    [0.000000, 0.000000],
    [0.000000, 0.000000]])
Ts = 0.020000
