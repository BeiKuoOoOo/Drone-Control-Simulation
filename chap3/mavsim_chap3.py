"""
mavsimPy
    - Chapter 3 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/18/2018 - RWB
        1/14/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import parameters.simulation_parameters as SIM


from chap2.mav_viewer import MavViewer
from chap3.data_viewer import DataViewer
from chap3.mav_dynamics import MavDynamics
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Rotation

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = MavViewer()  # initialize the mav viewer
data_view = DataViewer()  # initialize view of data plots
if VIDEO is True:
    from chap2.video_writer import VideoWriter
    video = VideoWriter(video_name="chap3_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

# initialize elements of the architecture
mav = MavDynamics(SIM.ts_simulation)
delta = MsgDelta()

# initialize the simulation time
sim_time = SIM.start_time
tArr = []
uArr = []
vArr = []
wArr = []
northArr = []
eastArr = []
downArr = []
e0Arr = []
e1Arr = []
e2Arr = []
e3Arr = []
pArr = []
qArr = []
rArr = []
# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:
    # -------vary forces and moments to check dynamics-------------
    f = Euler2Rotation(mav.true_state.theta, mav.true_state.phi, mav.true_state.psi).T @ np.array([[0], [0], [MAV.mass*MAV.gravity]])
    fx = f.item(0)
    fy = f.item(1)  # 10
    fz = f.item(2)  # 10
    Mx = 0.1  # 0.1
    My = 0.2  # 0.1
    Mz = 0.1  # 0.1
    forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T

    # -------physical system-------------
    mav.update(forces_moments)  # propagate the MAV dynamics
  
    # -------update viewer-------------
    mav_view.update(mav.true_state)  # plot body of MAV
    data_view.update(mav.true_state,  # true states
                     mav.true_state,  # estimated states
                     mav.true_state,  # commanded states
                     delta,  # inputs to the aircraft
                     SIM.ts_simulation)
    
    tArr.append(sim_time)
    northArr.append(mav._state.item(0))
    eastArr.append(mav._state.item(1))
    downArr.append(mav._state.item(2))
    uArr.append(mav._state.item(3))
    vArr.append(mav._state.item(4))
    wArr.append(mav._state.item(5))
    e0Arr.append(mav._state.item(6))
    e1Arr.append(mav._state.item(7))
    e2Arr.append(mav._state.item(8))
    e3Arr.append(mav._state.item(9))
    pArr.append(mav._state.item(10))
    qArr.append(mav._state.item(11))
    rArr.append(mav._state.item(12))


    if VIDEO is True:
        video.update(sim_time)

    # -------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO is True:
    video.close()

plt.figure(0)
plt.plot(tArr, northArr, 'r')
plt.plot(tArr, eastArr, 'b')
plt.plot(tArr, downArr, 'g')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend(['North', 'East', 'Down'])
plt.title('Position vs time')

plt.figure(1)
plt.plot(tArr, uArr, 'r')
plt.plot(tArr, vArr, 'b')
plt.plot(tArr, wArr, 'g')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend(['Along x-axis', 'Along y-axis', 'Along z-axis'])
plt.title('Velocity vs time')


plt.figure(2)
plt.plot(tArr, e0Arr, 'black')
plt.plot(tArr, e1Arr, 'r')
plt.plot(tArr, e2Arr, 'b')
plt.plot(tArr, e3Arr, 'g')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('')
plt.legend(['e0', 'e1', 'e2', 'e3'])


plt.figure(3)
plt.plot(tArr, pArr, 'r')
plt.plot(tArr, qArr, 'b')
plt.plot(tArr, rArr, 'g')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('')
plt.legend(['Roll Rate', 'Pitch Rate', 'Yaw Rate'])
plt.title('Rotation Rate')
plt.show()