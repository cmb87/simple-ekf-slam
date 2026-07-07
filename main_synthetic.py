import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from sim_utils import plot_camera_and_world_views, project_landmarks, plot_ekf_estimates

from gen_ekf.python.symforce.sym.robot_state_update import robot_state_update
from gen_ekf.python.symforce.sym.robot_state_update_jacobian import robot_state_update_jacobian

from gen_ekf.python.symforce.sym.landmark_measurement import landmark_measurement
from gen_ekf.python.symforce.sym.landmark_measurement_jacobian import landmark_measurement_jacobian

from gen_ekf.python.symforce.sym.landmark_initialization import landmark_initialization
from gen_ekf.python.symforce.sym.landmark_initialization_jacobian import landmark_initialization_jacobian

np.random.seed(0)


if __name__ == "__main__":
    # -------------------------------------------
    # Robot    State = [x, y, z, q1, q2, q3, q0, vx, vy, vz, p, q, r] (position, orientation, velocity and rates of the camera) [13]
    # Landmark State = [x, y, z, theta, phi, rho] Landmark state in Anchored  [6]

    # Prr  Covariance matrix of the robot
    # Prl  Covariance matrix of the robot-landmark cross-correlation
    # Pll  Covariance matrix of the landmark

    # P = [Prr, Prl; Prl', Pll] Covariance matrix of the robot and the landmarks


    nsteps = 200
    nLandmarks = 20
    dt = 0.03

    # Measurement noise
    R = np.array([[1.0, 0.0], [0.0, 1.0]]) # Pixel measurement noise covariance (must be positive-definite!)

    # Prior settings
    rho0 = 0.25  # inverse distance prior
    sigma_rho = 0.25 # inverse distance prior uncertainty
    S_prior = sigma_rho**2

    # Camera settings
    f = 500 # focal length in pixels
    u0 = 320 # principal point in pixels
    v0 = 240 # principal point in pixels
    img_width = u0 * 2
    img_height = v0 * 2

    # 3D landmark position in the world frame (ground truth)
    # Put landmarks in front of the camera: X in [-5, 5], Y in [-5, 5], Z in [2, 15]
    xl_true = np.zeros((nLandmarks, 3))
    xl_true[:, 0] = np.random.uniform(-5, 5, nLandmarks)
    xl_true[:, 1] = np.random.uniform(-5, 5, nLandmarks)
    xl_true[:, 2] = np.random.uniform(2, 15, nLandmarks)

    # Motion noise
    an_sigma = 0.1 # Acceleration noise standard deviation
    gn_sigma = 0.1 # Gyroscope noise standard deviation


    # Initialize the robot state and covariance
    # vz = 1.5 m/s (index 9) to simulate forward motion under the model
    s_groundTruth = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0]) # Initial state of the robot

    s = np.zeros((13+nLandmarks*6))
    s[6] = 1.0 # Set the real part of quaternion (q0) to 1.0 (identity rotation)
    s[9] = 1.5 # Set estimated velocity to match ground truth forward motion
    

    registered_landmarks_ids = []
    landmark_states = np.zeros((nLandmarks, 6)) # Empty array to hold landmark states


    # Global covariance (Robot+Landmarks)
    P = np.zeros((13+6*nLandmarks,13+6*nLandmarks))
    P[:13, :13] = np.eye(13) * 1e-4 # Initialize robot covariance


    # Interactive plotting setup
    plt.ion()
    fig_cam = plt.figure(figsize=(8, 6))
    fig_ekf = plt.figure(figsize=(8, 6))

    trajectory_est = []
    trajectory_gt = []

    for n in range(nsteps):

        # ----------------------------------
        # Step simulation (Ground Truth)
        # Propagate ground truth state with its actual motion equations and sinusoidal slalom yaw rate
        s_groundTruth[11] = 0.01 * np.sin(0.15 * n) # Sinusoidal slalom yaw rate (rad/s)
        s_groundTruth = np.array(robot_state_update(s_groundTruth, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt, 1e-6)).flatten()

        # ------------------------------------
        # Generate simulated measurements of the landmarks in the camera frame (must be within the FOV of the camera!)

        visible_landmarks, u_true_plot, v_true_plot, u_meas_plot, v_meas_plot, visible_ids = project_landmarks(
            s_groundTruth, xl_true, f, u0, v0, img_width, img_height, R
        )

        pos_gt = s_groundTruth[0:3]
        quat_gt = s_groundTruth[3:7] # [q1, q2, q3, q0]

        # Record trajectories
        trajectory_est.append(s[0:3].copy())
        trajectory_gt.append(s_groundTruth[0:3].copy())

        # Render visualizations only once every 5 steps to run 5x-10x faster
        if n % 5 == 0 or n == nsteps - 1:
            # plot_camera_and_world_views(
            #     u_meas_plot, v_meas_plot, u_true_plot, v_true_plot, visible_ids,
            #     img_width, img_height,
            #     pos_gt, quat_gt, xl_true, f, u0, v0, n, fig_cam
            # )
            
            plot_ekf_estimates(
                fig_ekf, n, s, s_groundTruth, P, xl_true, registered_landmarks_ids, 
                trajectory_est, trajectory_gt, f, u0, v0
            )

        # ----------------------------------
        # Filter Prediction
        # Propagate the robot state and covariance using the motion model (expected noise is zero!)

        # Feed the noisy gyroscope measurements directly into the estimated state
        gnoise = gn_sigma * np.random.normal(size=3)
        s[10:13] = s_groundTruth[10:13] + gnoise

        # Predict new state with zero expected noise
        s[0:13] = np.array(robot_state_update(s[0:13], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt, 1e-6)).flatten()

        # Evaluate Jacobian at the expected noise (zero)
        F = np.array(robot_state_update_jacobian(s[0:13], 0.0, 0.0, 0.0, dt, 1e-6))

        Prr = P[:13,:13]
        Prm = P[:13,13:]
        
        Q = np.diag([
            1e-4, 1e-4, 1e-4,          # Position noise (x, y, z)
            1e-5, 1e-5, 1e-5, 1e-5,    # Orientation quaternion noise
            1e-3, 1e-3, 1e-3,          # Velocity noise (vx, vy, vz)
            1e-3, 1e-3, 1e-3           # Angular rate noise (omega)
        ])

        P[:13, :13] = F @ Prr @ F.T + Q
        P[13:,:13] = P[13:,:13] @ F.T
        P[:13,13:] = P[13:,:13].T

        
        # ----------------------------------
        # Filter Measurement Update
        
        for i, u_meas, v_meas in visible_landmarks:
            if i in registered_landmarks_ids:
                ### Case 1: Landmark is already in the state vector
                idx = 13 + 6 * i 

                # Extract the 6D state of landmark i
                l_state = s[idx : idx+6]

                # 1. Compute predicted measurement
                h = np.array(landmark_measurement(s[0:13], f, 1.0, 1.0, u0, v0, *l_state)).flatten()

                # 2. Extract Jacobians directly
                Gr, Gl = landmark_measurement_jacobian(s[0:13], f, 1.0, 1.0, u0, v0, *l_state)
                Hr = np.array(Gr).reshape((2, 13))
                Hl = np.array(Gl).reshape((2, 6))

                # 3. Covariance sub-blocks
                Prr = P[:13, :13]       
                Pmr = P[13:, :13]       
                
                Prl = P[:13, idx:idx+6] 
                Plr = P[idx:idx+6, :13] 
                Pll = P[idx:idx+6, idx:idx+6] 

                Pml = P[13:, idx:idx+6] 

                # 4. Innovation covariance (Z)
                temp_r = Hr @ Prr + Hl @ Plr  
                temp_l = Hr @ Prl + Hl @ Pll  
                Z = temp_r @ Hr.T + temp_l @ Hl.T + R 

                # 5. Full-state cross covariance (P * H.T) split into top and bottom
                K_num_top = Prr @ Hr.T + Prl @ Hl.T   
                K_num_bottom = Pmr @ Hr.T + Pml @ Hl.T 

                # 6. Kalman Gain
                Z_inv = np.linalg.inv(Z)
                K_top = K_num_top @ Z_inv       
                K_bottom = K_num_bottom @ Z_inv 
                K = np.vstack([K_top, K_bottom]) 
                
                # 7. Update state and covariance
                y_res = np.array([u_meas, v_meas]) - h
                s = (s + K @ y_res).flatten()
                
                # Create full H for covariance update
                H_full = np.zeros((2, s.size))
                H_full[:, 0:13] = Hr
                H_full[:, idx:idx+6] = Hl
                
                P = (np.eye(s.size) - K @ H_full) @ P

            else:
                ### Case 2: Landmark is not in the state vector, initialize it
                idx = 13 + 6 * i 
                
                # 1. Initialize the new landmark state
                Y_new = np.array(landmark_initialization(s[0:13], u_meas, v_meas, rho0, u0, v0, f, 1e-9)).flatten()
                
                # 2. Compute the initialization Jacobians (J_r: 6x13, J_z: 6x2, J_s: 6x1)
                Gr, Gy, Gs = landmark_initialization_jacobian(s[0:13], u_meas, v_meas, rho0, u0, v0, f, 1e-9)
                Gs = Gs.reshape((6, 1))

                S = np.asarray([[sigma_rho**2]])

                # 4. Compute new covariance blocks
                P_ll = Gr @ P[0:13, 0:13] @ Gr.T + Gy @ R @ Gy.T + Gs @ S @ Gs.T # New landmark self-covariance (6x6)
                P_rl = P[:, 0:13] @ Gr.T                                # New cross-correlation blocks (size of P x 6)
                
                # 5. Place in the pre-allocated state vector
                s[idx:idx+6] = Y_new
                
                # 6. Place in the pre-allocated covariance matrix (ORDER MATTERS!)
                # Assign cross-covariance blocks first, then place self-covariance P_ll last
                # to prevent P_ll from being silently overwritten with zeros.
                P[:, idx:idx+6] = P_rl
                P[idx:idx+6, :] = P_rl.T
                P[idx:idx+6, idx:idx+6] = P_ll
                
                registered_landmarks_ids.append(i)

    # Keep window open at the end
    plt.ioff()
    plt.show()
