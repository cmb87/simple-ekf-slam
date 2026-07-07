import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from matplotlib.patches import Ellipse

def plot_camera_and_world_views(
    u_meas, v_meas, u_true, v_true, visible_ids,
    img_width, img_height,
    pos_gt, quat_gt, xl_true, f, u0, v0, step, fig_cam
):
    """
    Plots a 2x2 grid showing the True Simulation:
      1. Camera Image View (u, v) with ID text labels
      2. Top-down view (Z-X plane) with FOV horizontal cone and landmark IDs
      3. Side view (Z-Y plane) with FOV vertical cone
      4. Front view (X-Y plane) with FOV cross-section at a distance
    """
    fig_cam.canvas.manager.set_window_title(f"True Camera Projection Simulation - Step {step}")
    fig_cam.clf() # Clear the entire figure

    # --- 1. Camera Frame (u, v) ---
    ax1 = fig_cam.add_subplot(2, 2, 1)
    ax1.set_autoscale_on(False)

    # Plot true projections (filter out extreme values)
    if len(u_true) > 0:
        u_true = np.array(u_true)
        v_true = np.array(v_true)
        mask = (u_true >= -100) & (u_true <= img_width + 100) & (v_true >= -100) & (v_true <= img_height + 100)
        if np.any(mask):
            ax1.scatter(u_true[mask], v_true[mask], c='g', marker='o', s=50, label='True Projection', alpha=0.5)
            for id_, u, v in zip(np.array(visible_ids)[mask], u_true[mask], v_true[mask]):
                ax1.text(u + 5, v - 5, f"L{id_}", color='green', fontsize=8, weight='bold')
            
    if len(u_meas) > 0:
        u_meas = np.array(u_meas)
        v_meas = np.array(v_meas)
        mask_meas = (u_meas >= -100) & (u_meas <= img_width + 100) & (v_meas >= -100) & (v_meas <= img_height + 100)
        if np.any(mask_meas):
            ax1.scatter(u_meas[mask_meas], v_meas[mask_meas], c='r', marker='x', s=50, label='Noisy Measurement')

    ax1.scatter(u0, v0, c='blue', marker='+', s=100, linewidths=2, label='Optical Center')
    ax1.axhline(v0, color='blue', linestyle=':', alpha=0.3)
    ax1.axvline(u0, color='blue', linestyle=':', alpha=0.3)

    ax1.set_xlim(0, img_width)
    ax1.set_ylim(img_height, 0)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title("True Camera Image View (pixels)")
    ax1.set_xlabel("u")
    ax1.set_ylabel("v")
    ax1.legend(loc='upper right', fontsize=8)

    # Compute FOV cone properties
    R_wc = R_scipy.from_quat(quat_gt) # Camera to World rotation
    d = 15.0 # Depth of the FOV cone to draw

    v_L = np.array([-u0 / f, 0.0, 1.0])
    v_R = np.array([u0 / f, 0.0, 1.0])
    v_T = np.array([0.0, -v0 / f, 1.0])
    v_B = np.array([0.0, v0 / f, 1.0])

    v_L_w = R_wc.apply(v_L)
    v_R_w = R_wc.apply(v_R)
    v_T_w = R_wc.apply(v_T)
    v_B_w = R_wc.apply(v_B)

    P_L = pos_gt + d * v_L_w
    P_R = pos_gt + d * v_R_w
    P_T = pos_gt + d * v_T_w
    P_B = pos_gt + d * v_B_w

    v_TL = np.array([-u0 / f, -v0 / f, 1.0])
    v_TR = np.array([u0 / f, -v0 / f, 1.0])
    v_BL = np.array([-u0 / f, v0 / f, 1.0])
    v_BR = np.array([u0 / f, v0 / f, 1.0])

    P_TL = pos_gt + d * R_wc.apply(v_TL)
    P_TR = pos_gt + d * R_wc.apply(v_TR)
    P_BL = pos_gt + d * R_wc.apply(v_BL)
    P_BR = pos_gt + d * R_wc.apply(v_BR)

    # --- 2. Top-down View (Z-X Plane) ---
    ax2 = fig_cam.add_subplot(2, 2, 2)
    ax2.set_autoscale_on(False)
    
    ax2.scatter(xl_true[:, 2], xl_true[:, 0], c='lightblue', marker='o', alpha=0.3, label='Inactive Landmarks')
    for i in range(len(xl_true)):
        if i in visible_ids:
            ax2.scatter(xl_true[i, 2], xl_true[i, 0], c='green', marker='o', s=40)
            ax2.text(xl_true[i, 2] + 0.2, xl_true[i, 0] + 0.2, f"L{i}", color='green', fontsize=8, weight='bold')

    ax2.scatter(pos_gt[2], pos_gt[0], c='red', marker='>', s=80, label='Camera (Z-X)')
    ax2.plot([pos_gt[2], P_L[2]], [pos_gt[0], P_L[0]], 'r--', alpha=0.7, label='FOV boundaries')
    ax2.plot([pos_gt[2], P_R[2]], [pos_gt[0], P_R[0]], 'r--', alpha=0.7)
    
    ax2.set_xlim(-1, 20)
    ax2.set_ylim(-6, 6)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title("True Top-down View (Z-X) [meters]")
    ax2.set_xlabel("Z (Depth)")
    ax2.set_ylabel("X (Horizontal)")
    ax2.legend(loc='upper right', fontsize=8)

    # --- 3. Side View (Z-Y Plane) ---
    ax3 = fig_cam.add_subplot(2, 2, 3)
    ax3.set_autoscale_on(False)
    ax3.scatter(xl_true[:, 2], xl_true[:, 1], c='lightblue', marker='o', alpha=0.3)
    
    for i in range(len(xl_true)):
        if i in visible_ids:
            ax3.scatter(xl_true[i, 2], xl_true[i, 1], c='green', marker='o', s=40)
            ax3.text(xl_true[i, 2] + 0.2, xl_true[i, 1] + 0.2, f"L{i}", color='green', fontsize=8, weight='bold')

    ax3.scatter(pos_gt[2], pos_gt[1], c='red', marker='>', s=80)
    ax3.plot([pos_gt[2], P_T[2]], [pos_gt[1], P_T[1]], 'r--', alpha=0.7)
    ax3.plot([pos_gt[2], P_B[2]], [pos_gt[1], P_B[1]], 'r--', alpha=0.7)
    
    ax3.set_xlim(-1, 20)
    ax3.set_ylim(-6, 6)
    ax3.invert_yaxis()
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_title("True Side View (Z-Y) [meters]")
    ax3.set_xlabel("Z (Depth)")
    ax3.set_ylabel("Y (Vertical)")

    # --- 4. Front View (X-Y Plane) ---
    ax4 = fig_cam.add_subplot(2, 2, 4)
    ax4.set_autoscale_on(False)
    ax4.scatter(xl_true[:, 0], xl_true[:, 1], c='lightblue', marker='o', alpha=0.3)
    
    for i in range(len(xl_true)):
        if i in visible_ids:
            ax4.scatter(xl_true[i, 0], xl_true[i, 1], c='green', marker='o', s=40)
            ax4.text(xl_true[i, 0] + 0.2, xl_true[i, 1] + 0.2, f"L{i}", color='green', fontsize=8, weight='bold')

    ax4.scatter(pos_gt[0], pos_gt[1], c='red', marker='o', s=80, label='Camera Pos')
    
    rect_x = [P_TL[0], P_TR[0], P_BR[0], P_BL[0], P_TL[0]]
    rect_y = [P_TL[1], P_TR[1], P_BR[1], P_BL[1], P_TL[1]]
    ax4.plot(rect_x, rect_y, 'r--', alpha=0.7, label=f"FOV Cross-section (d={d}m)")
    
    ax4.set_xlim(-8, 8)
    ax4.set_ylim(-8, 8)
    ax4.invert_yaxis()
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.set_title("True Front View (X-Y) [meters]")
    ax4.set_xlabel("X (Horizontal)")
    ax4.set_ylabel("Y (Vertical)")
    ax4.legend(loc='upper right', fontsize=8)

    fig_cam.tight_layout()
    plt.pause(0.001)

def project_landmarks(s_groundTruth, xl_true, f, u0, v0, img_width, img_height, R_cov):
    """
    Projects 3D world landmarks onto the 2D camera image plane, checks FOV limits, 
    and simulates noisy measurements.
    """
    pos_gt = s_groundTruth[0:3]
    quat_gt = s_groundTruth[3:7] # [q1, q2, q3, q0]

    # Scipy uses [x,y,z,w] which matches [q1,q2,q3,q0]
    rot_w2c = R_scipy.from_quat(quat_gt).inv()

    visible_landmarks = []
    u_meas_plot, v_meas_plot = [], []
    u_true_plot, v_true_plot = [], []

    for i, xl in enumerate(xl_true):
        p_cam = rot_w2c.apply(xl - pos_gt)

        # Check if landmark is in front of the camera
        if p_cam[2] > 0.1:
            u_proj = (p_cam[0] / p_cam[2]) * f + u0
            v_proj = (p_cam[1] / p_cam[2]) * f + v0

            # Check FOV limits
            if 0 <= u_proj <= img_width and 0 <= v_proj <= img_height:
                u_noisy = u_proj + np.random.normal(0, np.sqrt(R_cov[0, 0]))
                v_noisy = v_proj + np.random.normal(0, np.sqrt(R_cov[1, 1]))

                visible_landmarks.append((i, u_noisy, v_noisy))
                u_true_plot.append(u_proj)
                v_true_plot.append(v_proj)
                u_meas_plot.append(u_noisy)
                v_meas_plot.append(v_noisy)

    visible_ids = [item[0] for item in visible_landmarks]

    return visible_landmarks, u_true_plot, v_true_plot, u_meas_plot, v_meas_plot, visible_ids

def draw_covariance_ellipse(ax, mean, cov, color='orange', alpha=0.3, sigma=3):
    """
    Draws a 2D 3-sigma covariance ellipse on the given Matplotlib axis.
    mean: 2-element array [x, y]
    cov: 2x2 covariance matrix
    """
    # Compute eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues in descending order
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    # Angle of rotation in degrees
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    
    # Width and height of the ellipse (sigma * standard deviations)
    # Ensure eigenvalues are positive to avoid NaNs
    vals = np.clip(vals, 1e-9, None)
    width, height = 2 * sigma * np.sqrt(vals)
    
    # Create Ellipse patch
    ell = Ellipse(
        xy=mean, width=width, height=height, angle=theta,
        edgecolor=color, facecolor='none', linestyle='-', alpha=alpha, linewidth=1.5
    )
    ax.add_patch(ell)

def plot_ekf_estimates(
    fig_ekf, step, s, s_groundTruth, P, xl_true, registered_landmarks_ids, 
    trajectory_est, trajectory_gt, f, u0, v0
):
    """
    Plots the EKF estimates (Robot Trajectory + Landmark Positions) 
    against Ground Truth in a 2x2 grid symmetric to the Simulation views:
      1. EKF Reconstructed Camera Image View (pixels)
      2. EKF estimated Top-down View (Z-X Plane) [meters] + 3-sigma error ellipses
      3. EKF estimated Side View (Z-Y Plane) [meters] + 3-sigma error ellipses
      4. EKF estimated Front View (X-Y Plane) [meters] + 3-sigma error ellipses
    """
    fig_ekf.canvas.manager.set_window_title(f"EKF SLAM Estimates - Step {step}")
    fig_ekf.clf()
    
    img_width = u0 * 2
    img_height = v0 * 2

    # Extract robot positions and orientation
    pos_est = s[0:3]
    pos_gt = s_groundTruth[0:3]
    
    quat_est = s[3:7] # [q1, q2, q3, q0]
    quat_gt = s_groundTruth[3:7] # [q1, q2, q3, q0]
    
    # Normalize EKF estimated quaternion to avoid singularities during plot rotation
    if np.linalg.norm(quat_est) > 1e-3:
        quat_est_norm = quat_est / np.linalg.norm(quat_est)
    else:
        quat_est_norm = np.array([0.0, 0.0, 0.0, 1.0])

    R_wc_est = R_scipy.from_quat(quat_est_norm)
    R_wc_gt = R_scipy.from_quat(quat_gt)

    # Pre-calculate estimated 3D world coordinates and Cartesian 3D covariance blocks
    p_l_world_est = {}
    p_l_cov_est = {}

    for i in registered_landmarks_ids:
        idx = 13 + 6 * i
        l_state = s[idx : idx+6]
        Pll = P[idx:idx+6, idx:idx+6] # Get 6x6 covariance block of the landmark
        
        xi, yi, zi, theta, phi, rho = l_state
        m_vec = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        rho_safe = max(rho, 1e-6)
        p_l_world = np.array([xi, yi, zi]) + (1.0 / rho_safe) * m_vec
        p_l_world_est[i] = p_l_world

        # Linearized Error Propagation (Jacobian conversion) to map 6D inverse depth cov -> 3D Cartesian cov
        JC = np.zeros((3, 6))
        JC[0:3, 0:3] = np.eye(3) # d_cartesian / d_[xi, yi, zi]
        
        # d_cartesian / d_theta
        JC[0, 3] = (1.0 / rho_safe) * np.cos(theta) * np.cos(phi)
        JC[1, 3] = (1.0 / rho_safe) * np.cos(theta) * np.sin(phi)
        JC[2, 3] = (1.0 / rho_safe) * (-np.sin(theta))
        
        # d_cartesian / d_phi
        JC[0, 4] = (1.0 / rho_safe) * (-np.sin(theta) * np.sin(phi))
        JC[1, 4] = (1.0 / rho_safe) * (np.sin(theta) * np.cos(phi))
        JC[2, 4] = 0.0
        
        # d_cartesian / d_rho
        JC[0, 5] = - (1.0 / rho_safe**2) * np.sin(theta) * np.cos(phi)
        JC[1, 5] = - (1.0 / rho_safe**2) * np.sin(theta) * np.sin(phi)
        JC[2, 5] = - (1.0 / rho_safe**2) * np.cos(theta)

        # 3D Cartesian Covariance = JC * Pll * JC^T (size 3x3)
        p_l_cov_est[i] = JC @ Pll @ JC.T

    # Compute FOV cone properties at depth d=15.0m
    d = 15.0
    v_L = np.array([-u0 / f, 0.0, 1.0])
    v_R = np.array([u0 / f, 0.0, 1.0])
    v_T = np.array([0.0, -v0 / f, 1.0])
    v_B = np.array([0.0, v0 / f, 1.0])

    v_TL = np.array([-u0 / f, -v0 / f, 1.0])
    v_TR = np.array([u0 / f, -v0 / f, 1.0])
    v_BL = np.array([-u0 / f, v0 / f, 1.0])
    v_BR = np.array([u0 / f, v0 / f, 1.0])

    # Endpoints/Corners (Ground Truth)
    P_L_gt = pos_gt + d * R_wc_gt.apply(v_L)
    P_R_gt = pos_gt + d * R_wc_gt.apply(v_R)
    P_T_gt = pos_gt + d * R_wc_gt.apply(v_T)
    P_B_gt = pos_gt + d * R_wc_gt.apply(v_B)
    
    P_TL_gt = pos_gt + d * R_wc_gt.apply(v_TL)
    P_TR_gt = pos_gt + d * R_wc_gt.apply(v_TR)
    P_BL_gt = pos_gt + d * R_wc_gt.apply(v_BL)
    P_BR_gt = pos_gt + d * R_wc_gt.apply(v_BR)

    # Endpoints/Corners (EKF Estimated)
    P_L_est = pos_est + d * R_wc_est.apply(v_L)
    P_R_est = pos_est + d * R_wc_est.apply(v_R)
    P_T_est = pos_est + d * R_wc_est.apply(v_T)
    P_B_est = pos_est + d * R_wc_est.apply(v_B)
    
    P_TL_est = pos_est + d * R_wc_est.apply(v_TL)
    P_TR_est = pos_est + d * R_wc_est.apply(v_TR)
    P_BL_est = pos_est + d * R_wc_est.apply(v_BL)
    P_BR_est = pos_est + d * R_wc_est.apply(v_BR)


    # --- 1. Reconstructed Camera View (u, v) ---
    ax1 = fig_ekf.add_subplot(2, 2, 1)
    ax1.set_autoscale_on(False)
    
    ax1.scatter(u0, v0, c='blue', marker='+', s=100, linewidths=2, label='Optical Center')
    ax1.axhline(v0, color='blue', linestyle=':', alpha=0.3)
    ax1.axvline(u0, color='blue', linestyle=':', alpha=0.3)

    if np.linalg.norm(quat_est) > 1e-3:
        R_cw_est = R_wc_est.inv()
        for i, p_l_world in p_l_world_est.items():
            p_cam_est = R_cw_est.apply(p_l_world - pos_est)
            if 0.1 < p_cam_est[2] < 30.0:
                u_est_proj = (p_cam_est[0] / p_cam_est[2]) * f + u0
                v_est_proj = (p_cam_est[1] / p_cam_est[2]) * f + v0
                
                # Rigid filter check to prevent axis stretching
                if -200 <= u_est_proj <= img_width + 200 and -200 <= v_est_proj <= img_height + 200:
                    ax1.scatter(u_est_proj, v_est_proj, c='orange', marker='*', s=80)
                    ax1.text(u_est_proj + 5, v_est_proj - 5, f"L{i}", color='orange', fontsize=8, weight='bold')

    ax1.set_xlim(0, img_width)
    ax1.set_ylim(img_height, 0)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title("Reconstructed Camera View (pixels)")
    ax1.set_xlabel("u")
    ax1.set_ylabel("v")
    ax1.legend(loc='upper right', fontsize=8)


    # --- 2. Top-down View (Z-X Plane) ---
    ax2 = fig_ekf.add_subplot(2, 2, 2)
    ax2.set_autoscale_on(False)
    
    # Plot trajectories
    if len(trajectory_est) > 0:
        traj_est_np = np.array(trajectory_est)
        traj_gt_np = np.array(trajectory_gt)
        ax2.plot(traj_est_np[:, 2], traj_est_np[:, 0], 'b-', linewidth=2, label="EKF Estimated")
        ax2.plot(traj_gt_np[:, 2], traj_gt_np[:, 0], 'k--', linewidth=2, label="Ground Truth")

    ax2.scatter(pos_gt[2], pos_gt[0], c='k', marker='>', s=100)
    ax2.scatter(pos_est[2], pos_est[0], c='b', marker='>', s=100)

    # Cones
    ax2.plot([pos_gt[2], P_L_gt[2]], [pos_gt[0], P_L_gt[0]], 'k:', alpha=0.4, label="True FOV")
    ax2.plot([pos_gt[2], P_R_gt[2]], [pos_gt[0], P_R_gt[0]], 'k:', alpha=0.4)
    ax2.plot([pos_est[2], P_L_est[2]], [pos_est[0], P_L_est[0]], 'b--', alpha=0.5, label="Estimated FOV")
    ax2.plot([pos_est[2], P_R_est[2]], [pos_est[0], P_R_est[0]], 'b--', alpha=0.5)

    # Plot True and Estimated Landmarks + 3-sigma ellipses in Z-X
    ax2.scatter(xl_true[:, 2], xl_true[:, 0], c='lightgray', marker='o', s=50, label="True Landmarks")
    for i, p_l_world in p_l_world_est.items():
        if -10.0 <= p_l_world[2] <= 30.0 and -15.0 <= p_l_world[0] <= 15.0:
            ax2.scatter(p_l_world[2], p_l_world[0], c='orange', marker='*', s=80)
            ax2.text(p_l_world[2] + 0.2, p_l_world[0] + 0.2, f"L{i}", color='orange', fontsize=8, weight='bold')

            # Draw 3-sigma error ellipse: Z is index 2, X is index 0
            cov_3d = p_l_cov_est[i]
            Cov_ZX = np.array([[cov_3d[2, 2], cov_3d[2, 0]], [cov_3d[0, 2], cov_3d[0, 0]]])
            draw_covariance_ellipse(ax2, [p_l_world[2], p_l_world[0]], Cov_ZX, color='orange')

    ax2.set_xlim(-1, 20)
    ax2.set_ylim(-6, 6)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title("Estimated Top-down (Z-X) [meters]")
    ax2.set_xlabel("Z (Depth)")
    ax2.set_ylabel("X (Horizontal)")
    
    handles, labels = ax2.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markersize=10))
    labels.append("Est Landmarks")
    ax2.legend(handles, labels, loc='upper left', fontsize=8)


    # --- 3. Side View (Z-Y Plane) ---
    ax3 = fig_ekf.add_subplot(2, 2, 3)
    ax3.set_autoscale_on(False)
    
    if len(trajectory_est) > 0:
        ax3.plot(traj_est_np[:, 2], traj_est_np[:, 1], 'b-', linewidth=2)
        ax3.plot(traj_gt_np[:, 2], traj_gt_np[:, 1], 'k--', linewidth=2)

    ax3.scatter(pos_gt[2], pos_gt[1], c='k', marker='>', s=100)
    ax3.scatter(pos_est[2], pos_est[1], c='b', marker='>', s=100)

    # Cones
    ax3.plot([pos_gt[2], P_T_gt[2]], [pos_gt[1], P_T_gt[1]], 'k:', alpha=0.4)
    ax3.plot([pos_gt[2], P_B_gt[2]], [pos_gt[1], P_B_gt[1]], 'k:', alpha=0.4)
    ax3.plot([pos_est[2], P_T_est[2]], [pos_est[1], P_T_est[1]], 'b--', alpha=0.5)
    ax3.plot([pos_est[2], P_B_est[2]], [pos_est[1], P_B_est[1]], 'b--', alpha=0.5)

    ax3.scatter(xl_true[:, 2], xl_true[:, 1], c='lightgray', marker='o', s=50)
    for i, p_l_world in p_l_world_est.items():
        if -10.0 <= p_l_world[2] <= 30.0 and -15.0 <= p_l_world[1] <= 15.0:
            ax3.scatter(p_l_world[2], p_l_world[1], c='orange', marker='*', s=80)
            ax3.text(p_l_world[2] + 0.2, p_l_world[1] + 0.2, f"L{i}", color='orange', fontsize=8, weight='bold')

            # Draw 3-sigma error ellipse: Z is index 2, Y is index 1
            cov_3d = p_l_cov_est[i]
            Cov_ZY = np.array([[cov_3d[2, 2], cov_3d[2, 1]], [cov_3d[1, 2], cov_3d[1, 1]]])
            draw_covariance_ellipse(ax3, [p_l_world[2], p_l_world[1]], Cov_ZY, color='orange')

    ax3.set_xlim(-1, 20)
    ax3.set_ylim(-6, 6)
    ax3.invert_yaxis()
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_title("Estimated Side View (Z-Y) [meters]")
    ax3.set_xlabel("Z (Depth)")
    ax3.set_ylabel("Y (Vertical)")


    # --- 4. Front View (X-Y Plane) ---
    ax4 = fig_ekf.add_subplot(2, 2, 4)
    ax4.set_autoscale_on(False)
    
    ax4.scatter(pos_gt[0], pos_gt[1], c='k', marker='o', s=100)
    ax4.scatter(pos_est[0], pos_est[1], c='b', marker='o', s=100)

    # Cross-sections (d=15m)
    rect_x_gt = [P_TL_gt[0], P_TR_gt[0], P_BR_gt[0], P_BL_gt[0], P_TL_gt[0]]
    rect_y_gt = [P_TL_gt[1], P_TR_gt[1], P_BR_gt[1], P_BL_gt[1], P_TL_gt[1]]
    ax4.plot(rect_x_gt, rect_y_gt, 'k:', alpha=0.4)

    rect_x_est = [P_TL_est[0], P_TR_est[0], P_BR_est[0], P_BL_est[0], P_TL_est[0]]
    rect_y_est = [P_TL_est[1], P_TR_est[1], P_BR_est[1], P_BL_est[1], P_TL_est[1]]
    ax4.plot(rect_x_est, rect_y_est, 'b--', alpha=0.5)

    ax4.scatter(xl_true[:, 0], xl_true[:, 1], c='lightgray', marker='o', s=50)
    for i, p_l_world in p_l_world_est.items():
        if -15.0 <= p_l_world[0] <= 15.0 and -15.0 <= p_l_world[1] <= 15.0:
            ax4.scatter(p_l_world[0], p_l_world[1], c='orange', marker='*', s=80)
            ax4.text(p_l_world[0] + 0.2, p_l_world[1] + 0.2, f"L{i}", color='orange', fontsize=8, weight='bold')

            # Draw 3-sigma error ellipse: X is index 0, Y is index 1
            cov_3d = p_l_cov_est[i]
            Cov_XY = np.array([[cov_3d[0, 0], cov_3d[0, 1]], [cov_3d[1, 0], cov_3d[1, 1]]])
            draw_covariance_ellipse(ax4, [p_l_world[0], p_l_world[1]], Cov_XY, color='orange')

    ax4.set_xlim(-8, 8)
    ax4.set_ylim(-8, 8)
    ax4.invert_yaxis()
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_title("Estimated Front View (X-Y) [meters]")
    ax4.set_xlabel("X (Horizontal)")
    ax4.set_ylabel("Y (Vertical)")

    fig_ekf.tight_layout()
    plt.pause(0.001)
