import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from matplotlib.patches import Ellipse

from gen_ekf.python.symforce.sym.robot_state_update import robot_state_update
from gen_ekf.python.symforce.sym.robot_state_update_jacobian import robot_state_update_jacobian

from gen_ekf.python.symforce.sym.landmark_measurement import landmark_measurement
from gen_ekf.python.symforce.sym.landmark_measurement_jacobian import landmark_measurement_jacobian

from gen_ekf.python.symforce.sym.landmark_initialization import landmark_initialization
from gen_ekf.python.symforce.sym.landmark_initialization_jacobian import landmark_initialization_jacobian

np.random.seed(0)

# -----------------------------------------------------------------------------
# 1. Computer Vision / Patch Tracking Classes and Utilities

class LandmarkPatch:
    def __init__(self, landmark_id, image_frame, u, v, patch_size=11):
        """
        Stores the reference intensity template of the landmark.
        """
        self.id = landmark_id
        self.miss_count = 0
        self.current_u = u
        self.current_v = v
        self.patch_size = patch_size
        
        half = patch_size // 2
        u_int, v_int = int(round(u)), int(round(v))
        
        # Grab the 11x11 patch
        self.template = image_frame[
            v_int - half : v_int + half + 1,
            u_int - half : u_int + half + 1
        ].copy()


def active_patch_search(current_gray_frame, patch, z_hat, S_cov, sigma_multiplier=3.0):
    """
    Performs Active Feature Tracking inside the EKF-predicted 3-sigma innovation region.
    """
    u_hat, v_hat = z_hat[0], z_hat[1]
    
    search_half_w = int(np.ceil(sigma_multiplier * np.sqrt(S_cov[0, 0])))
    search_half_h = int(np.ceil(sigma_multiplier * np.sqrt(S_cov[1, 1])))
    
    search_half_w = max(15, search_half_w)
    search_half_h = max(15, search_half_h)

    h, w = current_gray_frame.shape
    u_min = max(0, int(round(u_hat)) - search_half_w)
    u_max = min(w, int(round(u_hat)) + search_half_w + 1)
    v_min = max(0, int(round(v_hat)) - search_half_h)
    v_max = min(h, int(round(v_hat)) + search_half_h + 1)
    
    temp_h, temp_w = patch.template.shape
    if (u_max - u_min < temp_w) or (v_max - v_min < temp_h):
        return None

    roi = current_gray_frame[v_min:v_max, u_min:u_max]
    if roi.shape[0] < temp_h or roi.shape[1] < temp_w:
        return None

    res = cv2.matchTemplate(roi, patch.template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    MATCH_THRESHOLD = 0.80 # Slightly lower threshold to be robust to real camera illumination noise
    if max_val >= MATCH_THRESHOLD:
        u_meas = u_min + max_loc[0] + temp_w // 2
        v_meas = v_min + max_loc[1] + temp_h // 2
        
        y_res = np.array([u_meas - u_hat, v_meas - v_hat])
        try:
            S_inv = np.linalg.inv(S_cov)
            mahalanobis_dist2 = y_res.T @ S_inv @ y_res
            if mahalanobis_dist2 < 5.99:
                return np.array([u_meas, v_meas])
        except np.linalg.LinAlgError:
            pass
            
    return None


def initialize_new_features_bucketed(frame_gray, active_patches, target_features=12, bucket_size=80):
    """
    Enforces spatially uniform feature extraction by partitioning the image frame 
    into grids and blocking out buckets around currently tracked active patches.
    """
    h, w = frame_gray.shape
    search_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    for patch in active_patches:
        u, v = int(round(patch.current_u)), int(round(patch.current_v))
        cv2.rectangle(
            search_mask,
            (u - bucket_size//2, v - bucket_size//2),
            (u + bucket_size//2, v + bucket_size//2),
            0, -1
        )
        
    needed = target_features - len(active_patches)
    if needed <= 0:
        return []
        
    new_corners = cv2.goodFeaturesToTrack(
        frame_gray,
        maxCorners=needed,
        qualityLevel=0.01,
        minDistance=bucket_size,
        mask=search_mask
    )
    
    if new_corners is not None:
        return new_corners.reshape(-1, 2)
    return []


def delete_landmark_from_ekf(s, P, landmark_id_to_delete, active_patches):
    """
    Removes a landmark from the state vector s and covariance matrix P.
    """
    row_start = 13 + 6 * landmark_id_to_delete
    row_end = row_start + 6
    
    s_new = np.delete(s, slice(row_start, row_end), axis=0)
    P_temp = np.delete(P, slice(row_start, row_end), axis=0)
    P_new = np.delete(P_temp, slice(row_start, row_end), axis=1)
    
    updated_patches = []
    for patch in active_patches:
        if patch.id == landmark_id_to_delete:
            continue
            
        if patch.id > landmark_id_to_delete:
            patch.id -= 1
            
        updated_patches.append(patch)
        
    return s_new, P_new, updated_patches


# -----------------------------------------------------------------------------
# 2. Visualizers (Live Matplotlib SLAM Map Panel)

def draw_covariance_ellipse(ax, mean, cov, color='orange', alpha=0.3, sigma=3):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    vals = np.clip(vals, 1e-9, None)
    width, height = 2 * sigma * np.sqrt(vals)
    ell = Ellipse(
        xy=mean, width=width, height=height, angle=theta,
        edgecolor=color, facecolor='none', linestyle='-', alpha=alpha, linewidth=1.5
    )
    ax.add_patch(ell)


def plot_webcam_slam_map(fig_map, s, P, active_patches, trajectory_est, f, u0, v0, step):
    """
    Plots the EKF's real-time estimated trajectory and mapped 3D landmarks top-down (Z-X Plane).
    """
    fig_map.canvas.manager.set_window_title(f"Real-Time EKF SLAM Map - Step {step}")
    fig_map.clf()
    ax = fig_map.add_subplot(1, 1, 1)
    ax.set_autoscale_on(False)

    # Plot estimated camera center
    pos_est = s[0:3]
    
    # Plot Trajectory path
    if len(trajectory_est) > 0:
        traj_est_np = np.array(trajectory_est)
        ax.plot(traj_est_np[:, 2], traj_est_np[:, 0], 'b-', linewidth=2, label="Estimated Path")

    ax.scatter(pos_est[2], pos_est[0], c='b', marker='>', s=120, label="Estimated Camera")

    # Compute FOV cone properties (X vs Z) at depth d=5.0m
    d = 5.0
    v_L = np.array([-u0 / f, 0.0, 1.0])
    v_R = np.array([u0 / f, 0.0, 1.0])

    quat_est = s[3:7] # [q1, q2, q3, q0]
    if np.linalg.norm(quat_est) > 1e-3:
        R_wc_est = R_scipy.from_quat(quat_est / np.linalg.norm(quat_est))
        P_L_est = pos_est + d * R_wc_est.apply(v_L)
        P_R_est = pos_est + d * R_wc_est.apply(v_R)
        ax.plot([pos_est[2], P_L_est[2]], [pos_est[0], P_L_est[0]], 'b--', alpha=0.5, label="FOV Cone")
        ax.plot([pos_est[2], P_R_est[2]], [pos_est[0], P_R_est[0]], 'b--', alpha=0.5)

    # Plot estimated 3D landmarks + 3-sigma ellipses
    for patch in active_patches:
        idx = 13 + 6 * patch.id
        l_state = s[idx : idx+6]
        Pll = P[idx:idx+6, idx:idx+6]
        
        xi, yi, zi, theta, phi, rho = l_state
        m_vec = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        rho_safe = max(rho, 1e-6)
        p_l_world = np.array([xi, yi, zi]) + (1.0 / rho_safe) * m_vec

        # Jacobian error propagation to map 6D inverse depth cov -> 3D Cartesian cov
        JC = np.zeros((3, 6))
        JC[0:3, 0:3] = np.eye(3)
        JC[0, 3] = (1.0 / rho_safe) * np.cos(theta) * np.cos(phi)
        JC[1, 3] = (1.0 / rho_safe) * np.cos(theta) * np.sin(phi)
        JC[2, 3] = (1.0 / rho_safe) * (-np.sin(theta))
        JC[0, 4] = (1.0 / rho_safe) * (-np.sin(theta) * np.sin(phi))
        JC[1, 4] = (1.0 / rho_safe) * (np.sin(theta) * np.cos(phi))
        JC[2, 4] = 0.0
        JC[0, 5] = - (1.0 / rho_safe**2) * np.sin(theta) * np.cos(phi)
        JC[1, 5] = - (1.0 / rho_safe**2) * np.sin(theta) * np.sin(phi)
        JC[2, 5] = - (1.0 / rho_safe**2) * np.cos(theta)

        cov_3d = JC @ Pll @ JC.T
        Cov_ZX = np.array([[cov_3d[2, 2], cov_3d[2, 0]], [cov_3d[0, 2], cov_3d[0, 0]]])
        
        # Plot estimated landmark
        ax.scatter(p_l_world[2], p_l_world[0], c='orange', marker='*', s=100)
        ax.text(p_l_world[2] + 0.1, p_l_world[0] + 0.1, f"L{patch.id}", color='orange', fontsize=8, weight='bold')
        
        # Only draw ellipse if covariance is numerically sound
        if not np.any(np.isnan(Cov_ZX)):
            draw_covariance_ellipse(ax, [p_l_world[2], p_l_world[0]], Cov_ZX, color='orange')

    ax.set_title("EKF Real-Time Top-Down Map (Z-X Plane)")
    ax.set_xlabel("Z (Forward) [meters]")
    ax.set_ylabel("X (Horizontal) [meters]")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Auto-scrolling camera window
    ax.set_xlim(pos_est[2] - 2, pos_est[2] + 8)
    ax.set_ylim(pos_est[0] - 5, pos_est[0] + 5)
    ax.legend(loc='upper left', fontsize=8)

    fig_map.tight_layout()
    plt.pause(0.005)


# -----------------------------------------------------------------------------
# 3. Main Webcam SLAM Loop

if __name__ == "__main__":
    # Open default system webcam
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("Error: Could not access the system webcam.")
        sys.exit()

    print("\n" + "="*60)
    print("  LAUCHING REAL-TIME WEBCAM VISUAL EKF SLAM")
    print("  Instructions:")
    print("    - Wave the camera slowly left and right to build parallax.")
    print("    - Press 'q' inside the OpenCV window to exit.")
    print("="*60 + "\n")

    # Time step interval
    dt = 0.033 # ~30 FPS
    max_active_features = 20 # Bounded track count for smooth 30 FPS updates

    # Pixel measurement noise covariance (real-world cameras have higher noise)
    R = np.array([[1.5, 0.0], [0.0, 1.5]])

    # Prior inverse depth settings
    rho0 = 0.50      # Assume features start ~2 meters away
    sigma_rho = 0.50 # High initial depth uncertainty

    miss_count_max = 6

    # Focal length and principal point
    # Generic pinhole values designed to fit a standard webcam resized to 640x480
    # Put this in main_webcam.py (around line 277) to use your custom calibration:
    f = 647.35 # Calibrated average focal length
    u0 = 315.15 # Calibrated principal point X
    v0 = 258.71 # Calibrated principal point Y
    img_width = 640
    img_height = 480
    
    # (Optional) If you want to undistort raw webcam frames before processing:
    # camera_matrix = np.array([[645.3043077974896, 0.0, 315.15228589833805], [0.0, 649.3999776308389, 258.7056674004233], [0.0, 0.0, 1.0]])
    # dist_coeffs = np.array([[0.13232068882952913, -0.333866199965113, 0.001770148892098156, -0.008927428347751643, 0.33581345968511733]])
    # frame_resized = cv2.undistort(frame_resized, camera_matrix, dist_coeffs)

    # Initialize EKF state vector s and covariance P
    # Estimates camera velocity and trajectory dynamically solely from visual tracks!
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Shape (13,)
    P = np.eye(13) * 1e-4

    active_patches = []
    trajectory_est = []

    # Map visualization setup
    plt.ion()
    fig_map = plt.figure(figsize=(7, 7))

    step = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # 1. Resize color frame to 640x480 so it matches our camera intrinsics perfectly
        frame_resized = cv2.resize(frame, (img_width, img_height))
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # 2. Filter Prediction (expected velocity/rate propagation)
        s[0:13] = np.array(robot_state_update(s[0:13], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt, 1e-6)).flatten()
        F = np.array(robot_state_update_jacobian(s[0:13], 0.0, 0.0, 0.0, dt, 1e-6))
        
        Prr = P[:13, :13]
        Q = np.diag([
            1e-3, 1e-3, 1e-3,          # Position uncertainty grows
            1e-4, 1e-4, 1e-4, 1e-4,    # Orientation uncertainty grows
            5e-3, 5e-3, 5e-3,          # Velocity uncertainty grows
            5e-3, 5e-3, 5e-3           # Angular rate uncertainty grows
        ])

        P[:13, :13] = F @ Prr @ F.T + Q
        if P.shape[0] > 13:
            P[13:, :13] = P[13:, :13] @ F.T
            P[:13, 13:] = P[13:, :13].T

        # 3. Active Feature Tracking & EKF Updates
        patches_to_delete_ids = []
        
        for patch in active_patches:
            idx = 13 + 6 * patch.id
            l_state = s[idx : idx+6]

            # Compute prediction
            h = np.array(landmark_measurement(s[0:13], f, 1.0, 1.0, u0, v0, *l_state)).flatten()
            Gr, Gl = landmark_measurement_jacobian(s[0:13], f, 1.0, 1.0, u0, v0, *l_state)
            Hr = np.array(Gr).reshape((2, 13))
            Hl = np.array(Gl).reshape((2, 6))

            Prr = P[:13, :13]       
            Pmr = P[13:, :13]       
            Prl = P[:13, idx:idx+6] 
            Plr = P[idx:idx+6, :13] 
            Pll = P[idx:idx+6, idx:idx+6] 
            Pml = P[13:, idx:idx+6] 

            temp_r = Hr @ Prr + Hl @ Plr  
            temp_l = Hr @ Prl + Hl @ Pll  
            Z = temp_r @ Hr.T + temp_l @ Hl.T + R 

            # Active Search using NCC Template matching inside EKF predicted uncertainty area
            z_meas = active_patch_search(frame_gray, patch, h, Z)

            if z_meas is not None:
                # Update visual track coordinates
                patch.current_u, patch.current_v = z_meas[0], z_meas[1]
                patch.miss_count = 0

                # Kalman Gain
                K_num_top = Prr @ Hr.T + Prl @ Hl.T   
                K_num_bottom = Pmr @ Hr.T + Pml @ Hl.T 

                Z_inv = np.linalg.inv(Z)
                K_top = K_num_top @ Z_inv       
                K_bottom = K_num_bottom @ Z_inv 
                K = np.vstack([K_top, K_bottom]) 
                
                # State correction
                y_res = z_meas - h
                s = (s + K @ y_res).flatten()
                
                # Covariance correction
                H_full = np.zeros((2, s.size))
                H_full[:, 0:13] = Hr
                H_full[:, idx:idx+6] = Hl
                P = (np.eye(s.size) - K @ H_full) @ P

                # Draw track cross and label directly on color webcam feed
                cv2.drawMarker(frame_resized, (int(z_meas[0]), int(z_meas[1])), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
                cv2.putText(frame_resized, f"L{patch.id}", (int(z_meas[0])+8, int(z_meas[1])-8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Draw cyan active bounding box
                search_half_w = int(np.ceil(3.0 * np.sqrt(Z[0, 0])))
                search_half_h = int(np.ceil(3.0 * np.sqrt(Z[1, 1])))
                search_half_w = max(15, search_half_w)
                search_half_h = max(15, search_half_h)
                cv2.rectangle(frame_resized, 
                              (int(h[0]) - search_half_w, int(h[1]) - search_half_h),
                              (int(h[0]) + search_half_w, int(h[1]) + search_half_h),
                              (255, 255, 0), 1)
            else:
                patch.miss_count += 1
                if patch.miss_count > miss_count_max:
                    patches_to_delete_ids.append(patch.id)

        # Delete lost features
        for landmark_id in sorted(patches_to_delete_ids, reverse=True):
            s, P, active_patches = delete_landmark_from_ekf(s, P, landmark_id, active_patches)

        # 4. Spatially-Uniform Bucketed Corner Detection
        if len(active_patches) < max_active_features:
            new_corners = initialize_new_features_bucketed(frame_gray, active_patches, target_features=max_active_features)
            
            for corner in new_corners:
                if len(active_patches) >= max_active_features:
                    break
                    
                u_new, v_new = corner[0], corner[1]
                
                # Enforce safety margin so we can safely extract an 11x11 patch
                margin = 7
                if not (margin <= u_new <= img_width - margin and margin <= v_new <= img_height - margin):
                    continue
                
                landmark_id = len(active_patches)
                
                # Initialize state variables
                Y_new = np.array(landmark_initialization(s[0:13], u_new, v_new, rho0, u0, v0, f, 1e-9)).flatten()

                Gr, Gy, Gs = landmark_initialization_jacobian(s[0:13], u_new, v_new, rho0, u0, v0, f, 1e-9)
                Gs = Gs.reshape((6, 1))

                S = np.asarray([[sigma_rho**2]])

                # 4. Compute new covariance blocks
                P_ll = Gr @ P[0:13, 0:13] @ Gr.T + Gy @ R @ Gy.T + Gs @ S @ Gs.T # New landmark self-covariance (6x6)
                P_rl = P[:, 0:13] @ Gr.T              
                
                # Expand s and P block-wise
                s = np.concatenate([s, Y_new])
                P_top = np.hstack([P, P_rl])
                P_bottom = np.hstack([P_rl.T, P_ll])
                P = np.vstack([P_top, P_bottom])
                
                new_patch = LandmarkPatch(landmark_id, frame_gray, u_new, v_new)
                active_patches.append(new_patch)

        # 5. Display Live Webcam Feed
        cv2.imshow("Visual SLAM Frontend (webcam)", frame_resized)
        
        # Save estimated trajectory
        trajectory_est.append(s[0:3].copy())

        # Render 2D Top-Down Estimated Map periodically
        if step % 3 == 0:
            plot_webcam_slam_map(fig_map, s, P, active_patches, trajectory_est, f, u0, v0, step)

        # Standard OpenCV waitKey to process UI events and capture 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        step += 1

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    print("\nVisual Webcam SLAM finished successfully!\n")
