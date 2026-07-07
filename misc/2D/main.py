# ruff: noqa: E402
import sys
import numpy as np
from pathlib import Path

# Add the path to the gen_ekf directory directly to avoid shadowing of the global symforce/sym packages
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir / "gen_ekf" / "python" / "symforce"))

from sym.process_model import process_model
from sym.measurement_model import measurement_model

# Configure Matplotlib for headless environments (saves plot to file)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

def wrap_angle(angle):
    """Wraps the input angle (or array of angles) to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def draw_covariance_ellipse(ax, mean, cov, n_std=3.0, edgecolor='r', **kwargs):
    """Draws an uncertainty ellipse representing the covariance of a 2D position."""
    # Eigenvalue decomposition of 2D covariance
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    # Rotation angle in degrees
    theta_deg = np.rad2deg(np.arctan2(vecs[1, 0], vecs[0, 0]))
    
    # Width and height of ellipse (diameters: 2 * n_std * std_dev)
    width = 2.0 * n_std * np.sqrt(max(vals[0], 1e-9))
    height = 2.0 * n_std * np.sqrt(max(vals[1], 1e-9))
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta_deg,
                      edgecolor=edgecolor, facecolor='none', **kwargs)
    ax.add_patch(ellipse)

def get_landmark_cartesian_covariance(xi, yi, alpha, rho, P_LL):
    """Converts 4D landmark inverse depth covariance block to 2D Cartesian covariance."""
    rho_clamped = max(rho, 1e-3)
    c = np.cos(alpha)
    s = np.sin(alpha)
    
    # J is 2x4 Jacobian of the transformation:
    # x_L = xi + cos(alpha)/rho
    # y_L = yi + sin(alpha)/rho
    # wrt [xi, yi, alpha, rho]
    J = np.array([
        [1.0, 0.0, -s / rho_clamped, -c / (rho_clamped**2)],
        [0.0, 1.0,  c / rho_clamped, -s / (rho_clamped**2)]
    ])
    
    return J @ P_LL @ J.T

def run_ekf_slam():
    # Simulation Parameters
    np.random.seed(42)  # For reproducible simulation results
    NLandmarks = 10
    NSteps = 150
    dt = 0.1

    # Generate True Landmarks (Ground Truth 2D positions)
    # Placed randomly in a 15m x 15m box centered around the origin
    xl_true = (np.random.rand(NLandmarks, 2) - 0.5) * 15.0

    # True Robot State trajectory initialization
    xr_true = np.array([0.0, -10.0, 0.0])  # [x, y, theta]

    # Nominal control inputs (moving the robot in a circular/arc path)
    u_nominal = np.array([1.5, 0.25])  # [v (m/s), omega (rad/s)]

    # Process and Measurement Noise Covariances
    # Q: Process noise on speed (v) and angular velocity (omega)
    Q = np.diag([0.05**2, (1.0 * np.pi / 180.0)**2])
    # R_val: Measurement noise variance for bearing observations
    R_val = (1.5 * np.pi / 180.0)**2

    # EKF State and Covariance Initialization
    # mu: Initial state vector containing [x, y, theta]. Landmarks are added dynamically.
    mu = xr_true.copy()  # Start with robot's initial pose
    # P: Initial covariance matrix (low uncertainty on the robot's starting pose as the map origin)
    P = np.diag([1e-4, 1e-4, 1e-6])

    # Dynamic landmark index mapping: landmark_id (0..NLandmarks-1) -> state index in mu
    landmark_idx_in_state = {}

    # Inverse Depth Hyperparameters for Initialization
    rho_init = 0.2          # Initial inverse depth guess (distance = 1 / 0.2 = 5 meters)
    sigma_rho = 0.1         # Uncertainty of initial inverse depth guess

    # Trajectory logging lists
    xr_true_history = []
    xr_est_history = []
    mu_history = []
    landmark_idx_history = []
    P_history = []

    print("Starting Bearing-Only EKF SLAM Simulation...")

    for step in range(NSteps):
        # ---------------------------------------------------------
        # 1. True Motion Simulation (with noise)
        # ---------------------------------------------------------
        u_true = u_nominal + np.random.normal(0, np.sqrt(np.diag(Q)))
        xr_true[0] += np.cos(xr_true[2]) * u_true[0] * dt
        xr_true[1] += np.sin(xr_true[2]) * u_true[0] * dt
        xr_true[2] = wrap_angle(xr_true[2] + u_true[1] * dt)
        
        xr_true_history.append(xr_true.copy())

        # ---------------------------------------------------------
        # 2. EKF Prediction Step (using nominal control)
        # ---------------------------------------------------------
        # Predict next robot state and compute process Jacobians
        mu_robot_next, F, W = process_model(mu[0:3], u_nominal, dt)
        
        # Update robot state in the state vector
        mu[0:3] = mu_robot_next
        mu[2] = wrap_angle(mu[2])

        # Predict Covariance using the EKF block structure (since landmarks are static)
        P[0:3, 0:3] = F @ P[0:3, 0:3] @ F.T + W @ Q @ W.T
        if len(P) > 3:
            # Cross-covariance blocks
            P[0:3, 3:] = F @ P[0:3, 3:]
            P[3:, 0:3] = P[0:3, 3:].T

        # ---------------------------------------------------------
        # 3. EKF Update Step (for each visible landmark)
        # ---------------------------------------------------------
        for i in range(NLandmarks):
            # Compute true relative displacement
            dx = xl_true[i, 0] - xr_true[0]
            dy = xl_true[i, 1] - xr_true[1]
            dist = np.hypot(dx, dy)

            # Sensors only observe landmarks within a range of 15 meters and a 60-degree FOV (front +/- 30 degrees)
            if dist > 15.0:
                continue

            # Compute true bearing relative to current heading
            z_true = np.arctan2(dy, dx) - xr_true[2]
            z_true_wrapped = wrap_angle(z_true)

            if np.abs(z_true_wrapped) > np.deg2rad(90.0):
                continue

            z = wrap_angle(z_true_wrapped + np.random.normal(0, np.sqrt(R_val)))

            if i not in landmark_idx_in_state:
                # ----------------------------------------------------------------------
                # LANDMARK INITIALIZATION (First Observation with Inverse Depth Prior)
                # ----------------------------------------------------------------------
                # Math explanation:
                # Since bearing-only measurements do not provide range (depth), depth is initially unobservable.
                # We use Inverse Depth Parametrization to represent the landmark:
                #   L_i = [x_i, y_i, alpha_i, rho_i]^T
                # where:
                #   - (x_i, y_i) is the robot position when the landmark is first observed (acting as an anchor).
                #   - alpha_i is the absolute bearing angle of the landmark relative to the anchor coordinate frame:
                #     alpha_i = theta + z
                #   - rho_i is the inverse depth (1 / distance) from the anchor position.
                #
                # Range Prior:
                #   We use a prior on the inverse depth: rho_init = 0.2 (which means initial depth = 5 meters),
                #   with an initial standard deviation sigma_rho = 0.1.
                #   The 3-sigma range of this prior is rho_i in [-0.1, 0.5].
                #   Crucially, this includes rho_i = 0, which represents a landmark at infinity (e.g. infinite range).
                #   This formulation is highly stable and allows EKF to handle distant/unbounded landmarks.
                # ----------------------------------------------------------------------
                idx = len(mu)
                landmark_idx_in_state[i] = idx

                # Initialize landmark state block: L_init = [x_r, y_r, theta_r + z, rho_init]
                z_wrapped = wrap_angle(z)
                L_init = np.array([mu[0], mu[1], wrap_angle(mu[2] + z_wrapped), rho_init])
                mu = np.concatenate([mu, L_init])

                # ----------------------------------------------------------------------
                # First-Order Covariance Propagation for State Expansion:
                # ----------------------------------------------------------------------
                # Let g(X_r, z, rho_0) be the initialization function mapping the current robot pose X_r,
                # the measurement z, and the inverse depth prior rho_0 to the new landmark state L_i:
                #   g(X_r, z, rho_0) = [ x_r, y_r, theta_r + z, rho_0 ]^T
                #
                # The Jacobians of g with respect to:
                #   - Current full state vector (Y): G_Y = [ I_(3x3), 0 ] of shape 4 x len(P)
                #   - Measurement noise (z):         G_z = [ 0, 0, 1, 0 ]^T
                #   - Inverse depth prior (rho_0):   G_rho = [ 0, 0, 0, 1 ]^T
                #
                # The expanded state covariance P_new is derived via first-order Taylor expansion:
                #   P_new = [ P       , P * G_Y^T ]
                #           [ G_Y * P , P_LL      ]
                # where:
                #   P_LL = G_Y * P * G_Y^T + G_z * R * G_z^T + G_rho * sigma_rho^2 * G_rho^T
                #
                # Simplifying analytically:
                #   - G_Y * P simply copies the first 3 rows of P (robot state block) and appends a row of zeros.
                #   - P * G_Y^T is the transpose of (G_Y * P).
                #   - P_LL is the top-left 3x3 block of P, with measurement noise R_val added to the angle term
                #     and sigma_rho^2 assigned to the inverse depth term:
                #     P_LL = [ P[0:3, 0:3]           0 ]
                #            [ 0           sigma_rho^2 ]   where P[0:3, 0:3][2,2] gets R_val added.
                # ----------------------------------------------------------------------
                
                # G_YP is G_Y * P of shape 4 x len(P)
                G_YP = np.zeros((4, len(P)))
                G_YP[0:3, 0:3] = np.eye(3)
                G_YP[0:3, :] = P[0:3, :]

                PG_YT = G_YP.T

                # Compute landmark covariance block P_LL
                P_LL = np.zeros((4, 4))
                P_LL[0:3, 0:3] = P[0:3, 0:3]
                P_LL[2, 2] += R_val
                P_LL[3, 3] = sigma_rho**2

                # Expand the covariance matrix P
                P_new = np.zeros((len(P) + 4, len(P) + 4))
                P_new[0:len(P), 0:len(P)] = P
                P_new[0:len(P), len(P):] = PG_YT
                P_new[len(P):, 0:len(P)] = G_YP
                P_new[len(P):, len(P):] = P_LL

                P = P_new
            else:
                # ----------------------------------------------------------------------
                # LANDMARK UPDATE (Known Landmark Observation)
                # ----------------------------------------------------------------------
                # Math explanation:
                # For an already initialized landmark, we compute the predicted measurement z_pred:
                #   z_pred = atan2(v_y, v_x) - theta_r
                # where relative coordinates v_x, v_y are scaled by inverse depth rho to prevent singularities:
                #   v_x = rho_i * (x_anchor - x_r) + cos(alpha_i)
                #   v_y = rho_i * (y_anchor - y_r) + sin(alpha_i)
                #
                # Jacobians:
                #   H_X (1x3) is the Jacobian of z_pred wrt current robot state [x_r, y_r, theta_r]
                #   H_L (1x4) is the Jacobian of z_pred wrt landmark state [x_i, y_i, alpha_i, rho_i]
                #
                # Full H Matrix (1 x state_dim):
                #   H = [ H_X , 0 , ... , H_L , ... , 0 ]
                # ----------------------------------------------------------------------
                idx = landmark_idx_in_state[i]

                # Compute predicted measurement and measurement Jacobians via SymForce generated models
                z_pred, H_X, H_L = measurement_model(mu[0:3], mu[idx : idx+4])

                # Construct full H matrix (1 x state_dim)
                H = np.zeros((1, len(mu)))
                H[0, 0:3] = H_X
                H[0, idx : idx+4] = H_L

                # Standard EKF Kalman Filter correction equations:
                #   S = H * P * H^T + R      (Innovation Covariance)
                #   K = P * H^T * S^-1       (Kalman Gain)
                #   y = z - z_pred           (Innovation / Residual)
                #   mu = mu + K * y          (State Update)
                #   P = (I - K * H) * P      (Covariance Update)
                S = H @ P @ H.T + R_val
                K = P @ H.T / S

                # Innovation
                innov = wrap_angle(z - z_pred[0])

                # State Correction
                mu = mu + K.flatten() * innov

                # Wrap angles in state vector (robot theta and all landmark bearings)
                mu[2] = wrap_angle(mu[2])
                for lm_id, lm_idx in landmark_idx_in_state.items():
                    mu[lm_idx + 2] = wrap_angle(mu[lm_idx + 2])

                # Covariance Correction
                P = (np.eye(len(mu)) - K @ H) @ P

        # Symmetrize and bound covariance to maintain numerical stability
        P = 0.5 * (P + P.T)
        
        xr_est_history.append(mu[0:3].copy())
        mu_history.append(mu.copy())
        landmark_idx_history.append(landmark_idx_in_state.copy())
        P_history.append(P.copy())

    print("Simulation finished. Processing results and generating plot...")

    # ---------------------------------------------------------
    # 4. Process Estimates and Plot Results
    # ---------------------------------------------------------
    xr_true_history = np.array(xr_true_history)
    xr_est_history = np.array(xr_est_history)

    # Convert estimated landmark states (Inverse Depth) to 2D Cartesian positions
    xl_est = np.zeros((NLandmarks, 2))
    for i in range(NLandmarks):
        if i in landmark_idx_in_state:
            idx = landmark_idx_in_state[i]
            xi, yi, alpha, rho = mu[idx : idx+4]
            # Avoid division by zero by clamping inverse depth
            rho_clamped = max(rho, 1e-3)
            x_est = xi + np.cos(alpha) / rho_clamped
            y_est = yi + np.sin(alpha) / rho_clamped
            xl_est[i] = [x_est, y_est]
        else:
            xl_est[i] = [np.nan, np.nan]

    # Create Plot
    fig_static, ax_static = plt.subplots(figsize=(11, 9))
    ax_static.plot(xr_true_history[:, 0], xr_true_history[:, 1], 'g-', linewidth=2.0, label='True Trajectory')
    ax_static.plot(xr_est_history[:, 0], xr_est_history[:, 1], 'r--', linewidth=2.0, label='Estimated Trajectory')

    # Draw Robot Uncertainty Ellipses (every 20 steps)
    first_ellipse = True
    for step_idx in range(0, NSteps, 20):
        P_step = P_history[step_idx]
        mean_step = xr_est_history[step_idx, 0:2]
        cov_step = P_step[0:2, 0:2]
        draw_covariance_ellipse(ax_static, mean_step, cov_step, n_std=3.0, edgecolor='b', linewidth=1.0, linestyle=':',
                                label='Robot 3σ Ellipse' if first_ellipse else "")
        first_ellipse = False

    # Plot True Landmarks
    ax_static.scatter(xl_true[:, 0], xl_true[:, 1], c='g', marker='x', s=120, linewidth=2.5, label='True Landmarks')

    # Plot Estimated Landmarks and draw lines back to their initial observation anchors
    first_scatter = True
    first_lm_ellipse = True
    for i in range(NLandmarks):
        if i in landmark_idx_in_state:
            idx = landmark_idx_in_state[i]
            # Initial robot position when first observed
            xi, yi, alpha, rho = mu[idx : idx+4]
            
            # Scatter plot estimated landmark
            ax_static.scatter(xl_est[i, 0], xl_est[i, 1], facecolors='none', edgecolors='r', marker='o', s=100, linewidth=2.0, 
                        label='Estimated Landmarks' if first_scatter else "")
            first_scatter = False
            
            # Draw dotted line showing initial reference frame anchor
            ax_static.plot([xi, xl_est[i, 0]], [yi, xl_est[i, 1]], 'k:', alpha=0.4)

            # Draw Landmark Uncertainty Ellipse
            P_LL = P[idx : idx+4, idx : idx+4]
            P_cart = get_landmark_cartesian_covariance(xi, yi, alpha, rho, P_LL)
            draw_covariance_ellipse(ax_static, xl_est[i], P_cart, n_std=3.0, edgecolor='r', linewidth=1.0, alpha=0.6,
                                    label='Landmark 3σ Ellipse' if first_lm_ellipse else "")
            first_lm_ellipse = False

    ax_static.set_title('Bearing-Only EKF SLAM with Inverse Depth Landmark Parametrization', fontsize=14)
    ax_static.set_xlabel('X (meters)', fontsize=12)
    ax_static.set_ylabel('Y (meters)', fontsize=12)
    ax_static.grid(True, linestyle='--', alpha=0.7)
    ax_static.legend(fontsize=11, loc='best')
    ax_static.set_aspect('equal')
    
    # Save output plot
    plot_path = current_dir / 'ekf_slam_results.png'
    plt.savefig(str(plot_path), dpi=150)
    plt.close(fig_static)
    print(f"Results plot saved successfully to {plot_path}")

    # ---------------------------------------------------------
    # 4.5 EKF Estimation Error and 3σ Bounds Plot
    # ---------------------------------------------------------
    print("Generating Matplotlib EKF estimation error and 3σ bounds plot...")
    P_diag_history = np.array([np.diag(P_step[0:3, 0:3]) for P_step in P_history])
    time_steps = np.arange(NSteps) * dt

    x_errors = xr_est_history[:, 0] - xr_true_history[:, 0]
    y_errors = xr_est_history[:, 1] - xr_true_history[:, 1]
    theta_errors = wrap_angle(xr_est_history[:, 2] - xr_true_history[:, 2])

    sigma_x = np.sqrt(P_diag_history[:, 0])
    sigma_y = np.sqrt(P_diag_history[:, 1])
    sigma_theta = np.sqrt(P_diag_history[:, 2])

    fig_err, axs = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # X error plot
    axs[0].plot(time_steps, x_errors, 'b-', linewidth=1.5, label='X Estimation Error')
    axs[0].plot(time_steps, 3 * sigma_x, 'r--', linewidth=1.2, label='+3σ Uncertainty Bound')
    axs[0].plot(time_steps, -3 * sigma_x, 'r--', linewidth=1.2)
    axs[0].set_ylabel('X Error (meters)', fontsize=11)
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend(loc='upper right')
    axs[0].set_title('EKF SLAM Robot Pose Estimation Errors and 3σ Bounds', fontsize=13)

    # Y error plot
    axs[1].plot(time_steps, y_errors, 'g-', linewidth=1.5, label='Y Estimation Error')
    axs[1].plot(time_steps, 3 * sigma_y, 'r--', linewidth=1.2, label='+3σ Uncertainty Bound')
    axs[1].plot(time_steps, -3 * sigma_y, 'r--', linewidth=1.2)
    axs[1].set_ylabel('Y Error (meters)', fontsize=11)
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc='upper right')

    # Theta error plot
    axs[2].plot(time_steps, np.rad2deg(theta_errors), 'm-', linewidth=1.5, label='θ Estimation Error')
    axs[2].plot(time_steps, np.rad2deg(3 * sigma_theta), 'r--', linewidth=1.2, label='+3σ Uncertainty Bound')
    axs[2].plot(time_steps, -np.rad2deg(3 * sigma_theta), 'r--', linewidth=1.2)
    axs[2].set_ylabel('θ Error (degrees)', fontsize=11)
    axs[2].set_xlabel('Time (seconds)', fontsize=11)
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend(loc='upper right')

    plt.tight_layout()
    err_plot_path = current_dir / 'ekf_slam_uncertainty.png'
    plt.savefig(str(err_plot_path), dpi=150)
    plt.close(fig_err)
    print(f"Uncertainty error plot saved successfully to {err_plot_path}")

    # ---------------------------------------------------------
    # 5. Create and Save Matplotlib Animation
    # ---------------------------------------------------------
    print("Generating Matplotlib EKF SLAM animation...")
    fig_anim, ax_anim = plt.subplots(figsize=(10, 8))

    def update_frame(frame):
        ax_anim.clear()
        
        # Plot Trajectories up to current frame
        ax_anim.plot(xr_true_history[:frame+1, 0], xr_true_history[:frame+1, 1], 'g-', linewidth=2.0, label='True Trajectory')
        ax_anim.plot(xr_est_history[:frame+1, 0], xr_est_history[:frame+1, 1], 'r--', linewidth=2.0, label='Estimated Trajectory')
        
        # Plot current robot position and its 3σ uncertainty ellipse
        ax_anim.plot(xr_true_history[frame, 0], xr_true_history[frame, 1], 'go', markersize=8)
        ax_anim.plot(xr_est_history[frame, 0], xr_est_history[frame, 1], 'ro', markersize=8)
        
        P_f = P_history[frame]
        draw_covariance_ellipse(ax_anim, xr_est_history[frame, 0:2], P_f[0:2, 0:2], n_std=3.0, edgecolor='b', linewidth=1.0, linestyle=':',
                                label='Robot 3σ Ellipse')
        
        # Plot True Landmarks
        ax_anim.scatter(xl_true[:, 0], xl_true[:, 1], c='g', marker='x', s=120, linewidth=2.5, label='True Landmarks')
        
        # Reconstruct and plot estimated landmarks at current frame
        mu_f = mu_history[frame]
        lm_map_f = landmark_idx_history[frame]
        
        first_scatter_anim = True
        first_ellipse_anim = True
        for i in range(NLandmarks):
            if i in lm_map_f:
                idx_f = lm_map_f[i]
                xi_f, yi_f, alpha_f, rho_f = mu_f[idx_f : idx_f+4]
                rho_clamped_f = max(rho_f, 1e-3)
                x_est_f = xi_f + np.cos(alpha_f) / rho_clamped_f
                y_est_f = yi_f + np.sin(alpha_f) / rho_clamped_f
                
                ax_anim.scatter(x_est_f, y_est_f, facecolors='none', edgecolors='r', marker='o', s=100, linewidth=2.0,
                                label='Estimated Landmarks' if first_scatter_anim else "")
                first_scatter_anim = False
                
                # Draw anchor line from observation origin
                ax_anim.plot([xi_f, x_est_f], [yi_f, y_est_f], 'k:', alpha=0.4)

                # Draw Landmark Uncertainty Ellipse
                P_LL_f = P_f[idx_f : idx_f+4, idx_f : idx_f+4]
                P_cart_f = get_landmark_cartesian_covariance(xi_f, yi_f, alpha_f, rho_f, P_LL_f)
                draw_covariance_ellipse(ax_anim, [x_est_f, y_est_f], P_cart_f, n_std=3.0, edgecolor='r', linewidth=1.0, alpha=0.5,
                                        label='Landmark 3σ Ellipse' if first_ellipse_anim else "")
                first_ellipse_anim = False
                
        ax_anim.set_title(f'Bearing-Only EKF SLAM (Step {frame+1}/{NSteps})', fontsize=14)
        ax_anim.set_xlabel('X (meters)', fontsize=12)
        ax_anim.set_ylabel('Y (meters)', fontsize=12)
        ax_anim.grid(True, linestyle='--', alpha=0.5)
        ax_anim.legend(fontsize=11, loc='upper left')
        
        # Fix axis limits to keep the view stable
        ax_anim.set_xlim(-12, 12)
        ax_anim.set_ylim(-12, 12)
        ax_anim.set_aspect('equal')

    anim = FuncAnimation(fig_anim, update_frame, frames=NSteps, interval=50)
    anim_path = current_dir / 'ekf_slam_animation.gif'
    anim.save(str(anim_path), writer='pillow', fps=20)
    plt.close(fig_anim)
    print(f"Animation saved successfully to {anim_path}")

if __name__ == "__main__":
    run_ekf_slam()
