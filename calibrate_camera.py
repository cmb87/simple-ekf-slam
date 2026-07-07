#!/usr/bin/env python3
"""
Webcam Camera Calibration Script
--------------------------------
Calibrates a webcam using a chessboard pattern to compute camera intrinsics (K) and distortion coefficients.
Saves the results to a JSON file so they can be loaded by other applications (like main_webcam.py).

Features:
- Live chessboard corner detection and visualization.
- Interactive image capture (SPACE/C to capture, Q/ESC to calibrate and save).
- Automatic camera fallback selection.
- Subpixel corner refinement for maximum accuracy.
- Computes mean reprojection error.
- Calibration testing/verification mode (visualize undistorted stream).
- Generates directly copy-pasteable Python code for main_webcam.py.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser(description="Webcam Chessboard Calibration")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners along the chessboard width (default: 9)")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners along the chessboard height (default: 6)")
    parser.add_argument("--square_size", type=float, default=23.0, help="Chessboard square size in mm or other units (default: 25.0)")
    parser.add_argument("--camera", type=int, default=4, help="Webcam device index (default: 4 to match main_webcam.py)")
    parser.add_argument("--width", type=int, default=640, help="Target image width for calibration (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Target image height for calibration (default: 480)")
    parser.add_argument("--output", type=str, default="camera_intrinsics.json", help="Path to save camera intrinsics (default: camera_intrinsics.json)")
    parser.add_argument("--test", action="store_true", help="Test calibration by showing real-time undistorted feed from a saved calibration file")

    args = parser.parse_args()

    if args.test:
        test_calibration(args)
    else:
        run_calibration(args)

def run_calibration(args):
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(cols-1,rows-1,0)
    # Scaled by the physical size of chessboard square
    pattern_size = (args.cols, args.rows)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_size

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    # Attempt to open the webcam
    print(f"Attempting to open camera with index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {args.camera}. Scanning for other active cameras...")
        found_cam = False
        # Scan standard camera indices
        for test_idx in [0, 1, 2, 3, 5, 6, 7]:
            cap = cv2.VideoCapture(test_idx)
            if cap.isOpened():
                print(f"Successfully opened camera {test_idx} instead!")
                found_cam = True
                break
        if not found_cam:
            print("Error: No active webcams could be opened.")
            sys.exit(1)

    # Set camera resolution if supported natively (or we will resize anyway to guarantee)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Termination criteria for subpixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n" + "="*70)
    print("  CAMERA CALIBRATION TOOL ACTIVE")
    print(f"  Target Resolution: {args.width}x{args.height}")
    print(f"  Chessboard Grid: {args.cols}x{args.rows} (cols x rows inner corners)")
    print(f"  Square Size: {args.square_size} units")
    print("\n  Instructions:")
    print("    - Position the chessboard in front of the camera.")
    print("    - When corners are detected (colored lines overlay), press SPACE or C to capture.")
    print("    - Capture 10-20 frames from different angles, distances, and screen positions.")
    print("    - Press Q or ESC to process calibration and exit.")
    print("="*70 + "\n")

    captured_frames_count = 0
    saved_frames = [] # keep the raw frames for final check/undistortion visual

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Resize to guarantee target resolution
        frame_resized = cv2.resize(frame, (args.width, args.height))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_corners, corners = cv2.findChessboardCorners(
            gray, pattern_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # Create copy for display
        display_frame = frame_resized.copy()

        status_text = "Chessboard: NOT FOUND"
        status_color = (0, 0, 255) # Red

        # If found, refine corners and draw them
        if ret_corners:
            status_text = "Chessboard: DETECTED"
            status_color = (0, 255, 0) # Green
            
            # Refine corner positions for subpixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_frame, pattern_size, corners_refined, ret_corners)

        # Add visual overlay instructions
        cv2.putText(display_frame, f"Captured: {captured_frames_count} frames", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "[SPACE/C] Capture  [Q] Calibrate & Exit", (15, args.height - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display detection status in top-right corner
        cv2.putText(display_frame, status_text, (args.width - 240, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow("Webcam Calibration", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord(' '), ord('c'), ord('C')]:
            if ret_corners:
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                saved_frames.append(gray)
                captured_frames_count += 1
                print(f"[*] Captured frame {captured_frames_count}! (Corners successfully recorded)")
            else:
                print("[!] Cannot capture: Chessboard corners not found in this frame!")
        elif key in [ord('q'), ord('Q'), 27]: # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frames_count < 3:
        print(f"\n[!] Calibration cancelled. You only captured {captured_frames_count} frames. At least 3 (preferably 10+) are required.")
        sys.exit(0)

    print("\n" + "-"*50)
    print("  RUNNING CAMERA CALIBRATION ALGORITHM...")
    print("-"*50)

    # Calibrate camera
    ret_val, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (args.width, args.height), None, None
    )

    if not ret_val:
        print("[Error] Calibration failed.")
        sys.exit(1)

    # Calculate mean reprojection error
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += error
        total_points += len(objpoints[i])
    
    mean_error = np.sqrt(total_error / total_points)

    print("\n>>> CALIBRATION RESULTS <<<")
    print("Status: Success!")
    print(f"Target Frame Size: {args.width} x {args.height}")
    print(f"Number of frames used: {captured_frames_count}")
    print(f"RMS Reprojection Error: {mean_error:.4f} pixels (Values < 0.5 are excellent, < 1.0 is standard)")
    
    # Extract components
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    avg_f = (fx + fy) / 2.0

    print("\n--- Intrinsic Matrix (K) ---")
    print(f"[{fx:10.4f} {mtx[0,1]:10.4f} {cx:10.4f}]")
    print(f"[{mtx[1,0]:10.4f} {fy:10.4f} {cy:10.4f}]")
    print(f"[{mtx[2,0]:10.4f} {mtx[2,1]:10.4f} {mtx[2,2]:10.4f}]")

    print("\n--- Calibration Constants for EKF-SLAM ---")
    print(f"Average Focal Length (f) : {avg_f:.4f}")
    print(f"Principal Point X (u0)    : {cx:.4f}")
    print(f"Principal Point Y (v0)    : {cy:.4f}")

    # Distortion coefficients: k1, k2, p1, p2, k3
    dist_flat = dist.ravel()
    print("\n--- Distortion Coefficients (D) ---")
    print(f"k1: {dist_flat[0]:.6f}")
    print(f"k2: {dist_flat[1]:.6f}")
    print(f"p1: {dist_flat[2]:.6f}")
    print(f"p2: {dist_flat[3]:.6f}")
    if len(dist_flat) > 4:
        print(f"k3: {dist_flat[4]:.6f}")

    # Prepare data to save
    calibration_data = {
        "width": args.width,
        "height": args.height,
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "rms_reprojection_error": float(mean_error),
        "f_x": float(fx),
        "f_y": float(fy),
        "f_avg": float(avg_f),
        "cx": float(cx),
        "cy": float(cy)
    }

    # Save to file
    with open(args.output, 'w') as f_out:
        json.dump(calibration_data, f_out, indent=4)
    print(f"\n[+] Saved calibration results to: {os.path.abspath(args.output)}")

    # Print code snippet
    print("\n" + "="*70)
    print("  COPY-PASTE READY CODE SNIPPET FOR main_webcam.py")
    print("="*70)
    print(f"""
    # Put this in main_webcam.py (around line 277) to use your custom calibration:
    f = {avg_f:.2f} # Calibrated average focal length
    u0 = {cx:.2f} # Calibrated principal point X
    v0 = {cy:.2f} # Calibrated principal point Y
    img_width = 640
    img_height = 480
    
    # (Optional) If you want to undistort raw webcam frames before processing:
    # camera_matrix = np.array({mtx.tolist()})
    # dist_coeffs = np.array({dist.tolist()})
    # frame_resized = cv2.undistort(frame_resized, camera_matrix, dist_coeffs)
""")
    print("="*70 + "\n")


def test_calibration(args):
    if not os.path.exists(args.output):
        print(f"Error: Calibration file '{args.output}' does not exist.")
        print("Please run calibration first (without --test option) to generate it.")
        sys.exit(1)

    # Load calibration parameters
    try:
        with open(args.output, 'r') as f_in:
            calib = json.load(f_in)
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        sys.exit(1)

    print(f"\n[+] Loaded calibration from {args.output}")
    print(f"Calibration resolution: {calib['width']}x{calib['height']}")
    print(f"RMS error: {calib['rms_reprojection_error']:.4f}")

    mtx = np.array(calib['camera_matrix'])
    dist = np.array(calib['distortion_coefficients'])

    print("\nAttempting to open camera...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {args.camera}. Scanning for other active cameras...")
        found_cam = False
        for test_idx in [0, 1, 2, 3, 5, 6, 7]:
            cap = cv2.VideoCapture(test_idx)
            if cap.isOpened():
                print(f"Successfully opened camera {test_idx}!")
                found_cam = True
                break
        if not found_cam:
            print("Error: No active webcams could be opened.")
            sys.exit(1)

    # Setup windows
    cv2.namedWindow("Calibration Test", cv2.WINDOW_AUTOSIZE)
    
    print("\n" + "="*70)
    print("  CALIBRATION VERIFICATION FEED RUNNING")
    print("  Instructions:")
    print("    - Look at both the raw and undistorted images.")
    print("    - Straight lines in the physical world (like edges of a door, window, etc.)")
    print("      should now appear perfectly straight in the Undistorted window.")
    print("    - Press 'SPACE' to toggle between RAW and UNDISTORTED views.")
    print("    - Press 'Q' or 'ESC' to exit.")
    print("="*70 + "\n")

    show_undistorted = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Resize to match calibration resolution
        frame_resized = cv2.resize(frame, (calib['width'], calib['height']))

        if show_undistorted:
            # Undistort the frame
            processed_frame = cv2.undistort(frame_resized, mtx, dist)
            mode_text = "Mode: UNDISTORTED (press SPACE to show Raw)"
            text_color = (0, 255, 0)
        else:
            processed_frame = frame_resized.copy()
            mode_text = "Mode: RAW (press SPACE to show Undistorted)"
            text_color = (0, 0, 255)

        # Draw info
        cv2.putText(processed_frame, mode_text, (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(processed_frame, "[Q] Exit", (15, calib['height'] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Calibration Test", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord(' '), ord('t'), ord('T')]:
            show_undistorted = not show_undistorted
        elif key in [ord('q'), ord('Q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
