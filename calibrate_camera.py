#!/usr/bin/env python3
"""
Camera calibration for Raspberry Pi HQ Camera (OpenCV checkerboard method).

Step 1 - capture images (on the Pi, with a display):
    python3 calibrate_camera.py capture
        - shows live preview with a FOCUS score (higher = sharper) -> use it to set the lens
        - green corners drawn when the checkerboard is found
        - press SPACE to save an image (only saved if the board is detected), q to quit
        - aim for 15-25 images: tilt the board, move it to corners/edges, near and far

Step 2 - compute calibration (on the Pi or your laptop):
    python3 calibrate_camera.py calibrate
        - writes calibration.npz with camera_matrix and dist_coeffs
        - reprojection error < 0.5 px is good, < 1.0 px is acceptable

Checkerboard: print one (e.g. 10x7 squares = 9x6 INNER corners), glue it flat to card,
measure one square with a ruler and pass --square-mm.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def capture(args):
    from picamera2 import Picamera2

    os.makedirs(args.dir, exist_ok=True)
    cam = Picamera2()
    # Use the SAME resolution you will use for detection - calibration is per resolution.
    cam.configure(cam.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}))
    cam.start()
    time.sleep(1)

    n = len(glob.glob(os.path.join(args.dir, "*.jpg")))
    pattern = (args.cols, args.rows)
    while True:
        frame = cam.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Focus measure: variance of Laplacian on the centre region
        h, w = gray.shape
        centre = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
        focus = cv2.Laplacian(centre, cv2.CV_64F).var()

        # Fast check so the preview stays smooth
        found, corners = cv2.findChessboardCorners(
            gray, pattern, flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH)

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        cv2.putText(view, f"focus={focus:.0f}  board={'YES' if found else 'no'}  saved={n}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("calibration capture", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if found:
                fn = os.path.join(args.dir, f"calib_{n:02d}.jpg")
                cv2.imwrite(fn, frame)  # save the clean frame, not the annotated one
                n += 1
                print("saved", fn)
            else:
                print("board not detected - not saved")

    cam.stop()
    cv2.destroyAllWindows()


def calibrate(args):
    pattern = (args.cols, args.rows)
    # 3D points of the board corners in its own plane (z=0), in mm
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm

    obj_points, img_points, size, used = [], [], None, []
    for fn in sorted(glob.glob(os.path.join(args.dir, "*.jpg"))):
        gray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not found:
            print("skip (no board):", fn)
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        obj_points.append(objp)
        img_points.append(corners)
        used.append(fn)

    if len(used) < 10:
        print(f"Only {len(used)} usable images - capture at least 10 (ideally 15-25).")
        return

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    # Per-image error, to spot bad shots (blurry / misdetected) worth deleting
    print("\nPer-image reprojection error:")
    for fn, op, ip, rv, tv in zip(used, obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        err = cv2.norm(ip, proj, cv2.NORM_L2) / np.sqrt(len(proj))
        flag = "  <-- consider deleting" if err > 1.0 else ""
        print(f"  {os.path.basename(fn)}: {err:.3f} px{flag}")

    print(f"\nImages used: {len(used)}   image size: {size}")
    print(f"Overall RMS reprojection error: {rms:.3f} px")
    print("Camera matrix K:\n", K)
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  (pixels)")
    print("Distortion coeffs (k1 k2 p1 p2 k3):\n", dist.ravel())

    np.savez(args.out, camera_matrix=K, dist_coeffs=dist, image_size=size, rms=rms)
    print("\nSaved", args.out)

    # Save an undistorted example so you can eyeball the result
    img = cv2.imread(used[0])
    cv2.imwrite("undistorted_example.jpg", cv2.undistort(img, K, dist))
    print("Wrote undistorted_example.jpg (straight lines should look straight)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "calibrate"])
    ap.add_argument("--dir", default="calib_images")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=25.0, help="measured square size")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default="calibration.npz")
    args = ap.parse_args()
    capture(args) if args.mode == "capture" else calibrate(args)


if __name__ == "__main__":
    main()
