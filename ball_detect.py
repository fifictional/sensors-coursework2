#!/usr/bin/env python3
"""
Colour-ball detection test for Raspberry Pi CM5 + HQ Camera.

Pipeline: capture -> BGR->HSV -> colour threshold -> clean mask
          -> find contours -> pick largest round blob -> report centre/radius

Usage:
    python3 ball_detect.py                 # preview window, default colour (blue)
    python3 ball_detect.py --colour red
    python3 ball_detect.py --headless      # no window (e.g. over SSH), prints positions

In the preview window:
    click on the ball  -> re-tunes the colour range to that object
    m                  -> toggle mask view (see what the threshold "sees")
    s                  -> save a snapshot
    q                  -> quit

Install on the Pi (Raspberry Pi OS):
    sudo apt install -y python3-picamera2 python3-opencv
"""
import argparse
import time

import cv2
import numpy as np
from picamera2 import Picamera2

# HSV ranges in OpenCV units: H 0-179, S 0-255, V 0-255.
# Red wraps around hue 0, so it needs two ranges.
PRESETS = {
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "orange": [((10, 120, 100), (25, 255, 255))],
    "yellow": [((22, 100, 100), (35, 255, 255))],
    "green":  [((40, 70, 50), (85, 255, 255))],
    "blue":   [((95, 120, 50), (130, 255, 255))],
    "pink":   [((140, 80, 80), (170, 255, 255))],
    # White = any hue, low saturation, high brightness
    "white":  [((0, 0, 180), (179, 50, 255))],
}

MIN_RADIUS_PX = 10        # ignore tiny blobs
MIN_CIRCULARITY = 0.6     # contour area / enclosing-circle area (1.0 = perfect circle)


def build_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove speckles
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill holes
    return mask


def find_ball(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < MIN_RADIUS_PX:
            break  # sorted by area, the rest are smaller
        circularity = cv2.contourArea(c) / (np.pi * r * r)
        if circularity >= MIN_CIRCULARITY:
            best = (int(x), int(y), int(r), circularity)
            break
    return best


def ranges_from_pixel(hsv, x, y, h_tol=10, s_tol=60, v_tol=60):
    """Sample a small patch around (x, y) and build an HSV range around it."""
    patch = hsv[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].reshape(-1, 3)
    h, s, v = np.median(patch, axis=0).astype(int)
    if s < 50:  # white / grey object: hue is meaningless, match on low saturation + brightness
        return [((0, 0, max(v - v_tol, 0)), (179, min(s + 40, 255), 255))]
    s_lo, s_hi = max(s - s_tol, 50), 255
    v_lo, v_hi = max(v - v_tol, 40), 255
    lo_h, hi_h = h - h_tol, h + h_tol
    if lo_h < 0:      # wrap around (reds)
        return [((0, s_lo, v_lo), (hi_h, s_hi, v_hi)), ((180 + lo_h, s_lo, v_lo), (179, s_hi, v_hi))]
    if hi_h > 179:
        return [((lo_h, s_lo, v_lo), (179, s_hi, v_hi)), ((0, s_lo, v_lo), (hi_h - 180, s_hi, v_hi))]
    return [((lo_h, s_lo, v_lo), (hi_h, s_hi, v_hi))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colour", default="blue", choices=PRESETS.keys())
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--headless", action="store_true", help="no preview window")
    args = ap.parse_args()

    state = {"ranges": PRESETS[args.colour], "hsv": None}

    cam = Picamera2()
    # "RGB888" in picamera2 is actually BGR byte order -> directly usable by OpenCV
    cam.configure(cam.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}))
    cam.start()
    time.sleep(1)  # let auto-exposure / white balance settle

    if not args.headless:
        win = "ball detect"
        cv2.namedWindow(win)

        def on_click(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and state["hsv"] is not None:
                state["ranges"] = ranges_from_pixel(state["hsv"], x, y)
                print("New HSV ranges:", state["ranges"])

        cv2.setMouseCallback(win, on_click)

    show_mask = False
    t_prev = time.time()
    try:
        while True:
            frame = cam.capture_array()
            blurred = cv2.GaussianBlur(frame, (7, 7), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            state["hsv"] = hsv

            mask = build_mask(hsv, state["ranges"])
            ball = find_ball(mask)

            now = time.time()
            fps = 1.0 / (now - t_prev)
            t_prev = now

            if ball:
                x, y, r, circ = ball
                msg = f"ball at ({x}, {y}) r={r}px circ={circ:.2f} fps={fps:.1f}"
            else:
                msg = f"no ball  fps={fps:.1f}"

            if args.headless:
                print(msg)
                continue

            view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask else frame.copy()
            if ball:
                cv2.circle(view, (x, y), r, (0, 255, 0), 2)
                cv2.circle(view, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                show_mask = not show_mask
            if key == ord("s"):
                fn = f"snapshot_{int(now)}.jpg"
                cv2.imwrite(fn, view)
                print("saved", fn)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
