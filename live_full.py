########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import sys
import pyzed.sl as sl
from signal import signal, SIGINT
import argparse
import os
import cv2
import numpy as np
import math

from multiprocessing import Pool

from ransac import *
import ransac.plane
import ransac.occu

cam = sl.Camera()


# Handler to deal with CTRL+C properly
def handler(signal_received, frame):
    cam.disable_recording()
    cam.close()
    sys.exit(0)


signal(SIGINT, handler)


def print_params(calibration_params: sl.CalibrationParameters):
    # LEFT CAMERA intrinsics
    fx_left = calibration_params.left_cam.fx
    fy_left = calibration_params.left_cam.fy
    cx_left = calibration_params.left_cam.cx
    cy_left = calibration_params.left_cam.cy

    # RIGHT CAMERA intrinsics
    fx_right = calibration_params.right_cam.fx
    fy_right = calibration_params.right_cam.fy
    cx_right = calibration_params.right_cam.cx
    cy_right = calibration_params.right_cam.cy

    # Translation (baseline) between left and right camera
    tx = calibration_params.stereo_transform.get_translation().get()[0]

    # Print results
    print("\n--- ZED Camera Calibration Parameters ---")
    print("Left Camera Intrinsics:")
    print(f"  fx = {fx_left:.3f}")
    print(f"  fy = {fy_left:.3f}")
    print(f"  cx = {cx_left:.3f}")
    print(f"  cy = {cy_left:.3f}\n")

    print("Right Camera Intrinsics:")
    print(f"  fx = {fx_right:.3f}")
    print(f"  fy = {fy_right:.3f}")
    print(f"  cx = {cx_right:.3f}")
    print(f"  cy = {cy_right:.3f}\n")

    print(f"Stereo Baseline (tx): {tx:.6f} meters")


def intrinsics_from_params(params: sl.CalibrationParameters):
    return ransac.Intrinsics(params.left_cam.cx, params.left_cam.cy,
                             params.left_cam.fx, params.left_cam.fy,
                             params.stereo_transform.get_translation().get()[0])


def main():
    global cam

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.async_image_retrieval = True  # maybe change to False if stuff breaks

    status = cam.open(init)

    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open", status, "Exit program.")
        exit(1)

    runtime = sl.RuntimeParameters()

    resolution = cam.get_camera_information().camera_configuration.resolution
    w = min(720, resolution.width)
    h = min(404, resolution.height)
    low_res = sl.Resolution(w, h)

    cam_info = cam.get_camera_information()
    calibration_params = cam_info.camera_configuration.calibration_parameters
    print_params(calibration_params)

    intr = intrinsics_from_params(calibration_params)
    grid_conf = ransac.GridConfiguration(5000, 5000, 50)

    thread_pool_size = 4
    thread_pool = Pool(thread_pool_size)
    px_coeffs = np.array([0.0, 0.0, 0.0])

    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    key = 0
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err > sl.ERROR_CODE.SUCCESS:  # good to go
            print("Grab ZED : ", err)
            break

        # FIXME pointing camera at only the ground causing a crash
        cam.retrieve_image(image_mat, sl.VIEW.LEFT, sl.MEM.CPU, low_res)
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU, low_res)

        image = image_mat.get_data()
        depths = ransac.plane.clean_depths(depth_mat.get_data())

        # ACTUAL USE
        # ransac_output, ransac_coeffs = ransac.plane.hsv_and_ransac(image, depths, 60, (1, 16), 0.15)
        # GROUND ONLY
        ground_mask, px_coeffs = ransac.plane.ground_plane(
            depths, 100, (1, 16), 0.13, px_coeffs, thread_pool, thread_pool_size
        )

        real_coeffs = ransac.plane.real_coeffs(px_coeffs, intr)
        rad = ransac.plane.real_angle(real_coeffs)

        occ = ransac.occu.oneshot(
            ground_mask, real_coeffs, intr, grid_conf, cam_h=0, thres=200)

        occ_img = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)
        occ_img = cv2.resize(
            occ_img, (600, 600), interpolation=cv2.INTER_NEAREST_EXACT
        )
        cv2.imshow("occupancy grid", occ_img)

        x = w // 2
        y = h // 2
        coords = np.array([[x, y]])  # pixel coordinates
        pred_real = np.array([])  # n by 2 array of (x, z) coordinates
        if coords is not None:
            pred = ransac.occu.ground_cloud(coords, px_coeffs)
            pred_real = ransac.occu.px_to_real(
                pred, real_coeffs, intr)[:, (0, 2)]
        print(pred_real)

        print(f"angle: {math.degrees(rad): .3f} deg")

        key = cv2.waitKey(1)
    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    main()
