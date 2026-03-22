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

import pyzed.sl as sl
from signal import signal, SIGINT
import cv2
import numpy as np
import math

from multiprocessing import Pool, Process

from ransac import *
import ransac.plane
import ransac.occu

from time import perf_counter

# >>> ros2 change
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PointStamped, Pose, Quaternion, Point
# <<< ros2 end of change


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


def intrinsics_from_params(params: sl.CalibrationParameters, sx, sy):

    return ransac.Intrinsics(params.left_cam.cx * sx, params.left_cam.cy * sy,
                             params.left_cam.fx * sx, params.left_cam.fy * sy,
                             params.stereo_transform.get_translation().get()[0])


class OccGridPublisher(Node):
    def __init__(self, side: str, width: int, height: int, resolution: float):
        super().__init__(f"occ_grid_publisher_{side}")
        self.pub = self.create_publisher(OccupancyGrid, f"occupancy_grid/{side}", 10)
        self.width = width
        self.height = height
        self.resolution = resolution

    def publish(self, grid_np):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        info = MapMetaData()
        info.width = self.width
        info.height = self.height
        info.resolution = self.resolution

        # Origin: where camera is roughly
        origin = Pose()
        origin.position = Point(
            x=0.0,
            y=-self.width * self.resolution / 2.0,
            z=0.0
        )
        origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        info.origin = origin

        msg.info = info

        # Convert internal 0/127/255 encoding to ROS -1/0/100
        flat = grid_np.astype('uint8')
        ros = np.full(flat.shape, -1, dtype=np.int8)
        ros[flat == 0] = 100   # occupied
        ros[flat == 255] = 0   # free

        # TODO: mark waypoints as 127

        ros = np.rot90(ros, k=3)
        ros = np.flipud(ros)

        msg.data = ros.flatten().tolist()

        self.pub.publish(msg)


def run_ransac_on_zed(side: str, cam_pos=CameraPosition(), serial_number=None):
    rclpy.init()
    cam = sl.Camera()

    init = sl.InitParameters()
    if serial_number is not None:
        init.set_from_serial_number(serial_number)
    # TODO: perhaps set resolution of init parameters to VGA to match 720x404
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

    intr = intrinsics_from_params(calibration_params,
                                  w / float(resolution.width),
                                  h / float(resolution.height))
    grid_conf = ransac.GridConfiguration(5000.0, 5000.0, 50.0)

    grid_width = int(grid_conf.gw // grid_conf.cw)
    grid_height = int(grid_conf.gh // grid_conf.cw)
    cell_resolution_m = grid_conf.cw / 1000.0
    occ_node = OccGridPublisher(side, grid_width, grid_height, cell_resolution_m)

    thread_pool_size = 4
    thread_pool = Pool(thread_pool_size)
    px_coeffs = np.array([0.0, 0.0, 0.0])

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    start = perf_counter()
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

        # generate occupancy grid
        # ACTUAL USE
        # ransac_output, ransac_coeffs = ransac.plane.hsv_and_ransac(image, depths, 60, (1, 16), 0.15)
        # GROUND ONLY
        hsv_mask = ransac.plane.hsv_mask(image)
        ground_mask, px_coeffs = ransac.plane.ground_plane(
            depths, 100, (1, 16), 0.13, px_coeffs, thread_pool, thread_pool_size
        )
        # lane_mask = ground_mask & 
        real_coeffs = ransac.plane.real_coeffs(px_coeffs, intr)
        rad = ransac.plane.real_angle(real_coeffs)
        occ = ransac.occu.oneshot(
            hsv_mask, real_coeffs, intr, grid_conf, cam_pos, thres=200)

        # convert occupancy grid to image format
        occ_img = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)
        occ_img = cv2.resize(
            occ_img, (600, 600), interpolation=cv2.INTER_NEAREST_EXACT
        )
        cv2.imshow("occupancy grid", occ_img)
        print(f"angle: {math.degrees(rad): .3f} deg")

        now = perf_counter()
        print(f"{cam.get_camera_information().serial_number}: {1 / (now - start):.2f} FPS")
        start = now


        occ_node.publish(occ)
        rclpy.spin_once(occ_node, timeout_sec=0.0)

        key = cv2.waitKey(1)

    cv2.destroyAllWindows()
    cam.close()
    rclpy.shutdown()


if __name__ == "__main__":
    p1 = Process(target=run_ransac_on_zed,
                 args=("left", CameraPosition(0, 0, math.radians(0)), None))
    # p2 = Process(target=run_ransac_on_zed,
    #              args=("right", CameraPosition(0, 0, math.radians(-45)), None))

    p1.start()
    # p2.start()

    p1.join()
    # p2.join()
