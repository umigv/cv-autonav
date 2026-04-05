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

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PointStamped, Pose, Quaternion, Point
import pyzed.sl as sl

import cv2
import math
from multiprocessing import Pool, Process
import numpy as np
from time import perf_counter
from signal import signal, SIGINT

import cv_depth_segmentation.src.ransac as rsc

# TODO! integrate all cv repos together

# >>> ros2 change
# <<< ros2 end of change


class OccGridPublisher(Node):
    def __init__(self, side: str, conf: rsc.GridConfiguration):
        super().__init__(f"occ_grid_publisher_{side}")
        self.pub = self.create_publisher(
            OccupancyGrid, f"occupancy_grid/{side}", 10)
        self.width = int(conf.gw // conf.cw)
        self.height = int(conf.gh // conf.cw)
        self.resolution = conf.cw / 1000.0

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

        # TODO! mark waypoints as 127

        ros = np.rot90(ros, k=3)
        ros = np.flipud(ros)

        msg.data = ros.flatten().tolist()

        self.pub.publish(msg)


def run_ransac_on_zed(side: str, cam_pos=rsc.CameraPosition(), serial_number=None):
    rclpy.init()

    # setup zed camera parameters
    init = sl.InitParameters()
    if serial_number is not None:
        init.set_from_serial_number(serial_number)

    init.async_image_retrieval = True
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.camera_resolution = sl.RESOLUTION.HD720  # try .VGA
    init.camera_fps = 30  # lower framerate to avoid issues
    # setup ransac pipeline
    live = rsc.LiveSource(init, (720, 404))
    conf = rsc.GridConfiguration(5000.0, 5000.0, 50.0)
    depseg = rsc.DepthSegementation([(live, cam_pos)], conf)

    # configure ROS publisher

    occ_node = OccGridPublisher(side, conf)

    key = 0
    start = perf_counter()
    while key != 113:  # for 'q' key
        key = cv2.waitKey(1)

        if not live.update():
            break

        updated = depseg.process()
        if not updated:
            continue

        occ = depseg.merge_simple()
        occ_img = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)
        occ_img = cv2.resize(
            occ_img, (600, 600), interpolation=cv2.INTER_NEAREST_EXACT
        )
        cv2.imshow(f"[{side}] occupancy grid", occ_img)

        now = perf_counter()
        # TODO: implement DepthSource.serial_number()
        # print(f"{live.serial_number()}: {1 / (now - start):.2f} FPS")
        start = now

        occ_node.publish(occ)
        rclpy.spin_once(occ_node, timeout_sec=0.0)

    cv2.destroyAllWindows()
    rclpy.shutdown()


def spin_up_node(side: str, pos: rsc.CameraPosition, ser: int | None):
    return Process(target=run_ransac_on_zed, args=(side, pos, ser))


if __name__ == "__main__":
    left_pos = rsc.CameraPosition(-110, 40, 0.39670597283903605)
    right_pos = rsc.CameraPosition(130, 40, -0.5410520681182421)

    left_proc = spin_up_node("left", left_pos, 39394535)
    right_proc = spin_up_node("right", right_pos, 36466710)

    left_proc.start()
    right_proc.start()
    left_proc.join()
    right_proc.join()
