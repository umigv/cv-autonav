#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


class GridMerger(Node):
    def __init__(self):
        super().__init__('occ_grid_merger')

        self.grid1 = None
        self.grid2 = None

        self.create_subscription(
            OccupancyGrid, 'occupancy_grid/left', self.cb1, 10)
        self.create_subscription(
            OccupancyGrid, 'occupancy_grid/right', self.cb2, 10)
        self.pub = self.create_publisher(
            OccupancyGrid, 'occupancy_grid/raw', 10)

    def cb1(self, msg):
        self.grid1 = msg
        self.try_publish()

    def cb2(self, msg):
        self.grid2 = msg
        self.try_publish()

    def try_publish(self):
        if self.grid1 is None or self.grid2 is None:
            return

        a = np.array(self.grid1.data, dtype=np.int8)
        b = np.array(self.grid2.data, dtype=np.int8)

        merged = np.full_like(a, -1)
        occ = (a == 100) | (b == 100)
        free = ((a == 0) | (b == 0)) & (~occ)

        merged[occ] = 100
        merged[free] = 0

        out = OccupancyGrid()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.grid1.header.frame_id
        out.info = self.grid1.info
        out.data = merged.tolist()

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GridMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
