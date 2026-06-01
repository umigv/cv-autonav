#!/usr/bin/env python3

import time
import numpy as np
from typing import cast
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


class GridMerger(Node):
    def __init__(self):
        super().__init__('occ_grid_merger')

        self.grid1 = None
        self.grid2 = None
        self.last1 = 0
        self.last2 = 0

        self.create_subscription(
            OccupancyGrid, 'occupancy_grid/raw_left', self.cb1, 10)
        self.create_subscription(
            OccupancyGrid, 'occupancy_grid/raw_right', self.cb2, 10)
        self.pub = self.create_publisher(
            OccupancyGrid, 'occupancy_grid/raw', 10)

    def cb1(self, msg):
        self.last1 = time.perf_counter()
        self.grid1 = msg
        if self.grid2 is None:
            self.grid2 = msg
        self.try_publish()

    def cb2(self, msg):
        self.last2 = time.perf_counter()
        self.grid2 = msg
        if self.grid1 is None:
            self.grid1 = msg
        self.try_publish()

    def try_publish(self):
        if self.grid1 is None or self.grid2 is None:
            return

        # 1. Extract dimensions (assuming grid1 and grid2 are identical in size)
        w = self.grid1.info.width
        h = self.grid1.info.height

        # 2. Convert to numpy arrays, BUT keep them 1D for the initial logical merge
        a = np.array(self.grid1.data, dtype=np.int8)
        b = np.array(self.grid2.data, dtype=np.int8)
        
        # Handle staleness
        if self.last1 > self.last2 + 0.5:
            b = a
        elif self.last2 > self.last1 + 0.5:
            a = b

        # 3. Basic logical merge (still 1D)
        merged_1d = np.full_like(a, -1)
        occ = (a == 100) | (b == 100)
        free = ((a == 0) | (b == 0)) & ~occ
        merged_1d[free] = 0
        merged_1d[occ] = 100
        
        # 4. Reshape to 2D for OpenCV
        merged_2d = merged_1d.reshape((h, w))

        # 5. Isolate obstacles into a proper uint8 binary image for OpenCV
        # 255 for obstacle, 0 for everything else
        occ_mask = np.where(merged_2d == 100, 255, 0).astype(np.uint8)

        # 6. Apply Morphology (Erode & Dilate)
        # It's good practice to explicitly define your kernel in OpenCV
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(occ_mask, kernel, iterations=2)
        dilated = cv2.dilate(eroded, kernel, iterations=4)
        
        # 7. Find and filter contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500
        
        # Create a clean mask to draw the surviving contours onto
        clean_occ_mask = np.zeros_like(dilated)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                # 255 is the scalar value (white) since this is a 1-channel image
                cv2.drawContours(clean_occ_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # 8. Reconstruct the final ROS-compliant OccupancyGrid
        # Start with everything unknown (-1)
        final_grid = np.full((h, w), -1, dtype=np.int8)
        
        # Apply the known free space from our original merge
        final_grid[merged_2d == 0] = 0
        
        # Overwrite with our newly cleaned obstacles
        final_grid[clean_occ_mask == 255] = 100

        # 9. Flatten back to 1D and publish
        out = OccupancyGrid()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.grid1.header.frame_id
        out.info = self.grid1.info
        out.data = final_grid.flatten().tolist()

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GridMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
