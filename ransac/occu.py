# ground plane mask to occupancy grid

from ransac import *
import ransac.plane
import numpy as np
import math


def ground_cloud(coords, ransac_coeffs):
    # coords is a Nx2 numpy array containing coordinates (x, y)
    # pass pixel coefficients

    c1, c2, c3 = ransac_coeffs

    z = 1 / (c1 * coords[:, 0] + c2 * coords[:, 1] + c3)
    z = z.reshape(-1, 1)
    return np.concatenate((coords.astype(np.float64), z), axis=1)


def px_to_real(
        pixel_cloud, real_coeffs, intr: Intrinsics, orientation: float = 0.0):
    # outputs (x,y,z) with real z as depth, y as height
    # y values are relative to the camera's height
    # orientation (radians) is positive to orient the camera left

    # converts px into mm
    cloud = pixel_cloud.copy()
    cloud[:, 0] = pixel_cloud[:, 2] * (pixel_cloud[:, 0] - intr.cx) / intr.fx
    cloud[:, 1] = pixel_cloud[:, 2] * (intr.cy - pixel_cloud[:, 1]) / intr.fy

    depression = ransac.plane.real_angle(real_coeffs)
    c_1 = math.cos(depression)
    s_1 = math.sin(depression)
    # each column affects the output (x, y, z) respectively
    rotation_matrix = np.array([[1.0, 0.0,  0.0],
                                [0.0, c_1, -s_1],
                                [0.0, s_1,  c_1]]).transpose()

    c_2 = math.cos(orientation)
    s_2 = math.sin(orientation)
    rotation_matrix = rotation_matrix @ np.array([[c_2, 0.0, -s_2],
                                                  [0.0, 1.0,  0.0],
                                                  [s_2, 0.0,  c_2]]).transpose()

    return cloud @ rotation_matrix


def oneshot(mask_in, real_coeffs, intr: Intrinsics, conf: GridConfiguration,
            pos: CameraPosition, thres=200):
    # the numpy fuckery in this just helps interpolation
    # grid should be symmetric
    # first and second indices are number of layers to compute
    grid_shape = (3, 3, 2 * int((0.5 * conf.gh) // conf.cw),
                  2 * int((0.5 * conf.gw) // conf.cw))
    true_width = conf.cw * grid_shape[3]
    true_height = conf.cw * grid_shape[2]

    # go there, is the difference in depth of the prediction matching the depth at that actual place? is this process the same as the masking process? yes

    lys = np.arange(grid_shape[0])[:, None, None, None]
    lxs = np.arange(grid_shape[1])[None, :, None, None]
    gys = np.arange(grid_shape[2])[None, None, :, None]
    gxs = np.arange(grid_shape[3])[None, None, None, :]

    # apply camera rotation around the correct point
    rgxs = gxs - grid_shape[3] / 2 + 0.5 - (pos.x / conf.cw)
    rgys = grid_shape[2] - gys - 0.5 - (pos.y / conf.cw)
    rgxs_tmp = rgxs * math.cos(pos.h) + rgys * math.sin(pos.h)
    rgys_tmp = -rgxs * math.sin(pos.h) + rgys * math.cos(pos.h)
    # intr.tx term compensates for depths being centered on left camera lens
    # shift "after" position because rg{x,y}s used to poll from the mask
    rgxs = rgxs_tmp + grid_shape[3] / 2 - 0.5 + (intr.tx / conf.cw / 2)
    rgys = grid_shape[2] - rgys_tmp - 0.5

    # occupancy grid locations into mm
    cxs = conf.cw * ((lxs + 0.5) / grid_shape[0] + rgxs) - 0.5 * true_width
    cys = true_height - conf.cw * ((lys + 0.5) / grid_shape[1] + rgys)

    # project onto the camera plane
    a, b, d = real_coeffs
    theta = ransac.plane.real_angle(real_coeffs)
    cam_height = math.sin(theta) * d
    cys = cys * math.sin(theta)
    cys = cys - math.cos(theta) * cam_height

    # use mask to highlight driveable regions
    # this equation enforces that all (cxs, cys, zs) are on a 2-d surface, creating a bijection between the mask and the ground plane, eliminating false positives (where the ground plane location is under an obstacle but it maps to a pixel that is not occupied)
    zs = np.clip(a * cxs + b * cys + d, 1.0, None)
    pxs = np.round((cxs * intr.fx) / zs + intr.cx)
    pys = np.round(intr.cy - (cys * intr.fy) / zs)

    # ignore mask's outer edge
    mask = 255 * mask_in.astype(np.float16)
    mask[[0, -1], :] = np.nan
    mask[:, [0, -1]] = np.nan

    pxs = np.clip(pxs, 0, mask.shape[1] - 1).astype(np.int16)
    pys = np.clip(pys, 0, mask.shape[0] - 1).astype(np.int16)

    grid = np.zeros(grid_shape, dtype=np.float16)
    grid[lys, lxs, gys, gxs] = mask[pys, pxs]
    grid = np.mean(grid, axis=(0, 1))
    grid = np.where(grid > thres, 255, grid)
    grid = np.where(grid < thres, 0, grid)
    grid = np.nan_to_num(grid, nan=127)

    return grid.astype(np.uint8)
