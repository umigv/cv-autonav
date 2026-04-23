# cv-autonav

## Repo Setup

This code depends on the `ransac` module inside `cv-depth-segmentation`. Run the following to set up the submodule

```bash
git submodule init # set up git submodules, do once
git submodule update --remote # do this when cv-depth-segmentation is updated
```

## HSV Setup
Run `./hsv_setup.sh` so tune the hsv_params. Adjust the key in hsv_tune.py to tune for different cameras.

## Running the Code
Run `one_cam.sh` for one central camera or `two_cam.sh` for two cameras. It's that simple!