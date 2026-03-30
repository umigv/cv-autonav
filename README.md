# cv-autonav

## setup

This code depends on the `ransac` module inside `cv-depth-segmentation`. Run the following to set up a symlink. Alternatively, copy `ransac` into the working directory (keeping in mind that it needs to be updated when the code is pulled)

```bash
git submodule init # set up git submodules, do once
git submodule update --remote # do this when cv-depth-segmentation is updated
# set up the symlink
ln -s cv-depth-segmentation/src/ransac ransac
```
