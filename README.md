# cv-autonav

## setup

This code depends on the `ransac` module inside `cv-depth-segmentation`. Run the following to set up a symlink. Alternatively, copy `ransac` into the working directory (keeping in mind that it needs to be updated when the code is pulled)

```bash
# set up the submodule
git submodule init
git submodule update
# set up the symlink
ln -s cv-depth-segmentation/src/ransac ransac
```
