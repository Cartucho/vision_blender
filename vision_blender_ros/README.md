## vision_blender_ros

Note: I believe this is currently only working for stereo data.

Note: Stereo data must use the standard naming `(img_name)_L` and `(img_name)_R`, this is the default way of Blender to output the stereo data so you don't need to change nothing.

### Set-up

Install `cv_bridge` on your computer:

  `sudo apt-get install ros-kinetic-cv-bridge`

Link folder to catkin_ws:

  `ln -s ~/path/to/vision_blender/vision_blender_ros ~/catkin_ws/src/`

install `catkin_pkg` in python:
* `conda activate`
* `conda install -c conda-forge catkin_pkg`
* `which python`
    save that path and then build the ROS package, e.g.: `catkin build vision_blender_ros -DPYTHON_EXECUTABLE=/home/tribta/dev/miniconda3/bin/python`

### How to use it

Inside this folder (`vision_blender_ros`) create a folder called `data/` and paste there the generated rendered data.
You have to copy to `data/` all the rendered images (we recommend you to render `.jpg` images for memory reasons), `.npz` files and the `camera_info.json` file.

Then create a `rosbag` file using that data. To do that run the following command:
`roslaunch vision_blender_ros vision_blender_ros.launch`

This will create a `rosbag` file called `vision_blender_gt.bag` that you can use to playback the synthetic data into ROS.

### Config

In the file `vision_blender_ros/config/settings.yaml` you can change the name of the output rosbag file, the framerate, and rostopic names.
