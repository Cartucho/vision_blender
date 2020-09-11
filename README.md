THIS CODE IS UNDER CONSTRUCTION, THE FIRST VERSION WILL BE RELEASED ASAP.

# vision_blender

A Blender addon to generate synthetic ground truth data (benchmarks) for Computer Vision applications.



To install the addon simply go to File > User Preferences > Add-on tab > Install from File...

then select the file `addon_ground_truth_generation.py`

Again in the add-on tab make sure that you enable the add-on once it appears



Rendering

            render individual images and not videos. Render images has a couple of advantages. For example, when rendering animations to image file formats the render job can be canceled and resumed at the last rendered frame by changing the frame range. This is useful if the animation takes a long time to render and the computers resources are needed for something else.

            you can later use Blender to generate a video from the individual frames.
            (Images can then be encoded to a video by adding the rendered image sequence into the Video Sequencer and choosing an appropriate Video Output.)

    Output
        Set-up path and desired image format.

    File format:
        If it is a `test animation` you should write to JPEG, a lossy format (which means you will loose some quality).
        If it is a `final animation` PNG is the way to go, it is a lossless format.

    Render > Render Animation... or press `Ctrl + F12`

    You can stop and then restart! Just select again the starting frame! Uooooouu!!



Blender getting the mask of the objects:
    Object Properties > Relations > Pass Index



## ROS

Stereo data must use the standard naming img_L and img_R

Install `cv_bridge` on your computer:

  sudo apt-get install ros-kinetic-cv-bridge

Link folder to catkin_ws:

ln -s ~/dev/vision_blender/vision_blender_ros ~/catkin_ws/src/


install catkin_pkg in python:
#. conda activate
#. conda install -c conda-forge catkin_pkg
#. which python

    save that path and then build the ROS package

e.g.: `catkin build vision_blender_ros -DPYTHON_EXECUTABLE=/home/tribta/dev/miniconda3/bin/python`

launch is for publishing data
rosrun cpp example is an example for how people can read it in C++

since we want the user to be able to control the frame rate then we create a rosba with the data
