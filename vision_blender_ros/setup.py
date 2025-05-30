## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=["vision_blender_ros"],
    package_dir={"": "scripts"},
    requires=["std_msgs", "rospy", "cv_bridge"],
)

setup(**setup_args)
