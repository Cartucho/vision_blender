# vision_blender

[![GitHub stars](https://img.shields.io/github/stars/Cartucho/vision_blender.svg?style=social&label=Stars)](https://github.com/Cartucho/vision_blender)

A Blender user-interface to generate synthetic ground truth data (benchmarks) for Computer Vision applications.

<img src="https://user-images.githubusercontent.com/15831541/94527156-7b944d80-022e-11eb-85bd-0b387fd519fb.png" width="40%">

<img src="https://user-images.githubusercontent.com/15831541/94527180-8353f200-022e-11eb-9bf5-5ebd6102bc9f.png" width="30%">

## Installation

To install the addon simply go to `Edit > Preferences > Add-on tab > Install an add-on`
, then select the file `path/to/vision_blender/addon_ground_truth_generation.py` and click `Install Add-on`.

Finally you have to enable the add-on; Search `VisionBlender` and tick the check-box.
You should now be able to find the `VisionBlender UI` in th bottom of the `Output Properties`.


## How to generate ground truth data?

Simply tick the boxes of what you want to save as ground truth in the `VisionBlender UI`.
Then start rendering and the outputs will be generated automatically.
To render you click `Render > Render Image` or `Render > Render Animation...`, alternatively you can click `F12` for image and `Ctrl F12` for animation.

You can change the output path in `Output Properties > Output > Output Path`.

Note: `Segmentation masks` and `Optical flow` are only available in Cycles.

#### Segmentation masks ####

To set-up the segmentation masks you need to choose a pass index for each object:
    `Object Properties > Relations > Pass Index`

#### Optical flow ####

You will only have optical flow if the camera or the objects are moving during an animation.

### How to read the data after generating it?

You simply have to load the numpy arrays from thr `.npz` files.
Go to the `vision_blender/samples` and have a look at the example there!
