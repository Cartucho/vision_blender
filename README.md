# vision_blender

[![GitHub stars](https://img.shields.io/github/stars/Cartucho/vision_blender.svg?style=social&label=Stars)](https://github.com/Cartucho/vision_blender)

A Blender user-interface to generate synthetic ground truth data (benchmarks) for Computer Vision applications.

<img src="https://user-images.githubusercontent.com/15831541/94527156-7b944d80-022e-11eb-85bd-0b387fd519fb.png" width="100%">

<img src="https://user-images.githubusercontent.com/15831541/94527180-8353f200-022e-11eb-9bf5-5ebd6102bc9f.png" width="100%">

VisionBlender is a synthetic computer vision dataset generator that adds a user interface to Blender, allowing users to generate monocular/stereo video sequences with ground truth maps of depth, disparity, segmentation masks, surface normals, optical flow, object pose, and camera parameters.

[![Presentation video](https://user-images.githubusercontent.com/15831541/95021661-388d0c80-066a-11eb-9216-a5deac6372df.png)](https://youtu.be/lMiBVAT3hkI "Paper presentation video - Click to Watch!")
[YouTube link](https://youtu.be/lMiBVAT3hkI)

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

## Paper

This work received the best paper award at a MICCAI 2020 workshop!

Note: The paper was not released in the journal yet but if you email me I can send it to you.

If you use this tool please consider citing our paper:

```bibtex
@article{joao2020visionblender,
  Author = {Joao Cartucho, Samyakh Tukra, Yunpeng Li, Daniel S. Elson, Stamatia Giannarou},
  Journal = {Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization},
  Title = {VisionBlender: A Tool to Efficiently Generate Computer Vision Datasets for Robotic Surgery},
  Year = {2020}
}
```
