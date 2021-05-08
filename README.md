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

<img src="https://media.giphy.com/media/8hi33WFpulQgNS0aKZ/giphy.gif" width="50%">

You should now be able to find the `VisionBlender UI` in the bottom of the `Output Properties`.

<img src="https://media.giphy.com/media/yohYefBMecG2zxTt6T/giphy.gif" width="50%">

## How to generate ground truth data?

### 1. Select render engine

If you want to get ground truth `Segmentation masks` or `Optical flow` you need first to set blender to use the `Cycles` Render Engine. Otherwise, use `Eevee` (it will be faster!) which is set by default.

<img src="https://media.giphy.com/media/s87Yo48JPQITTzVnbl/giphy.gif" width="50%">

##### How to set-up segmentation masks? #####

To set-up the segmentation masks you need to choose a pass index other than zero (!= 0) for each object:
    `Object Properties > Relations > Pass Index`

<img src="https://media.giphy.com/media/FLL3LQWg1x01efAc1e/giphy.gif" width="50%">

Each integer (e.g., `Pass Index = 1`) represents a class of objects to be segmented.

##### How to set-up optical flow? #####

You will only have optical flow if the camera or the objects are moving during an animation. In the following gif, I show you an example of how to move an object between frames:

<img src="https://media.giphy.com/media/N77idgOsPjkxbk5tfd/giphy.gif" width="50%">

### 2. Set output path

Set up the output path in `Output Properties > Output > Output Path`. This is the path where both your rendered images and ground truth will be saved.

<img src="https://media.giphy.com/media/pkonIVp8o8slvsC3Nf/giphy.gif" width="50%">

### 3. Select ground truth maps and render

First, tick the boxes of what you want to save as ground truth in the `VisionBlender UI`. Then, start rendering. To start rendering you click `Render > Render Image` or `Render > Render Animation...`, alternatively you can click `F12` for image and `Ctrl F12` for animation.

<img src="https://media.giphy.com/media/evpNpfJMYzwEyaHeQG/giphy.gif" width="50%">

Note: The ground-truth maps are always calculated using meters [m] as unit of distance.

### How to read the data after generating it?

You simply have to load the numpy arrays from thr `.npz` files.
Go to the `vision_blender/samples` and have a look at the example there!

## Paper

This work received the best paper award at a MICCAI 2020 workshop!

The paper can be found at [this link](https://www.tandfonline.com/doi/full/10.1080/21681163.2020.1835546)

If you use this tool please consider citing our paper:

```bibtex
@article{cartucho2020visionblender,
  title={VisionBlender: a tool to efficiently generate computer vision datasets for robotic surgery},
  author={Cartucho, Jo{\~a}o and Tukra, Samyakh and Li, Yunpeng and S. Elson, Daniel and Giannarou, Stamatia},
  journal={Computer Methods in Biomechanics and Biomedical Engineering: Imaging \& Visualization},
  pages={1--8},
  year={2020},
  publisher={Taylor \& Francis}
}
```
