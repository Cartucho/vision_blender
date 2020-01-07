# BlenderVision
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
