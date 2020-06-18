# bl_info # read more: https://wiki.blender.org/wiki/Process/Addons/Guidelines/metainfo
bl_info = {
        "name":"Ground Truth Generation",
        "description":"Generate ground truth data (e.g., depth map) for Computer Vision applications.",
        "author":"Joao Cartucho",
        "version":(1, 0),
        "blender":(2, 82, 7),
        "location":"PROPERTIES",
        "warning":"", # used for warning icon and text in addons panel
        "wiki_url":"https://github.com/Cartucho/vision_blender",
        "support":"COMMUNITY",
        "category":"Render"
    }

import json
import os
import numpy as np # TODO: check if Blender has numpy by default

import bpy
from bpy.props import (StringProperty, # TODO: not being used
                   BoolProperty,
                   IntProperty, # TODO: not being used
                   FloatProperty, # TODO: not being used
                   EnumProperty, # TODO: not being used
                   PointerProperty
                   )
from bpy.types import (Panel,
                   Operator,
                   PropertyGroup
                   )
from bpy.app.handlers import persistent


def get_scene_resolution(scene):
    resolution_scale = (scene.render.resolution_percentage / 100.0)
    resolution_x = scene.render.resolution_x * resolution_scale # [pixels]
    resolution_y = scene.render.resolution_y * resolution_scale # [pixels]
    return int(resolution_x), int(resolution_y)


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def get_camera_parameters_intrinsic(scene):
    """ Get intrinsic camera parameters: focal length and principal point. """
    # ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera/120063#120063
    focal_length = scene.camera.data.lens # [mm]
    res_x, res_y = get_scene_resolution(scene)
    cam_data = scene.camera.data
    sensor_size_in_mm = get_sensor_size(cam_data.sensor_fit, cam_data.sensor_width, cam_data.sensor_height)
    sensor_fit = get_sensor_fit(
        cam_data.sensor_fit,
        scene.render.pixel_aspect_x * res_x,
        scene.render.pixel_aspect_y * res_y
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = res_x
    else:
        view_fac_in_px = pixel_aspect_ratio * res_y
    pixel_size_mm_per_px = (sensor_size_in_mm / focal_length) / view_fac_in_px
    f_x = 1.0 / pixel_size_mm_per_px
    f_y = (1.0 / pixel_size_mm_per_px) / pixel_aspect_ratio
    c_x = (res_x - 1) / 2.0 - cam_data.shift_x * view_fac_in_px
    c_y = (res_y - 1) / 2.0 + (cam_data.shift_y * view_fac_in_px) / pixel_aspect_ratio
    return f_x, f_y, c_x, c_y


def get_camera_parameters_extrinsic(scene):
    """ Get extrinsic camera parameters. 
    
      There are 3 coordinate systems involved:
         1. The World coordinates: "world"
            - right-handed
         2. The Blender camera coordinates: "bcam"
            - x is horizontal
            - y is up
            - right-handed: negative z look-at direction
         3. The desired computer vision camera coordinates: "cv"
            - x is horizontal
            - y is down (to align to the actual pixel coordinates 
               used in digital images)
            - right-handed: positive z look-at direction

      ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    """
    # bcam stands for blender camera
    bcam = scene.camera
    R_bcam2cv = np.array([[1,  0,  0],
                          [0, -1,  0],
                          [0,  0, -1]])

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location = np.array([bcam.matrix_world.decompose()[0]]).T
    R_world2bcam = np.array(bcam.matrix_world.decompose()[1].to_matrix().transposed())

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*bcam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = np.matmul(R_world2bcam.dot(-1), location)

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.matmul(R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)

    extr = np.concatenate((R_world2cv, T_world2cv), axis=1)
    return extr


def correct_cycles_depth(z_map, res_x, res_y, f_x, f_y, c_x, c_y, INVALID_POINT):
    for y in range(res_y):
        b = ((c_y - y) / f_y)
        for x in range(res_x):
            val = z_map[y][x]
            if val != INVALID_POINT:
                a = ((c_x - x) / f_x)
                z_map[y][x] = val / np.linalg.norm([1, a, b])
    return z_map


# classes
class MyAddonProperties(PropertyGroup):
    # booleans
    bool_save_gt_data : BoolProperty(
        name = "Ground truth",
        description = "Save ground truth data",
        default = True,
        )
    bool_save_depth : BoolProperty(
        name="Depth",
        description="Save depth data",
        default = False
        )


class GroundTruthGeneratorPanel(Panel):
    """Creates a Panel in the Output properties window for exporting ground truth data"""
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "output"


class RENDER_PT_gt_generator(GroundTruthGeneratorPanel):
    """Parent panel"""
    global intrinsic_mat
    bl_label = "Ground Truth Generator"
    bl_idname = "RENDER_PT_gt_generator"
    COMPAT_ENGINES = {'BLENDER_EEVEE', 'BLENDER_CYCLES'}#, 'BLENDER_WORKBENCH' # TODO: see what happens when using the WORKBENCH render
    bl_options = {'DEFAULT_CLOSED'}


    def draw_header(self, context):
        rd = context.scene.render
        self.layout.prop(context.scene.my_addon, "bool_save_gt_data", text="")


    def draw(self, context):
        scene = context.scene
        my_addon = scene.my_addon
        layout = self.layout
        layout.active = my_addon.bool_save_gt_data

        layout.use_property_split = False
        layout.use_property_decorate = False  # No animation.

        """ testing a bool """
        layout.prop(my_addon, "bool_save_depth", text="Bool Property")
        # check if bool property is enabled
        if (my_addon.bool_save_depth == True):
            print ("Save depth Enabled")
        else:
            print ("Save depth Disabled")

        # Get camera parameters
        """ show intrinsic parameters """
        layout.label(text="Intrinsic parameters [pixels]:")
        f_x, f_y, c_x, c_y = get_camera_parameters_intrinsic(scene)

        box_intr = self.layout.box()
        col_intr = box_intr.column()

        row_intr_0 = col_intr.split()
        row_intr_0.label(text=str(f_x))# "{}".format(round(f_x, 3))
        row_intr_0.label(text='0')
        row_intr_0.label(text=str(c_x))

        row_intr_1 = col_intr.split()
        row_intr_1.label(text='0')
        row_intr_1.label(text=str(f_y))
        row_intr_1.label(text=str(c_y))

        row_intr_2 = col_intr.split()
        row_intr_2.label(text='0')
        row_intr_2.label(text='0')
        row_intr_2.label(text='1')

        """ show extrinsic parameters """
        layout.label(text="Extrinsic parameters [pixels]:")

        extr = get_camera_parameters_extrinsic(scene)

        box_ext = self.layout.box()
        col_ext = box_ext.column()

        row_ext_0 = col_ext.split()
        row_ext_0.label(text=str(extr[0, 0]))
        row_ext_0.label(text=str(extr[0, 1]))
        row_ext_0.label(text=str(extr[0, 2]))
        row_ext_0.label(text=str(extr[0, 3]))

        row_ext_1 = col_ext.split()
        row_ext_1.label(text=str(extr[1, 0]))
        row_ext_1.label(text=str(extr[1, 1]))
        row_ext_1.label(text=str(extr[1, 2]))
        row_ext_1.label(text=str(extr[1, 3]))

        row_ext_2 = col_ext.split()
        row_ext_2.label(text=str(extr[2, 0]))
        row_ext_2.label(text=str(extr[2, 1]))
        row_ext_2.label(text=str(extr[2, 2]))
        row_ext_2.label(text=str(extr[2, 3]))

classes = (
    RENDER_PT_gt_generator,
    MyAddonProperties,
)


def get_or_create_node(tree, node_type, node_name):
    node_ind = tree.nodes.find(node_name)
    if node_ind == -1:
        v = tree.nodes.new(node_type)
        v.name = node_name
    else:
        v = tree.nodes[node_ind]
    return v


@persistent # TODO: not sure if I should be using @persistent
def load_handler_render_init(scene):
    #print("Initializing a render job...")
    # 1. Set-up Passes
    if not scene.use_nodes:
        scene.use_nodes = True
    if not scene.view_layers["View Layer"].use_pass_z:
        scene.view_layers["View Layer"].use_pass_z = True
    if not scene.view_layers["View Layer"].use_pass_normal:
        scene.view_layers["View Layer"].use_pass_normal = True
    if scene.render.engine == 'CYCLES':
        if not scene.view_layers["View Layer"].use_pass_object_index:
            scene.view_layers["View Layer"].use_pass_object_index = True
        if not scene.view_layers["View Layer"].use_pass_vector:
            scene.view_layers["View Layer"].use_pass_vector = True

    # 2. Set-up nodes
    tree = scene.node_tree
    rl = scene.node_tree.nodes["Render Layers"] # I assumed there is always a Render Layers
    node_norm_and_z = get_or_create_node(tree, "CompositorNodeViewer", "normal_and_zmap")

    VIEWER_FIXED = False # TODO: change code when https://developer.blender.org/T54314 is fixed
    if scene.render.engine == "CYCLES":
        if VIEWER_FIXED:
            node_obj_ind = get_or_create_node(tree, "CompositorNodeViewer", "obj_ind")
            node_opt_flow = get_or_create_node(tree, "CompositorNodeViewer", "opt_flow")
        else:
            ## create two output nodes
            node_obj_ind = get_or_create_node(tree, "CompositorNodeOutputFile", "obj_ind")
            node_opt_flow = get_or_create_node(tree, "CompositorNodeOutputFile", "opt_flow")
            ### set-up their output paths
            path_render = scene.render.filepath
            node_obj_ind.base_path = os.path.join(path_render, "obj_ind")
            node_opt_flow.base_path = os.path.join(path_render, "opt_flow")

    # 3. Set-up links between nodes
    ## create new links if necessary
    links = tree.links
    ## Trick: we already have the RGB image so we can connect the Normal to Image
    ##        and the Z to the Alpha channel
    if not node_norm_and_z.inputs["Image"].is_linked:
        links.new(rl.outputs["Normal"], node_norm_and_z.inputs["Image"])
    if not node_norm_and_z.inputs["Alpha"].is_linked:
        links.new(rl.outputs["Depth"], node_norm_and_z.inputs["Alpha"])
    if scene.render.engine == "CYCLES":
        if VIEWER_FIXED:
            if not node_obj_ind.inputs["Image"].is_linked:
                links.new(rl.outputs["IndexOB"], node_obj_ind.inputs["Image"])
            ## The optical flow needs to be connected to both `Image` and `Alpha`
            if not node_opt_flow.inputs["Image"].is_linked:
                links.new(rl.outputs["Vector"], node_opt_flow.inputs["Image"])
                links.new(rl.outputs["Vector"], node_opt_flow.inputs["Alpha"])
        else:
            if not node_obj_ind.inputs["Image"].is_linked:
                links.new(rl.outputs["IndexOB"], node_obj_ind.inputs["Image"])
            if not node_opt_flow.inputs["Image"].is_linked:
                links.new(rl.outputs["Vector"], node_opt_flow.inputs["Image"])

    # 4. Save camera_info
    dict_cam_info = {}
    render = scene.render
    cam = scene.camera
    ## img file format
    dict_cam_info['img_format'] = render.image_settings.file_format
    ## camera image resolution
    res_x, res_y = get_scene_resolution(scene)
    dict_cam_info['img_res_x'] = res_x
    dict_cam_info['img_res_y'] = res_y
    ## camera intrinsic matrix parameters
    cam_mat_intr = {}
    f_x, f_y, c_x, c_y = get_camera_parameters_intrinsic(scene)
    cam_mat_intr['f_x'] = f_x
    cam_mat_intr['f_y'] = f_y
    cam_mat_intr['c_x'] = c_x
    cam_mat_intr['c_y'] = c_y
    dict_cam_info['cam_mat_intr'] = cam_mat_intr
    ## is_stereo
    is_stereo = render.use_multiview
    dict_cam_info['is_stereo'] = is_stereo
    if is_stereo:
        stereo_info = {}
        ### left camera file suffix
        stereo_info['stereo_left_suffix'] = render.views["left"].file_suffix
        ### right camera file suffix
        stereo_info['stereo_right_suffix'] = render.views["right"].file_suffix
        ### stereo mode
        stereo_info['stereo_mode'] = cam.data.stereo.convergence_mode
        ### stereo interocular distance
        stereo_info['stereo_interocular_distance'] = cam.data.stereo.interocular_distance
        ### stereo pivot
        stereo_info['stereo_pivot'] = cam.data.stereo.pivot
        dict_cam_info['stereo_info'] = stereo_info
    ## save data to a json file
    gt_dir_path = scene.render.filepath
    out_path = os.path.join(gt_dir_path, 'camera_info.json')
    with open(out_path, 'w') as tmp_file:
        json.dump(dict_cam_info, tmp_file)


@persistent # TODO: not sure if I should be using @persistent
def load_handler_after_rend_frame(scene): # TODO: not sure if this is the best place to put this function, should it be above the classes?
    """ This script runs after rendering each frame """
    # ref: https://blenderartists.org/t/how-to-run-script-on-every-frame-in-blender-render/699404/2
    # check if user wants to generate the ground truth data
    if scene.my_addon.bool_save_gt_data:
        gt_dir_path = scene.render.filepath
        #print(gt_dir_path)
        # save ground truth data
        #print(scene.frame_current)
        """ Camera parameters """
        ## update camera - ref: https://blender.stackexchange.com/questions/5636/how-can-i-get-the-location-of-an-object-at-each-keyframe
        scene.frame_set(scene.frame_current) # needed to update the camera position
        ### extrinsic
        extrinsic_mat = get_camera_parameters_extrinsic(scene)
        ### intrinsic
        f_x, f_y, c_x, c_y = get_camera_parameters_intrinsic(scene)
        """ Zmap + Normal """
        ## get data
        pixels = bpy.data.images['Viewer Node'].pixels
        #print(len(pixels)) # size = width * height * 4 (rgba)
        pixels_numpy = np.array(pixels[:])
        res_x, res_y = get_scene_resolution(scene)
        #   .---> y
        #   |
        #   |
        #   v
        #    x
        pixels_numpy.resize((res_y, res_x, 4)) # Numpy works with (y, x, channels)
        normal = pixels_numpy[:, :, 0:3]
        z = pixels_numpy[:, :, 3]
        # points at infinity get a -1 value
        max_dist = scene.camera.data.clip_end
        INVALID_POINT = -1.0
        normal[z > max_dist] = INVALID_POINT
        z[z > max_dist] = INVALID_POINT
        if scene.render.engine == "CYCLES":
            z = correct_cycles_depth(z, res_x, res_y, f_x, f_y, c_x, c_y, INVALID_POINT)
        """ Obj Index + Opt flow"""
        VIEWER_FIXED = False # TODO: change code when https://developer.blender.org/T54314 is fixed
        if VIEWER_FIXED:
            # TODO: make each Image Viewer active one-by-one and copy values
            pass
        else:
            # TODO
            #obj_ind_file_path = os.path.join(gt_dir_path, "obj_ind", "Image{}.png".format(5))
            pass
        """ Save data """
        # Blender by default assumes a padding of 4 digits
        out_path = os.path.join(gt_dir_path, '{:04d}.npz'.format(scene.frame_current))
        #print(out_path)
        np.savez_compressed(out_path,
                            extr=extrinsic_mat,
                            normal_map=normal,
                            z_map=z
                           )
        # ref: https://stackoverflow.com/questions/35133317/numpy-save-some-arrays-at-once


# registration
def register():
    # register the classes
    for cls in classes:
        bpy.utils.register_class(cls)
    # register the properties
    bpy.types.Scene.my_addon = PointerProperty(type=MyAddonProperties)
    # register the function being called when rendering starts
    bpy.app.handlers.render_init.append(load_handler_render_init)
    # register the function being called after rendering each frame
    bpy.app.handlers.render_post.append(load_handler_after_rend_frame)


def unregister():
    # unregister the classes
    for cls in classes:
        bpy.utils.unregister_class(cls)
    # unregister the properties
    del bpy.types.Scene.my_addon
    # unregister the function being called when rendering each frame
    bpy.app.handlers.render_init.remove(load_handler_render_init)
    # unregister the function being called when rendering each frame
    bpy.app.handlers.render_post.remove(load_handler_after_rend_frame)

if __name__ == "__main__":
    register()

# reference: https://github.com/sobotka/blender/blob/662d94e020f36e75b9c6b4a258f31c1625573ee8/release/scripts/startup/bl_ui/properties_output.py
