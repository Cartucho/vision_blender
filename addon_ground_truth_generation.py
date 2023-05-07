# bl_info # read more: https://wiki.blender.org/wiki/Process/Addons/Guidelines/metainfo
bl_info = {
        "name":"VisionBlender - Computer Vision Ground Truth Generation",
        "description":"Generate ground truth data (e.g., depth map) for Computer Vision applications.",
        "author":"Joao Cartucho",
        "version":(1, 1),
        "blender":(2, 83, 4),
        "location":"PROPERTIES",
        "warning":"", # used for warning icon and text in addons panel
        "wiki_url":"https://github.com/Cartucho/vision_blender",
        "support":"COMMUNITY",
        "category":"Render"
    }

import json
import os
import shutil
import numpy as np

import bpy
from bpy.props import (BoolProperty,
                   PointerProperty,
                   FloatVectorProperty
                   )
from bpy.types import (Panel,
                   Operator,
                   PropertyGroup
                   )
from bpy.app.handlers import persistent

""" Defining fuctions to obtain ground truth data """
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


def get_obj_poses():
    n_chars = get_largest_object_name_length()
    n_object = len(bpy.data.objects)
    obj_poses = np.zeros(n_object, dtype=[('name', 'U{}'.format(n_chars)), ('pose', np.float64, (4, 4))])
    for ind, obj in enumerate(bpy.data.objects):
        obj_poses[ind] = (obj.name, obj.matrix_world)
    return obj_poses


def check_if_node_exists(tree, node_name):
    node_ind = tree.nodes.find(node_name)
    if node_ind == -1:
        return False
    return True


def create_node(tree, node_type, node_name):
    node_exists = check_if_node_exists(tree, node_name)
    if not node_exists:
        v = tree.nodes.new(node_type)
        v.name = node_name
    else:
        v = tree.nodes[node_name]
    return v


def remove_old_vision_blender_nodes(tree):
    for node in tree.nodes:
        if 'vision_blender' in node.name:
            tree.nodes.remove(node)


def clean_folder(folder_path):
    # ref : https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(folder_path)


def get_set_of_non_zero_obj_ind():
    non_zero_obj_ind_set = set([])
    for obj in bpy.data.objects:
        if obj.pass_index != 0:
            non_zero_obj_ind_set.add(obj.pass_index)
    return non_zero_obj_ind_set


def check_any_obj_with_non_zero_index():
    for obj in bpy.data.objects:
        if obj.pass_index != 0:
            return True
    return False


def get_largest_object_name_length():
    max_chars = 0
    for obj in bpy.data.objects:
        if len(obj.name) > max_chars:
            max_chars = len(obj.name)
    return max_chars


def get_struct_array_of_obj_indexes():
    # ref: https://numpy.org/doc/stable/user/basics.rec.html
    n_chars = get_largest_object_name_length()
    n_object = len(bpy.data.objects)
    # max_index = 32767 in Blender version 2.83, so unsigned 2-byte is more than enough memory
    obj_indexes = np.zeros(n_object, dtype=[('name', 'U{}'.format(n_chars)), ('pass_index', '<u2')])
    for ind, obj in enumerate(bpy.data.objects):
        obj_indexes[ind] = (obj.name, obj.pass_index)
    return obj_indexes


def is_stereo_ok_for_disparity(scene):
    if (scene.render.use_multiview and
        scene.camera.data.stereo.convergence_mode == 'PARALLEL'):
        return True
    return False


def get_transf0to1(scene):
    transf = None
    if scene.camera.data.stereo.convergence_mode == 'PARALLEL':
        translation_x = scene.camera.data.stereo.interocular_distance
        transf = np.zeros((4, 4))
        np.fill_diagonal(transf, 1)
        transf[0, 3] = - translation_x
    return transf


def load_file_data_to_numpy(scene, tmp_file_path, data_map):
    if not os.path.isfile(tmp_file_path):
        return None
    out_data = bpy.data.images.load(tmp_file_path)
    pixels_numpy = np.array(out_data.pixels[:])
    res_x, res_y = get_scene_resolution(scene)
    pixels_numpy.resize((res_y, res_x, 4)) # Numpy works with (y, x, channels)
    pixels_numpy = np.flip(pixels_numpy, 0) # flip vertically (in Blender y in the image points up instead of down)
    if data_map == 'Normal':
        normal = pixels_numpy[:, :, 0:3]
        return normal
    elif data_map == 'Depth':
        z = pixels_numpy[:, :, 0]
        # Points at infinity get a -1 value
        max_dist = scene.camera.data.clip_end
        INVALID_POINT = -1.0
        z[z > max_dist] = INVALID_POINT
        """ disparity """
        # If stereo also calculate disparity
        disp = None
        if not is_stereo_ok_for_disparity(scene):
            return z, disp
        baseline_m = scene.camera.data.stereo.interocular_distance # [m]
        disp = np.zeros_like(z) # disp = 0.0, on the invalid points
        f_x, _f_y, _c_x, _c_y = get_camera_parameters_intrinsic(scene) # needed for z in Cycles and for disparity
        disp[z != INVALID_POINT] = (baseline_m * f_x) / z[z != INVALID_POINT]
        # Check `tmp_file_path` if it is for the left or right camera
        suffix1 = scene.render.views[1].file_suffix
        if suffix1 in tmp_file_path: # By default, if '_R' in `tmp_file_path`
            np.negative(disp)
        return z, disp
    elif data_map == 'Segmentation':
        tmp_seg_mask = pixels_numpy[:,:,0]
        return tmp_seg_mask
    elif data_map == 'OptFlow':
        opt_flw = pixels_numpy[:,:,:2] # We are only interested in the first two channels
        # In Blender y is up instead of down, so the y optical flow should be -
        #opt_flw[:,:,1] = np.negative(opt_flw[:,:,1]) # channel 1 - y optical flow
        # However, I want forward flow (from current to next frame) instead of backward (next frame to current)
        # so I invert the optical flow both in x and y
        opt_flw[:,:,0] = np.negative(opt_flw[:,:,0])
        #opt_flw[:,:,1] = np.negative(opt_flw[:,:,1]) # Doing the `-` twice is the same as not doing
        return opt_flw


def save_data_to_npz(scene, is_stereo_activated,
                      normal0, normal1,
                      z0, z1, disp0, disp1,
                      opt_flw0, opt_flw1,
                      seg_masks0, seg_masks1,
                      seg_masks_indexes,
                      intrinsic_mat,
                      extrinsic_mat0, extrinsic_mat1,
                      obj_poses):
    # ref: https://stackoverflow.com/questions/35133317/numpy-save-some-arrays-at-once
    gt_dir_path = os.path.dirname(scene.render.filepath)
    #print(gt_dir_path)
    out_dict0 = {'optical_flow'              : opt_flw0,
                'segmentation_masks'         : seg_masks0,
                'segmentation_masks_indexes' : seg_masks_indexes,
                'intrinsic_mat'              : intrinsic_mat,
                'extrinsic_mat'              : extrinsic_mat0,
                'normal_map'                 : normal0,
                'depth_map'                  : z0,
                'disparity_map'              : disp0,
                'object_poses'               : obj_poses
               }
    if is_stereo_activated:
        # Camera 1
        suffix1 = scene.render.views[1].file_suffix # By default '_R'
        out_dict1 = {'optical_flow'              : opt_flw1,
                    'segmentation_masks'         : seg_masks1,
                    'segmentation_masks_indexes' : seg_masks_indexes,
                    'intrinsic_mat'              : intrinsic_mat,
                    'extrinsic_mat'              : extrinsic_mat1,
                    'normal_map'                 : normal1,
                    'depth_map'                  : z1,
                    'disparity_map'              : disp1,
                    'object_poses'               : obj_poses
                   }
        out_path1 = os.path.join(gt_dir_path, '{:04d}{}.npz'.format(scene.frame_current, suffix1))
        out_dict_filtered1 = {k: v for k, v in out_dict1.items() if v is not None}
        np.savez_compressed(out_path1, **out_dict_filtered1)
        # Camera 0
        suffix0 = scene.render.views[0].file_suffix # By default '_L'
        out_path0 = os.path.join(gt_dir_path, '{:04d}{}.npz'.format(scene.frame_current, suffix0))
    else:
        out_path0 = os.path.join(gt_dir_path, '{:04d}.npz'.format(scene.frame_current))
    out_dict_filtered0 = {k: v for k, v in out_dict0.items() if v is not None}
    np.savez_compressed(out_path0, **out_dict_filtered0)


@persistent
def load_handler_render_init(scene):
    """ This function is called before starting to render """
    # check if user wants to generate the ground truth data
    if scene.vision_blender.bool_save_gt_data:
        #print("Initializing a render job...")
        vision_blender = scene.vision_blender
        # 1. Set-up Passes
        if not scene.use_nodes:
            scene.use_nodes = True
        if vision_blender.bool_save_depth:
            if not bpy.context.view_layer.use_pass_z:
                bpy.context.view_layer.use_pass_z = True
        if vision_blender.bool_save_normals:
            if not bpy.context.view_layer.use_pass_normal:
                bpy.context.view_layer.use_pass_normal = True
        ## Segmentation masks and optical flow only work in Cycles
        if scene.render.engine == 'CYCLES':
            if vision_blender.bool_save_segmentation_masks:
                if not bpy.context.view_layer.use_pass_object_index:
                    bpy.context.view_layer.use_pass_object_index = True
            if vision_blender.bool_save_opt_flow:
                if not bpy.context.view_layer.use_pass_vector:
                    bpy.context.view_layer.use_pass_vector = True

        """ All the data will be saved to a MultiLayer OpenEXR image. """
        # 2. Set-up nodes
        tree = scene.node_tree
        ## Remove old nodes (from previous rendering)
        remove_old_vision_blender_nodes(tree)
        ## Create new output node
        node_output = create_node(tree, "CompositorNodeOutputFile", "output_vision_blender")
        ## Set-up the output img format
        node_output.format.file_format = 'OPEN_EXR'
        node_output.format.color_mode = 'RGBA'
        node_output.format.color_depth = '32'
        node_output.format.exr_codec = 'PIZ'
        ## Set-up output path
        TMP_FILES_PATH = os.path.join(os.path.dirname(scene.render.filepath), 'tmp_vision_blender')
        clean_folder(TMP_FILES_PATH)
        node_output.base_path = TMP_FILES_PATH

        # 3. Set-up links between nodes
        node_output.layer_slots.clear() # Remove all the default layer slots
        links = tree.links
        rl = scene.node_tree.nodes["Render Layers"] # I assumed there is always a Render Layers
        """ Normal map """
        if vision_blender.bool_save_normals:
            slot_normal = node_output.layer_slots.new('####_Normal')
            links.new(rl.outputs["Normal"], slot_normal)
        """ Depth map """
        if vision_blender.bool_save_depth:
            slot_depth = node_output.layer_slots.new('####_Depth')
            links.new(rl.outputs["Depth"], slot_depth)

        # 4. Set-up nodes and links for Cycles only (for optical flow and segmentation masks)
        if scene.render.engine == "CYCLES":
            """ Segmentation masks """
            if vision_blender.bool_save_segmentation_masks:
                # We can only generate segmentation masks if that are any labeled objects (objects w/ index set)
                non_zero_obj_ind_found = check_any_obj_with_non_zero_index()
                if non_zero_obj_ind_found:
                    slot_seg_mask = node_output.layer_slots.new('####_Segmentation_Mask')
                    links.new(rl.outputs["IndexOB"], slot_seg_mask)
            """ Optical flow - Current to next frame """
            if vision_blender.bool_save_opt_flow:
                # Create new slot in output node
                slot_opt_flow = node_output.layer_slots.new("####_Optical_Flow")
                # Get optical flow
                node_rg_separate = create_node(tree, "CompositorNodeSepRGBA", "BA_sep_vision_blender")
                node_rg_combine = create_node(tree, "CompositorNodeCombRGBA", "RG_comb_vision_blender")
                links.new(rl.outputs["Vector"], node_rg_separate.inputs["Image"])
                links.new(node_rg_separate.outputs["B"], node_rg_combine.inputs["R"])
                links.new(node_rg_separate.outputs["A"], node_rg_combine.inputs["G"])
                # Connect to output node
                links.new(node_rg_combine.outputs['Image'], slot_opt_flow)

        # 5. Save camera_info (for vision_blender_ros)
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
            stereo_info['stereo_interocular_distance [m]'] = cam.data.stereo.interocular_distance
            ### stereo pivot
            stereo_info['stereo_pivot'] = cam.data.stereo.pivot
            dict_cam_info['stereo_info'] = stereo_info
        ## save data to a json file
        gt_dir_path = os.path.dirname(scene.render.filepath)
        out_path = os.path.join(gt_dir_path, 'camera_info.json')
        with open(out_path, 'w') as tmp_file:
            json.dump(dict_cam_info, tmp_file)


@persistent
def load_handler_after_rend_frame(scene): # TODO: not sure if this is the best place to put this function, should it be above the classes?
    """ This script runs after rendering each frame """
    # ref: https://blenderartists.org/t/how-to-run-script-on-every-frame-in-blender-render/699404/2
    # check if user wants to generate the ground truth data
    if scene.vision_blender.bool_save_gt_data:
        vision_blender = scene.vision_blender
        is_stereo_activated = scene.render.use_multiview
        if is_stereo_activated:
            suffix0 = scene.render.views[0].file_suffix # By default '_L'
            suffix1 = scene.render.views[1].file_suffix # By default '_R'
        """ Camera parameters """
        ## update camera - ref: https://blender.stackexchange.com/questions/5636/how-can-i-get-the-location-of-an-object-at-each-keyframe
        #print(scene.frame_current)
        scene.frame_set(scene.frame_current) # needed to update the camera position
        intrinsic_mat = None
        extrinsic_mat0 = None
        extrinsic_mat1 = None
        if vision_blender.bool_save_cam_param:
            extrinsic_mat0 = get_camera_parameters_extrinsic(scene)
            if is_stereo_activated:
                transf0to1 = get_transf0to1(scene)
                if transf0to1 is not None:
                    extrinsic_mat0 = np.vstack((extrinsic_mat0, [0, 0, 0, 1.]))
                    extrinsic_mat1 = np.matmul(transf0to1, extrinsic_mat0)
                    # Remove homogeneous row
                    extrinsic_mat0 = extrinsic_mat0[:3,:]
                    extrinsic_mat1 = extrinsic_mat1[:3,:]
            # Intrinsic mat is the same for both stereo cameras
            f_x, f_y, c_x, c_y = get_camera_parameters_intrinsic(scene)
            intrinsic_mat = np.array([[f_x,   0,  c_x],
                                      [  0, f_y,  c_y],
                                      [  0,   0,    1]])
        """ Objects' pose """
        obj_poses = None
        if vision_blender.bool_save_obj_poses:
            obj_poses = get_obj_poses()
        """ Get the data from the output node """
        normal0 = None
        normal1 = None
        z0 = None
        z1 = None
        disp0 = None
        disp1 = None
        seg_masks0 = None
        seg_masks1 = None
        seg_masks_indexes = None
        opt_flw0 = None
        opt_flw1 = None
        if check_if_node_exists(scene.node_tree, 'output_vision_blender'):
            node_output = scene.node_tree.nodes['output_vision_blender']
            TMP_FILES_PATH = node_output.base_path
            """ Normal map """
            if vision_blender.bool_save_normals:
                if is_stereo_activated:
                    tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Normal{}.exr'.format(scene.frame_current, suffix1))
                    normal1 = load_file_data_to_numpy(scene, tmp_file_path1, 'Normal')
                    tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Normal{}.exr'.format(scene.frame_current, suffix0))
                else:
                    tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Normal.exr'.format(scene.frame_current))
                normal0 = load_file_data_to_numpy(scene, tmp_file_path0, 'Normal')
            """ Depth + Disparity """
            if vision_blender.bool_save_depth:
                if is_stereo_activated:
                    tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Depth{}.exr'.format(scene.frame_current, suffix1))
                    z1, disp1 = load_file_data_to_numpy(scene, tmp_file_path1, 'Depth')
                    tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Depth{}.exr'.format(scene.frame_current, suffix0))
                else:
                    tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Depth.exr'.format(scene.frame_current))
                z0, disp0 = load_file_data_to_numpy(scene, tmp_file_path0, 'Depth')
            if scene.render.engine == "CYCLES":
                """ Segmentation masks """
                if vision_blender.bool_save_segmentation_masks and check_any_obj_with_non_zero_index():
                    seg_masks_indexes = get_struct_array_of_obj_indexes()
                    if is_stereo_activated:
                        tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Segmentation_Mask{}.exr'.format(scene.frame_current, suffix1))
                        seg_masks1 = load_file_data_to_numpy(scene, tmp_file_path1, 'Segmentation')
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Segmentation_Mask{}.exr'.format(scene.frame_current, suffix0))
                    else:
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Segmentation_Mask.exr'.format(scene.frame_current))
                    seg_masks0 = load_file_data_to_numpy(scene, tmp_file_path0, 'Segmentation')
                """ Optical flow - Forward -> from current to next frame"""
                if vision_blender.bool_save_opt_flow:
                    if is_stereo_activated:
                        tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Optical_Flow{}.exr'.format(scene.frame_current, suffix1))
                        opt_flw1 = load_file_data_to_numpy(scene, tmp_file_path1, 'OptFlow')
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Optical_Flow{}.exr'.format(scene.frame_current, suffix0))
                    else:
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Optical_Flow.exr'.format(scene.frame_current))
                    opt_flw0 = load_file_data_to_numpy(scene, tmp_file_path0, 'OptFlow')
            # Optional step - delete the tmp output files
            clean_folder(TMP_FILES_PATH)
        """ Save data """
        save_data_to_npz(scene, is_stereo_activated,
                         normal0, normal1,
                         z0, z1, disp0, disp1,
                         opt_flw0, opt_flw1,
                         seg_masks0, seg_masks1,
                         seg_masks_indexes, # Same indexes for both cameras
                         intrinsic_mat, # Both cameras have the same intrinsic parameters
                         extrinsic_mat0, extrinsic_mat1,
                         obj_poses) # Object poses are relative to the world coordinate frame, so they are the same


@persistent
def load_handler_after_rend_finish(scene):
    if check_if_node_exists(scene.node_tree, 'output_vision_blender'):
        node_output = scene.node_tree.nodes['output_vision_blender']
        TMP_FILES_PATH = node_output.base_path
        shutil.rmtree(TMP_FILES_PATH)


# classes
class MyAddonProperties(PropertyGroup):
    # booleans
    bool_save_gt_data : BoolProperty(
        name = "Ground truth",
        description = "Save ground truth data",
        default = False,
        )
    bool_save_depth : BoolProperty(
        name = "Depth",
        description = "Save depth maps",
        default = True
        )
    bool_save_normals : BoolProperty(
        name = "Normals",
        description = "Save surface normals",
        default = True
        )
    bool_save_cam_param : BoolProperty(
        name = "Camera parameters",
        description = "Save camera parameters",
        default = True
        )
    bool_save_opt_flow : BoolProperty(
        name = "Optical Flow",
        description = "Save optical flow",
        default = True
        )
    bool_save_segmentation_masks : BoolProperty(
        name = "Semantic",
        description = "Save semantic segmentation",
        default = True
        )
    bool_save_obj_poses : BoolProperty(
        name = "Objects Pose",
        description = "Save object pose",
        default = True
        )

    def get_cam_intrinsic(self):
        scene = bpy.context.scene
        f_x, f_y, c_x, c_y = get_camera_parameters_intrinsic(scene)
        intrinsic_mat = np.array([[f_x,   0,  c_x],
                                  [  0, f_y,  c_y],
                                  [  0,   0,    1]])
        return intrinsic_mat.flatten("F").tolist()

    cam_intrinsic : FloatVectorProperty(
        name="Intrinsic",
        size=9,
        subtype="MATRIX",
        get=get_cam_intrinsic
        )

    def get_cam_extrinsic(self):
        scene = bpy.context.scene
        extr_mat = get_camera_parameters_extrinsic(scene)
        extr_mat = np.vstack([extr_mat, [0, 0, 0, 1]])
        return extr_mat.flatten("F").tolist()


    cam_extrinsic : FloatVectorProperty(
        name="extrinsic",
        size=16,
        subtype="MATRIX",
        get=get_cam_extrinsic
        )

class GroundTruthGeneratorPanel(Panel):
    """Creates a Panel in the Output properties window for exporting ground truth data"""
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "output"


class RENDER_PT_gt_generator(GroundTruthGeneratorPanel):
    """Parent panel"""
    bl_label = "VisionBlender UI"
    bl_idname = "RENDER_PT_gt_generator"
    COMPAT_ENGINES = {'BLENDER_EEVEE', 'CYCLES'}
    #bl_options = {'DEFAULT_CLOSED'} # makes panel closed by default
    #bl_options = {'HIDE_HEADER'} # shows the panel on the top, not collapsable

    @classmethod
    def poll(cls, context):
        return (context.engine in cls.COMPAT_ENGINES)

    def draw_header(self, context):
        self.layout.prop(context.scene.vision_blender, "bool_save_gt_data", text="")

    def draw(self, context):
        scene = context.scene
        rd = scene.render
        layout = self.layout

        vision_blender = scene.vision_blender
        layout.active = vision_blender.bool_save_gt_data

        layout.use_property_split = False
        layout.use_property_decorate = False  # No animation.

        # boolean flags to control what is being saved
        #  reference: https://github.com/sobotka/blender/blob/662d94e020f36e75b9c6b4a258f31c1625573ee8/release/scripts/startup/bl_ui/properties_output.py
        flow = layout.grid_flow(row_major=True, columns=0, even_columns=True, even_rows=False, align=False)
        col = flow.column()
        col.prop(vision_blender, "bool_save_depth", text="Depth / Disparity")
        col = flow.column()
        col.enabled = context.engine == 'CYCLES' # ref: https://blenderartists.org/t/how-to-disable-a-checkbox-when-a-dropdown-option-is-picked/612801/2
        col.prop(vision_blender, "bool_save_segmentation_masks", text="Segmentation Masks")
        col = flow.column()
        col.prop(vision_blender, "bool_save_normals", text="Normals")
        col = flow.column()
        col.enabled = context.engine == 'CYCLES'
        col.prop(vision_blender, "bool_save_opt_flow", text="Optical Flow")
        col = flow.column()
        col.prop(vision_blender, "bool_save_obj_poses", text="Objects' Pose")
        col = flow.column()
        col.prop(vision_blender, "bool_save_cam_param", text="Camera Parameters")

        if context.engine != 'CYCLES':
            col = layout.column(align=True)
            col.label(text="Optical Flow and Segmentation Masks requires Cycles!", icon='ERROR')

        if vision_blender.bool_save_segmentation_masks and context.engine == 'CYCLES':
            non_zero_obj_ind_found = check_any_obj_with_non_zero_index()
            if not non_zero_obj_ind_found:
                col = layout.column(align=True)
                col.label(text="No object index found yet for Segmentation Masks...", icon='ERROR')

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
        layout.label(text="Extrinsic parameters:")

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

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    # register the properties
    bpy.types.Scene.vision_blender = PointerProperty(type=MyAddonProperties)
    # register the function being called when rendering starts
    bpy.app.handlers.render_init.append(load_handler_render_init)
    # register the function being called after rendering each frame
    bpy.app.handlers.render_post.append(load_handler_after_rend_frame)
    # register the function being called after rendering all the frames, or being cancelled
    bpy.app.handlers.render_complete.append(load_handler_after_rend_finish)
    bpy.app.handlers.render_cancel.append(load_handler_after_rend_finish)


def unregister():
    # unregister the classes
    for cls in classes:
        bpy.utils.unregister_class(cls)
    # unregister the properties
    del bpy.types.Scene.vision_blender
    # unregister the function being called when rendering starts
    bpy.app.handlers.render_init.remove(load_handler_render_init)
    # unregister the function being called after rendering each frame
    bpy.app.handlers.render_post.remove(load_handler_after_rend_frame)
    # unregister the function being called after rendering all the frames, or being cancelled
    bpy.app.handlers.render_complete.append(load_handler_after_rend_finish)
    bpy.app.handlers.render_cancel.append(load_handler_after_rend_finish)


if __name__ == "__main__": # only for live edit.
    register()
