# bl_info # read more: https://wiki.blender.org/wiki/Process/Addons/Guidelines/metainfo
bl_info = {
        "name":"VisionBlender - Computer Vision Ground Truth Generation",
        "description":"Generate ground truth data (e.g., depth map) for Computer Vision applications.",
        "author":"Joao Cartucho",
        "version":(1, 0),
        "blender":(2, 83, 4),
        "location":"PROPERTIES",
        "warning":"", # used for warning icon and text in addons panel
        "wiki_url":"https://github.com/Cartucho/vision_blender",
        "support":"COMMUNITY",
        "category":"Render"
    }

import json
import os
import shutil # to remove files for Cycles
import numpy as np # TODO: check if Blender has numpy by default

import bpy
from bpy.props import (#StringProperty, # TODO: not being used
                   BoolProperty,
                   #IntProperty, # TODO: not being used
                   #FloatProperty, # TODO: not being used
                   #EnumProperty, # TODO: not being used
                   PointerProperty
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


def get_objects_pose(scene):
    poses_labels = []
    poses = None
    for obj in bpy.data.objects:
        poses_labels.append(obj.name)
        mw = np.asarray(obj.matrix_world)
        if poses is None:
            poses = mw[None]
        else:
            poses = np.vstack((poses, mw[None]))
    # conver list to array (so that we can save in the npz file)
    poses_labels = np.array(poses_labels)
    return poses_labels, poses


def correct_cycles_depth(z_map, res_x, res_y, f_x, f_y, c_x, c_y, INVALID_POINT):
    for y in range(res_y):
        b = ((c_y - y) / f_y)
        for x in range(res_x):
            val = z_map[y][x]
            if val != INVALID_POINT:
                a = ((c_x - x) / f_x)
                z_map[y][x] = val / np.linalg.norm([1, a, b])
    return z_map


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
            print(file_path)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


def check_any_obj_with_index():
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


def get_img_extension(file_format):
    if file_format == 'PNG':
        return '.png'
    elif file_format == 'JPEG':
        return '.jpg'
    elif file_format == 'TARGA':
        return '.tga'
    elif file_format == 'OPEN_EXR':
        return '.exr'
    # TODO: test other formats
    #bpy.path.extensions_image


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
            if not scene.view_layers["View Layer"].use_pass_z:
                scene.view_layers["View Layer"].use_pass_z = True
        if vision_blender.bool_save_normals:
            if not scene.view_layers["View Layer"].use_pass_normal:
                scene.view_layers["View Layer"].use_pass_normal = True
        if scene.render.engine == 'CYCLES':
            if vision_blender.bool_save_segmentation_masks:
                if not scene.view_layers["View Layer"].use_pass_object_index:
                    scene.view_layers["View Layer"].use_pass_object_index = True
            if vision_blender.bool_save_opt_flow:
                if not scene.view_layers["View Layer"].use_pass_vector:
                    scene.view_layers["View Layer"].use_pass_vector = True

        # 2. Set-up nodes
        tree = scene.node_tree
        remove_old_vision_blender_nodes(tree) # Clean nodes from previous rendering
        rl = scene.node_tree.nodes["Render Layers"] # I assumed there is always a Render Layers
        if vision_blender.bool_save_normals or vision_blender.bool_save_depth:
            node_norm_and_z = create_node(tree, "CompositorNodeViewer", "normal_and_zmap_vision_blender")

        # 3. Set-up links between nodes
        ## create new links if necessary
        links = tree.links
        ## Trick: we already have the RGB image so we can connect the Normal to Image
        ##        and the Z to the Alpha channel
        if vision_blender.bool_save_normals:
            links.new(rl.outputs["Normal"], node_norm_and_z.inputs["Image"])
        if vision_blender.bool_save_depth:
            links.new(rl.outputs["Depth"], node_norm_and_z.inputs["Alpha"])

        # 4. Set-up nodes and links for Cycles only (for optical flow and segmentation masks)
        """
            VIEWER_FIXED refers to a Blender bug that does not allow us to access all the data from the script directly,
                         instead we have to save the data into output pictures and then read the data from those pictures.

                         bug link: https://developer.blender.org/T54314

            Currently, it is also not possible to read all the segmentation_masks in the output image file directly.
            Therefore, we have to first slipt each of the masks individually and create an output image for each mask.
            TODO: probably, there is a way to merge all the different masks into a single image.
        """
        VIEWER_FIXED = False # TODO: change code when https://developer.blender.org/T54314 is fixed
        if scene.render.engine == "CYCLES":
            if VIEWER_FIXED:
                # Here I would be creating a viewer for each type of input
                if vision_blender.bool_save_segmentation_masks:
                    node_segmentation_masks = create_node(tree, "CompositorNodeViewer", "segmentation_masks_vision_blender")
                    links.new(rl.outputs["IndexOB"], node_segmentation_masks.inputs["Image"])
                    # TODO: check, on the viewer node if we can read the indexes of each mask directly, so I don't need to split using ID Mask
                if vision_blender.bool_save_opt_flow:
                    node_opt_flow = create_node(tree, "CompositorNodeViewer", "opt_flow_vision_blender")
                    ## The optical flow needs to be connected to both `Image` and `Alpha`
                    links.new(rl.outputs["Vector"], node_opt_flow.inputs["Image"])
                    links.new(rl.outputs["Vector"], node_opt_flow.inputs["Alpha"])
            else:
                path_render = os.path.dirname(scene.render.filepath)
                segmentation_masks_path = os.path.join(path_render, "segmentation_masks_vision_blender")
                opt_flow_path = os.path.join(path_render, "opt_flow_vision_blender")
                """ segmentation masks """
                clean_folder(segmentation_masks_path)
                if vision_blender.bool_save_segmentation_masks:
                    obj_ind_found = check_any_obj_with_index() # Check if there are any object with object index set
                    if obj_ind_found:
                        ## create output node
                        node_segmentation_masks = create_node(tree, "CompositorNodeOutputFile", "segmentation_masks_vision_blender")
                        ### set-up the output img format
                        node_segmentation_masks.format.file_format = 'TARGA'
                        ### set-up the output path
                        node_segmentation_masks.base_path = segmentation_masks_path
                        ## For the segmentation masks we need to set-up an output image for each pass index
                        ### ref: https://blender.stackexchange.com/questions/18243/how-to-use-index-passes-in-other-compositing-packages
                        node_segmentation_masks.layer_slots.clear()

                        for obj in bpy.data.objects:
                            obj_pass_ind = obj.pass_index
                            if obj_pass_ind != 0:
                                ind_str = '{}_'.format(obj_pass_ind)
                                # ref: https://blender.stackexchange.com/questions/65013/not-able-to-add-node-sockets-to-an-existing-node-using-python-scripting
                                node_segmentation_masks.layer_slots.new(ind_str)
                                node_id_mask = create_node(tree, "CompositorNodeIDMask", "{}_mask_vision_blender".format(ind_str))
                                node_id_mask.index = obj_pass_ind
                                # create link
                                links.new(node_id_mask.outputs["Alpha"], node_segmentation_masks.inputs[ind_str])
                                links.new(rl.outputs["IndexOB"], node_id_mask.inputs["ID value"])
                """ optical flow - Current to next frame """
                clean_folder(opt_flow_path)
                if vision_blender.bool_save_opt_flow:
                    node_opt_flow = create_node(tree, 'CompositorNodeOutputFile', 'opt_flow_vision_blender')
                    ### set-up the output img format
                    node_opt_flow.format.file_format = 'OPEN_EXR'
                    node_opt_flow.format.exr_codec = 'PIZ'
                    ### set-up the output path
                    node_opt_flow.base_path = opt_flow_path
                    node_rg_separate = create_node(tree, "CompositorNodeSepRGBA", "BA_sep_vision_blender")
                    node_rg_combine = create_node(tree, "CompositorNodeCombRGBA", "RG_comb_vision_blender")
                    links.new(rl.outputs["Vector"], node_rg_separate.inputs["Image"])
                    links.new(node_rg_separate.outputs["B"], node_rg_combine.inputs["R"])
                    links.new(node_rg_separate.outputs["A"], node_rg_combine.inputs["G"])
                    links.new(node_rg_combine.outputs['Image'], node_opt_flow.inputs['Image'])


        # 4. Save camera_info (for vision_blender_ros)
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
        gt_dir_path = os.path.dirname(scene.render.filepath)
        #print(gt_dir_path)
        # save ground truth data
        #print(scene.frame_current)
        """ Camera parameters """
        ## update camera - ref: https://blender.stackexchange.com/questions/5636/how-can-i-get-the-location-of-an-object-at-each-keyframe
        scene.frame_set(scene.frame_current) # needed to update the camera position
        ### extrinsic
        extrinsic_mat = get_camera_parameters_extrinsic(scene) # needed for objects' pose
        ### intrinsic
        f_x, f_y, c_x, c_y = get_camera_parameters_intrinsic(scene) # needed for z in Cycles and for disparity
        """ Depthmap + Normal """
        ## get data
        normal = None
        z = None
        disp = None
        res_x, res_y = get_scene_resolution(scene)
        if vision_blender.bool_save_depth or vision_blender.bool_save_normals:
            pixels = bpy.data.images['Viewer Node'].pixels
            #print(len(pixels)) # size = width * height * 4 (rgba)
            pixels_numpy = np.array(pixels[:])
            #   .---> y
            #   |
            #   |
            #   v
            #    x
            pixels_numpy.resize((res_y, res_x, 4)) # Numpy works with (y, x, channels)
            if vision_blender.bool_save_normals:
                normal = pixels_numpy[:, :, 0:3]
                normal = np.flip(normal, 0) # flip vertically (in Blender y in the image points up instead of down)
            if vision_blender.bool_save_depth:
                z = pixels_numpy[:, :, 3]
                z = np.flip(z, 0) # flip vertically (in Blender y in the image points up instead of down)
                # points at infinity get a -1 value
                max_dist = scene.camera.data.clip_end
                INVALID_POINT = -1.0
                #normal[z > max_dist] = INVALID_POINT # TODO: I think this is not necessary, maybe there is another way to see if that normal point is valid?
                z[z > max_dist] = INVALID_POINT
                if scene.render.engine == "CYCLES":
                    z = correct_cycles_depth(z, res_x, res_y, f_x, f_y, c_x, c_y, INVALID_POINT)
                """ disparity """
                # if stereo also calculate disparity
                cam = scene.camera
                if (scene.render.use_multiview and
                    cam.data.stereo.convergence_mode == 'PARALLEL' and
                    cam.data.stereo.pivot == 'LEFT'): # TODO: handle the case where the pivot is the right camera
                    baseline_m = cam.data.stereo.interocular_distance # [m]
                    disp = np.zeros_like(z) # disp = 0.0, on the invalid points
                    disp[z != INVALID_POINT] = (baseline_m * f_x) / z[z != INVALID_POINT]


        """ Segmentation Masks + Opt flow"""
        seg_masks = None
        opt_flw = None
        if vision_blender.bool_save_segmentation_masks or vision_blender.bool_save_opt_flow:
            VIEWER_FIXED = False # TODO: change code when https://developer.blender.org/T54314 is fixed
            if scene.render.engine == "CYCLES":
                if VIEWER_FIXED:
                    # TODO: make each Image Viewer active one-by-one and copy values
                    pass
                else:
                    # in `load_handler_render_init` and at each rendering frame we clean these folders, so all the images are output data
                    # TODO: this part of the code is really slow, essentially I am opening the images one by one, so many segmentation masks would make it even slower
                    if vision_blender.bool_save_segmentation_masks:
                        # check if node exists
                        if check_if_node_exists(scene.node_tree, 'segmentation_masks_vision_blender'):
                            seg_masks_node = scene.node_tree.nodes['segmentation_masks_vision_blender']
                            seg_masks_path = seg_masks_node.base_path
                            file_format = seg_masks_node.format.file_format
                            extension = get_img_extension(file_format)
                            if os.path.isdir(seg_masks_path):
                                for tmp_file in os.listdir(seg_masks_path):
                                    if tmp_file.endswith(extension):
                                        if seg_masks is None:
                                            seg_masks = np.zeros((res_y, res_x), dtype=np.uint16)
                                        img_path = os.path.join(seg_masks_path, tmp_file)
                                        obj_pass_ind = tmp_file.split('_', 1)[0] # add nsplits = 1 for efficiency 
                                        #print(obj_pass_ind)
                                        tmp_img = bpy.data.images.load(img_path)
                                        tmp_seg_mask = np.array(tmp_img.pixels[:])
                                        tmp_seg_mask.resize((res_y, res_x, 4)) # Numpy works with (y, x, channels)
                                        tmp_seg_mask = tmp_seg_mask[:,:,0]
                                        tmp_seg_mask = np.flip(tmp_seg_mask, 0) # flip vertically (in Blender y in the image points up instead of down)
                                        seg_masks[tmp_seg_mask != 0] = obj_pass_ind
                                        os.remove(img_path)
                                os.rmdir(seg_masks_path)
                    if vision_blender.bool_save_opt_flow:
                        """ Forward optical flow - from current to next frame """
                        if check_if_node_exists(scene.node_tree, 'opt_flow_vision_blender'):
                            opt_flw_node = scene.node_tree.nodes['opt_flow_vision_blender']
                            opt_flw_path = opt_flw_node.base_path
                            file_format = opt_flw_node.format.file_format
                            extension = get_img_extension(file_format)
                            if os.path.isdir(opt_flw_path):
                                for tmp_file in os.listdir(opt_flw_path):
                                    #bpy.path.extensions_image
                                    if tmp_file.endswith(extension):
                                        img_path = os.path.join(opt_flw_path, tmp_file)
                                        tmp_img = bpy.data.images.load(img_path)
                                        #tmp_img.alpha_mode = 'STRAIGHT'
                                        opt_flw = np.array(tmp_img.pixels[:])
                                        opt_flw.resize((res_y, res_x, 4)) # Numpy works with (y, x, channels)
                                        opt_flw = opt_flw[:,:,:2] # We are only interested in the first two channels
                                        opt_flw = np.flip(opt_flw, 0) # flip vertically (in Blender y in the image points up instead of down)
                                        # In Blender y is up instead of down, so the y optical flow should be -
                                        #opt_flw[:,:,1] = np.negative(opt_flw[:,:,1]) # channel 1 - y optical flow
                                        # However, I want forward flow (from current to next frame) instead of backward (next frame to current)
                                        # so I invert the optical flow both in x and y
                                        opt_flw[:,:,0] = np.negative(opt_flw[:,:,0])
                                        #opt_flw[:,:,1] = np.negative(opt_flw[:,:,1]) # Doing the `-` twice is the same as not doing

                                        os.remove(img_path)
                                os.rmdir(opt_flw_path)
        seg_masks_indexes = None
        if seg_masks is not None:
            seg_masks_indexes = get_struct_array_of_obj_indexes()
        """ Objects' pose """
        object_pose_labels = None
        object_pose_mats = None
        if vision_blender.bool_save_obj_poses:
            object_pose_labels, object_pose_mats = get_objects_pose(scene)
        """ Save data """
        intrinsic_mat = None
        if not vision_blender.bool_save_cam_param:
            extrinsic_mat = None
        else:
            intrinsic_mat = np.array([[f_x,   0,  c_x],
                             [  0, f_y,  c_y],
                             [  0,   0,    1]])
        # Blender by default assumes a padding of 4 digits
        out_path = os.path.join(gt_dir_path, '{:04d}.npz'.format(scene.frame_current))
        #print(out_path)
        out_dict = {'optical_flow'               : opt_flw,
                    'segmentation_masks'         : seg_masks,
                    'segmentation_masks_indexes' : seg_masks_indexes,
                    'intrinsic_mat'              : intrinsic_mat,
                    'extrinsic_mat'              : extrinsic_mat,
                    'normal_map'                 : normal,
                    'depth_map'                  : z,
                    'disparity_map'              : disp,
                    'object_pose_labels'         : object_pose_labels,
                    'object_pose_mats'           : object_pose_mats
                   }
        out_dict_filtered = {k: v for k, v in out_dict.items() if v is not None}
        np.savez_compressed(out_path, **out_dict_filtered)
        # ref: https://stackoverflow.com/questions/35133317/numpy-save-some-arrays-at-once

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


class GroundTruthGeneratorPanel(Panel):
    """Creates a Panel in the Output properties window for exporting ground truth data"""
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "output"


class RENDER_PT_gt_generator(GroundTruthGeneratorPanel):
    """Parent panel"""
    global intrinsic_mat
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
            obj_ind_found = check_any_obj_with_index()
            if not obj_ind_found:
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

if __name__ == "__main__": # only for live edit.
    register()
