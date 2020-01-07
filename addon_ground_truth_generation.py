# bl_info
bl_info = {
        "name":"Ground Truth Generation",
        "description":"Generate ground truth data (e.g., depth map) for Computer Vision applications.",
        "author":"Joao Cartucho, YP Li",
        "version":(1, 0),
        "blender":(2, 81, 16),
        "location":"PROPERTIES",
        "warning":"", # used for warning icon and text in addons panel
        "wiki_url":"",
        "support":"TESTING",
        "category":"Render"
    }

import bpy
import os
import numpy as np # TODO: check if Blender has numpy by default
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


intrinsic_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

# classes
class MyAddonProperties(PropertyGroup):
    # boolean to choose between saving ground truth data or not
    save_gt_data : BoolProperty(
        name = "Ground truth",
        default = True,
        description = "Save ground truth data",
    )


#class RENDER_OT_save_gt_data(bpy.types.Operator):
#    """ Saves the ground truth data that was created with the add-on """
#    bl_label = "Save ground truth data"
#    bl_idname = "RENDER_OT_" # How Blender refers to this operator

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
    COMPAT_ENGINES = {'BLENDER_RENDER', 'BLENDER_EEVEE', 'BLENDER_WORKBENCH'}
    bl_options = {'DEFAULT_CLOSED'}


    def draw_header(self, context):
        rd = context.scene.render
        self.layout.prop(context.scene.my_addon, "save_gt_data", text="")


    def draw(self, context):
        layout = self.layout
        layout.active = context.scene.my_addon.save_gt_data

        layout.use_property_split = False
        layout.use_property_decorate = False  # No animation.

        # Get camera parameters
        """ show intrinsic parameters """
        layout.label(text="Intrinsic parameters [pixels]:")

        focal_length = bpy.context.scene.camera.data.lens # TODO: I am assuming [mm]
        resolution_scale = (bpy.context.scene.render.resolution_percentage / 100.0)
        resolution_x = bpy.context.scene.render.resolution_x * resolution_scale # [pixels]
        resolution_y = bpy.context.scene.render.resolution_y * resolution_scale # [pixels]
        sensor_width = bpy.context.scene.camera.data.sensor_width # [mm]
        sensor_height = bpy.context.scene.camera.data.sensor_height # [mm]
        ### f_x
        f_x = focal_length * (resolution_x / sensor_width) # [pixels]
        ### f_y
        f_y = focal_length * (resolution_y / sensor_height) # [pixels]
        scale_x = bpy.context.scene.render.pixel_aspect_x
        scale_y = bpy.context.scene.render.pixel_aspect_y
        pixel_aspect_ratio = scale_x / scale_y
        if pixel_aspect_ratio != 1.0:
            if bpy.context.scene.camera.data.sensor_fit == 'VERTICAL':
                f_x = f_x / pixel_aspect_ratio
            else:
                f_y = f_y * pixel_aspect_ratio  
        ### c_x
        shift_x = bpy.context.scene.camera.data.shift_x # [mm]
        c_x = (resolution_x - 1) / 2.0 #+ shift_pixels_x [pixels] TODO: shift_x to pixel
        ### c_y
        shift_y = bpy.context.scene.camera.data.shift_y # [mm]
        c_y = (resolution_y - 1) /2.0 #+ shift_pixels_y [pixels] TODO: shift_y to pixel

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

        ## update the global variable of the intrinsic mat
        intrinsic_mat[0, 0] = f_x
        intrinsic_mat[0, 2] = c_x
        intrinsic_mat[1, 1] = f_y
        intrinsic_mat[1, 2] = c_y

        """ show extrinsic parameters """
        layout.label(text="Extrinsic parameters [pixels]:")

        cam_mat_world = bpy.context.scene.camera.matrix_world.inverted()
        
        box_ext = self.layout.box()
        col_ext = box_ext.column()

        row_ext_0 = col_ext.split()
        row_ext_0.label(text=str(cam_mat_world[0][0]))
        row_ext_0.label(text=str(cam_mat_world[0][1]))
        row_ext_0.label(text=str(cam_mat_world[0][2]))
        row_ext_0.label(text=str(cam_mat_world[0][3]))

        row_ext_1 = col_ext.split()
        row_ext_1.label(text=str(cam_mat_world[1][0]))
        row_ext_1.label(text=str(cam_mat_world[1][1]))
        row_ext_1.label(text=str(cam_mat_world[1][2]))
        row_ext_1.label(text=str(cam_mat_world[1][3]))

        row_ext_2 = col_ext.split()
        row_ext_2.label(text=str(cam_mat_world[2][0]))
        row_ext_2.label(text=str(cam_mat_world[2][1]))
        row_ext_2.label(text=str(cam_mat_world[2][2]))
        row_ext_2.label(text=str(cam_mat_world[2][3]))

classes = (
    MyAddonProperties,
    RENDER_PT_gt_generator,
    #RENDER_OT_save_gt_data
)


@persistent
def load_handler_render_init(scene):
    print("Initialization of a render job")

    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["View Layer"].use_pass_z = True
    bpy.context.scene.view_layers["View Layer"].use_pass_normal = True

    # check connections
    tree = bpy.context.scene.node_tree
    rl = bpy.context.scene.node_tree.nodes["Render Layers"]
    viewer_ind = tree.nodes.find("Viewer")
    v = None
    if viewer_ind == -1:
        v = tree.nodes.new("CompositorNodeViewer")
    else:
        v = bpy.context.scene.node_tree.nodes["Viewer"]
    # create new links if necessary
    links = tree.links
    if not v.inputs["Image"].is_linked:
        links.new(rl.outputs["Image"], v.inputs["Image"])
    if not v.inputs["Z"].is_linked:
        links.new(rl.outputs["Depth"], v.inputs["Z"])


@persistent
def load_handler_render_frame(scene): # TODO: not sure if this is the best place to put this
    """ This script runs after rendering each frame """
    # ref: https://blenderartists.org/t/how-to-run-script-on-every-frame-in-blender-render/699404/2
    # check if user wants to generate the ground truth data
    if scene.my_addon.save_gt_data:
        gt_dir_path = scene.render.filepath
        #print(gt_dir_path)
        # save ground truth data
        #print(scene.frame_current)
        """ camera parameters """
        ### extrinsic
        cam_mat_world = bpy.context.scene.camera.matrix_world.inverted()
        extrinsic_mat = np.array(cam_mat_world)
        #### note: by default blender has 4 padded zeros
        cam_para_path_extr = os.path.join(gt_dir_path, 'cam_param_extrinsic_{}.out'.format(scene.frame_current)) # TODO: %04d
        np.savetxt(cam_para_path_extr, extrinsic_mat)
        ### intrinsic
        #print(intrinsic_mat)
        cam_para_path_intr = os.path.join(gt_dir_path, 'cam_param_intrinsic_{}.out'.format(scene.frame_current))
        np.savetxt(cam_para_path_intr, intrinsic_mat)


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
    bpy.app.handlers.render_post.append(load_handler_render_frame)


def unregister():
    # unregister the classes
    for cls in classes:
        bpy.utils.unregister_class(cls)
    # unregister the properties
    del bpy.types.Scene.my_addon
    # unregister the function being called when rendering each frame
    bpy.app.handlers.render_init.remove(load_handler_render_init)
    # unregister the function being called when rendering each frame
    bpy.app.handlers.render_post.remove(load_handler_render_frame)

if __name__ == "__main__":
    register()

# reference: https://github.com/sobotka/blender/blob/662d94e020f36e75b9c6b4a258f31c1625573ee8/release/scripts/startup/bl_ui/properties_output.py
