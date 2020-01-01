# bl_info
bl_info = {
        "name":"Ground Truth Generation",
        "description":"Generate ground truth data (e.g., depth map) for Computer Vision applications.",
        "author":"Joao Cartucho, YP Li",
        "version":(1, 0),
        "blender":(2, 81, 0),
        "location":"PROPERTIES",
        "warning":"", # used for warning icon and text in addons panel
        "wiki_url":"",
        "support":"TESTING",
        "category":"Render"
    }

import bpy
import numpy as np # TODO: check if blender has numpy by dafault
from bpy.props import (StringProperty,
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

# classes
class MyAddonProperties(PropertyGroup):
    # boolean to choose between saving ground truth data or not

    # output dir path
    gt_dir_path : StringProperty(
        name = "",
        default = "",
        description = "Define the directory path for storing data",
        subtype = 'DIR_PATH'
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

        """ select output path """
        layout.label(text="Output directory path:")
        
        # get dir path to store the generated data
        col = layout.column()
        col.prop(context.scene.my_addon, 'gt_dir_path')

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


# registration
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # register the properties
    bpy.types.Scene.my_addon = PointerProperty(type=MyAddonProperties)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    # unregister the properties
    del bpy.types.Scene.my_addon

if __name__ == "__main__":
    register()

# reference: https://github.com/sobotka/blender/blob/662d94e020f36e75b9c6b4a258f31c1625573ee8/release/scripts/startup/bl_ui/properties_output.py
