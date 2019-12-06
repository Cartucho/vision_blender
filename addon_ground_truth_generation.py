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
from pyntcloud import PyntCloud

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
    ## output dir path
    gt_dir_path : StringProperty(
        name = "",
        default = "",
        description = "Define the directory path for storing data",
        subtype = 'DIR_PATH'
    )

    is_cam_matrix_selected : BoolProperty(
        name="Enable or Disable", # TODO: change
        description="A simple bool property",
        default = False
    )

class RENDER_OT_save_projection_matrix(Operator):
    """Save the camera's projection matrix"""
    bl_label = "Save Ground Truth"
    bl_idname = "scene.projection"

    # only saves if at least one of the options is selected

    def execute(self, context):
        #print(context.scene.my_addon["testprop"])
        #context.scene.my_addon["testprop"] = 15.0
        print("save {}".format(5));
        return {'FINISHED'}


class GroundTruthGeneratorPanel(Panel):
    """Creates a Panel in the Output properties window for exporting ground truth data"""
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "output"


class RENDER_PT_gt_generator(GroundTruthGeneratorPanel):
    """Parent panel"""
    bl_label = "Ground Truth Generator"
    bl_idname = "RENDER_PT_gt_generator"
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False  # No animation.

        layout.label(text="Output directory path:")
        
        # get dir path to store the generated data
        col = layout.column()
        col.prop(context.scene.my_addon, 'gt_dir_path')
        
        # save projection matrix
        col.operator(RENDER_OT_save_projection_matrix.bl_idname)
        layout.label(text="Select what you want to save:")


class RENDER_PT_camera_matrix(GroundTruthGeneratorPanel, Panel):
    """Panel showing intrinsic, extrinsic, and projection matrix"""
    bl_label = "Camera Matrix"
    bl_parent_id = "RENDER_PT_gt_generator"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'BLENDER_RENDER', 'BLENDER_EEVEE', 'BLENDER_WORKBENCH'}

    def draw_header(self, context):
        self.layout.prop(context.scene.my_addon, "is_cam_matrix_selected", text="")
        
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.
        layout.active = context.scene.my_addon.is_cam_matrix_selected

        # Get camera parameters
        """ intrinsic """
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

        """ extrinsic """
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
        
        """ projection matrix """
        layout.label(text="Projection matrix [pixels]:")
        depsgraph = bpy.context.evaluated_depsgraph_get()
        camera_matrix = bpy.context.scene.camera.calc_matrix_camera(depsgraph, x=resolution_x, y=resolution_y, scale_x=scale_x, scale_y=scale_y)
        projection_matrix = camera_matrix @ cam_mat_world

        box_proj = self.layout.box()
        col_proj = box_proj.column()

        row_proj_0 = col_proj.split()
        row_proj_0.label(text=str(projection_matrix[0][0]))
        row_proj_0.label(text=str(projection_matrix[0][1]))
        row_proj_0.label(text=str(projection_matrix[0][2]))
        row_proj_0.label(text=str(projection_matrix[0][3]))

        row_proj_1 = col_proj.split()
        row_proj_1.label(text=str(projection_matrix[1][0]))
        row_proj_1.label(text=str(projection_matrix[1][1]))
        row_proj_1.label(text=str(projection_matrix[1][2]))
        row_proj_1.label(text=str(projection_matrix[1][3]))

        row_proj_2 = col_proj.split()
        row_proj_2.label(text=str(projection_matrix[2][0]))
        row_proj_2.label(text=str(projection_matrix[2][1]))
        row_proj_2.label(text=str(projection_matrix[2][2]))
        row_proj_2.label(text=str(projection_matrix[2][3]))

        row_proj_3 = col_proj.split()
        row_proj_3.label(text=str(projection_matrix[3][0]))
        row_proj_3.label(text=str(projection_matrix[3][1]))
        row_proj_3.label(text=str(projection_matrix[3][2]))
        row_proj_3.label(text=str(projection_matrix[3][3]))

classes = (
    MyAddonProperties,
    RENDER_OT_save_projection_matrix,
    RENDER_PT_gt_generator,
    RENDER_PT_camera_matrix
)


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
