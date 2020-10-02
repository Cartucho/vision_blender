import numpy as np

data = np.load('0001.npz')
#print(data.files)

if 'intrinsic_mat' in data.files:
    intrinsic_mat = data['intrinsic_mat']
    print("\tCamera intrinsic mat:\n{}\n".format(intrinsic_mat))
    """
    f_x = intrinsic_mat[0, 0]
    f_y = intrinsic_mat[1, 1]
    c_x = intrinsic_mat[0, 2]
    c_y = intrinsic_mat[1, 2]
    print('f_x:{} f_y:{} c_x:{} c_y:{}'.format(f_x, f_y, c_x, c_y))
    """

if 'extrinsic_mat' in data.files:
    extrinsic_mat = data['extrinsic_mat']
    print("\tCamera extrinsic mat:\n{}\n".format(extrinsic_mat))

if 'object_pose_labels' in data.files and 'object_pose_mats' in data.files:
    obj_pose_labels = data['object_pose_labels']
    obj_pose_mats = data['object_pose_mats']
    obj_poses = [{'obj_name': i, 'obj_pose_mat': j} for i, j in zip(obj_pose_labels, obj_pose_mats)]
    print('\tObject poses:')
    for obj in obj_poses:
        print(obj['obj_name'])
        print(obj['obj_pose_mat'])
        """ Get 2d coordinate of object """
        if 'intrinsic_mat' in data.files:
            t_x, t_y, t_z = obj['obj_pose_mat'][:,3]
            #print('t_x:[{}] t_y:[{}] t_z:[{}]'.format(t_x, t_y, t_z))
            point_3d = np.array([[t_x],
                                 [t_y],
                                 [t_z]])
            #print(point_3d)
            point_2d_scaled = np.matmul(intrinsic_mat, point_3d)
            point_2d = point_2d_scaled / point_2d_scaled[2]
            u, v = point_2d[:2]
            print('u:{} v:{}'.format(u, v))
try:
    import cv2

    if 'optical_flow' in data.files:
        opt_flow = data['optical_flow']
        cv2.imshow("Optical flow", opt_flow)

    if 'normal_map' in data.files:
        normals = data['normal_map']
        cv2.imshow("Surface normals", normals)

    if 'segmentation_masks' in data.files:
        sg_msk = data['segmentation_masks']
        height, width = sg_msk.shape
        sg_msk_img = np.zeros((height, width, 3), np.uint8)
        sg_msk_img[sg_msk == 1] = [255, 0, 0] # Draw in blue where `obj_ind = 1`
        cv2.imshow("Segmentation masks", sg_msk_img)

    if 'depth_map' in data.files:
        depth = data['depth_map']
        INVALID_DEPTH = -1
        depth_min = np.amin(depth[depth != INVALID_DEPTH])
        """ option 1 """
        #"""
        depth_max = np.amax(depth) # if you have multiple images you can feed the min and max over all the images, to get a consistent looking depthmap
        normalized_depth = (depth - depth_min)/(depth_max - depth_min) * 255.0
        normalized_depth = normalized_depth.astype(np.uint8)
        #"""
        """ option 2 """
        """
        depth_copy = np.copy(depth)
        depth_copy[depth == INVALID_DEPTH] = depth_min
        normalized_depth = cv2.normalize(depth_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) # alternatively use CV_8UC3
        #"""
        #normalized_depth = 255.0 - normalized_depth # invert values for draw
        depth_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        depth_colored[depth == INVALID_DEPTH] = [0, 0, 0] # paint in black the regions with invalid depth
        cv2.imshow("Depth map", depth_colored)

    cv2.waitKey(0)
except ImportError:
    print("\"opencv-python\" not found, please install to visualize the rest of the results.")
