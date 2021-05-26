import numpy as np


data = np.load('0001.npz')
#print(data.files)

path_img = '0001.png'
path_img_next = '0002.png'

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

if 'object_poses' in data.files:
    obj_poses = data['object_poses']
    print('\tObject poses:')
    for obj in obj_poses:
        obj_name = obj['name']
        obj_mat  = obj['pose']
        print(obj_name)
        print(obj_mat)
        # Get 2d pixel coordinate of object
        if obj_name != 'Light' and obj_name != 'Camera':
            if ('intrinsic_mat' in data.files) and ('extrinsic_mat' in data.files):
                point_3d = obj_mat[:,3]
                point_3d_cam = np.matmul(extrinsic_mat, point_3d)
                point_2d_scaled = np.matmul(intrinsic_mat, point_3d_cam)
                if point_2d_scaled[2] != 0:
                    point_2d = point_2d_scaled / point_2d_scaled[2]
                    u, v = point_2d[:2]
                    print(' 2D image projection u:{} v:{}'.format(u, v))

try:
    import cv2 as cv


    if 'optical_flow' in data.files:
        opt_flow = data['optical_flow']

        img = cv.imread(path_img)
        img_next = cv.imread(path_img_next)
        # Alpha blending images (so that we can see both at the same time)
        ## The next frame will appear like a ghost after the current frame
        dst = cv.addWeighted(img, 0.75, img_next, 0.25, 0)

        # Draw optical flow - with arrows
        gap_pixels = 15
        rows, cols = dst.shape[:2]
        arrows = np.zeros_like(dst)
        for v in range(gap_pixels, rows, gap_pixels):
            for u in range(gap_pixels, cols, gap_pixels):
                flow_tmp = opt_flow[v, u]
                pt1 = (u, v)
                pt2 = (u + int(round(flow_tmp[0])), v + int(round(flow_tmp[1])))
                cv.arrowedLine(arrows,
                               pt1=pt1,
                               pt2=pt2,
                               color=(0, 255, 0),
                               thickness=1, 
                               tipLength=.03)
        dst = cv.addWeighted(dst, 1.00, arrows, 0.25, 0)
        cv.imshow('Optical Flow: From current to next - arrows', dst)

    if 'normal_map' in data.files:
        normals = data['normal_map']
        cv.imshow("Surface normals", normals)

    if 'segmentation_masks' in data.files:
        # we only have segmentation masks if at least 1 object's pass_index != 0
        sg_msk = data['segmentation_masks']
        sg_msk_inds = data['segmentation_masks_indexes']
        #print(sg_msk_inds)
        # You can also access the individual fields using:
        #print(sg_msk_inds['name'])
        #print(sg_msk_inds['pass_index'])

        # Get a unique color for each of the indexes
        inds = sg_msk_inds['pass_index']
        inds = inds[inds != 0] # remove zeros
        inds = np.unique(inds) # remove repeated (returns the sorted unique)
        # Distribute the `inds` to values between 0 and 255
        n_inds = len(inds)
        cmap_vals =  np.linspace(0., 1., n_inds)
        cmap_vals = cmap_vals * 255.

        height, width = sg_msk.shape
        sg_msk_img = np.zeros((height, width, 3), np.uint8)
        for counter, ind in enumerate(inds):
            sg_msk_img[sg_msk == ind] = cmap_vals[counter]

        sg_msk_img = cv.applyColorMap(sg_msk_img, cv.COLORMAP_RAINBOW)
        sg_msk_img[sg_msk == 0] = [0, 0, 0]
        cv.imshow("Segmentation masks", sg_msk_img)

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
        normalized_depth = cv.normalize(depth_copy, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC1) # alternatively use CV_8UC3
        #"""
        #normalized_depth = 255.0 - normalized_depth # invert values for draw
        depth_colored = cv.applyColorMap(normalized_depth, cv.COLORMAP_JET)
        depth_colored[depth == INVALID_DEPTH] = [0, 0, 0] # paint in black the regions with invalid depth
        cv.imshow("Depth map", depth_colored)

    cv.waitKey(0)
except ImportError:
    print("\"opencv-python\" not found, please install to visualize the rest of the results.")
