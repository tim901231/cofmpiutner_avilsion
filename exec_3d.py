import numpy as np
import csv
import yaml
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R

def p23(P_plus, corner):
	#to homogeneous 
	x = np.array([corner[0], corner[1], 1])
	#X = P_plus * x + lambda * C 
	X = np.matmul(P_plus, x) 
	
	# X = X/X[1] * 1.63
	X[3] = 1
	return X[0:3]

# test2
# /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710720_960889218
# /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710720_966775996
# /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710720_970174987
# /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710720_993855093

def exec_3d(path):
    # path = 'ITRI_dataset/seq1/dataset/1681710720_993855093'
    csvfile = open(path + r'/camera.csv', 'r')
    camera = list(csv.reader(csvfile,  delimiter=','))[0][0]
    camera = camera.split('/')[2]

    camera_info = f'ITRI_dataset/camera_info/lucid_cameras_x00/{camera}_camera_info.yaml'
    # print(camera)

    with open(camera_info, 'r') as stream:  
        camera_d=yaml.safe_load(stream)

    # w = camera_d['image_width']
    # h = camera_d['image_height']
    # K = np.array(camera_d['camera_matrix']['data']).reshape(3, 3)
    # D = np.array(camera_d['distortion_coefficients']['data']).reshape(4)
    # R = np.array(camera_d['rectification_matrix']['data']).reshape(3, 3)
    P = np.array(camera_d['projection_matrix']['data']).reshape(3, 4)
    # C = np.array([0, 0, 0, 1])
    # P_undist = np.hstack((cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D,(w, h) ,R), [[0], [0], [0]]))
    # print(P)

    P_plus = np.linalg.pinv(P)
    # print(P_plus)

    rotations_r =  [-0.0806252, 0.607127, 0.0356452, 0.789699]
    fr2f_shift = [0.559084, 0.0287952, -0.0950537]
    fr2f_rotation = R.from_quat(rotations_r)

    rotations_l =  [-0.117199, -0.575476, -0.0686302, 0.806462]
    fl2f_shift = [-0.564697, 0.0402756, -0.028059]
    fl2f_rotation = R.from_quat(rotations_l)

    rotations_b =  [0.074732, -0.794, -0.10595, 0.59393]
    b2fl_shift = [-1.2446, 0.21365, -0.91917]
    b2fl_rotation = R.from_quat(rotations_b)

    r = R.from_quat([-0.5070558775462676, 0.47615311808704197, -0.4812773544166568, 0.5334272708696808])
    r_inv = r.inv()

    corners = np.load(path + '/corners.npy').astype(np.float32)
    #corners = cv2.fisheye.undistortPoints(corners.reshape(1, -1, 2), K, D, R = R, P = P)[0]
    #print(corners)
    all = []
    for corner in corners:
        # print (corner)



        X = p23(P_plus, corner)

        # print("before", X)
        # print(camera)
        if camera == 'gige_100_fr_hdr':
            X = X + fr2f_shift
            X = fr2f_rotation.apply(X)
        elif camera == 'gige_100_fl_hdr':
            X = X + fl2f_shift
            X = fl2f_rotation.apply(X)
        elif camera == 'gige_100_b_hdr':
            X = X + b2fl_shift
            X = b2fl_rotation.apply(X)

            X = X + fl2f_shift
            X = fl2f_rotation.apply(X)
            

        # print("after", X)

        X = r.apply(X)
        X = X / X[2] * (-1.63)
        # print("result", X)
        # print(X)
        

        all.append(X)
    all = np.array(all)
    # a1 = []
    # for a in all:
    #     a1.append([a[0], a[2], a[1]])
    # all = np.array(a1)
    # print(all.shape)
    np.save(path +'/predict.npy', all)
    pcd = o3d.geometry.PointCloud()

    # pcd.points = o3d.utility.Vector3dVector(all)


    # o3d.visualization.draw_geometries([pcd])

    # meshframe = o3d.geometry.TriangleMesh.create_coordinate_frame(
    # size = 2, origin = [0, 0 , 0]
    # )
    # o3d.visualization.draw_geometries([pcd, meshframe])
