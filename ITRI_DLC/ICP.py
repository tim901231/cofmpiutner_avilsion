import os, sys, argparse, csv, copy
import numpy as np
import open3d as o3d
from glob import glob
from scipy.spatial.transform import Rotation as R


def ICP(source, target, threshold, init_pose, iteration=100):
    # implement iterative closet point and return transformation matrix
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration)
    )
    # print(reg_p2p)
    # print(reg_p2p.transformation)
    return reg_p2p.transformation

def csv_reader(filename):
    # read csv file into numpy array
    data = np.loadtxt(filename, delimiter=',')
    return data

def numpy2pcd(arr):
    # turn numpy array into open3d point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd


# if __name__ == '__main__':
def exec_ICP(path_name):
    # /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710720_970174987
    # paths = []
    # path_name = '../ITRI_dataset/seq1/dataset/1681710720_970174987'

    # Target point cloud
    target = csv_reader(f'{path_name}/sub_map.csv')
    target_pcd = numpy2pcd(target)
    # o3d.visualization.draw_geometries([target_pcd])
    # Source point cloud
    #TODO: Read your point cloud here#
    source = csv_reader(f'{path_name}/predict.csv')
    source_pcd = numpy2pcd(source)
    # print(source_pcd)
    # o3d.visualization.draw_geometries([source_pcd])

    # Initial pose
    init_pose = csv_reader(f'{path_name}/initial_pose.csv')

    # print(target.shape)

    
    # print(init_pose)
    shift = init_pose[:3, 3]
    target = target - shift
    arr = init_pose[:3, :3]
    arr = np.linalg.inv(arr)
    # print(arr.shape)
    # print(target.shape)
    target = np.matmul(arr, target.T)

    # r = R.from_quat([-0.5070558775462676, 0.47615311808704197, -0.4812773544166568, 0.5334272708696808])
    # r = r.inv()
    # target = r.apply(target.T).T

    # print("pred.csv should be", target)
    np.savetxt('target.csv', target.T, delimiter = ',')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target.T)

    meshframe = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size = 3, origin = [0, 0 , 0]
    )
    # o3d.visualization.draw_geometries([pcd, meshframe])

    
    # Implement ICP
    transformation = ICP(source_pcd, target_pcd, threshold=1, init_pose=init_pose)
    pred_x = transformation[0,3]
    pred_y = transformation[1,3]

    return pred_x, pred_y
    # print(pred_x, pred_y)

