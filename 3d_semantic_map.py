import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import csv
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import math

pcd_list = []
pcd_down_list = []
trans_list = []


def depth_image_to_point_cloud_o3d(path_rgb, path_depth, intrinsic):
    if os.path.isfile(path_rgb):
        rgb_raw = o3d.io.read_image(path_rgb)
        depth_raw = o3d.io.read_image(path_depth)
        depth_raw = np.asarray(depth_raw, dtype=np.float32)
        depth_raw = depth_raw/255*10*1000
        depth_raw = o3d.geometry.Image(depth_raw)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_raw, depth_raw, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic)
        # (x, -y, -z)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                      [0, 0, -1, 0], [0, 0, 0, 1]])
        # remove ceiling
        pcd = pcd.select_by_index(
            np.where(np.asarray(pcd.points)[:, 1] < 0.5)[0])
        pcd_list.append(pcd)


def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def custom_voxel_down(pcd, voxel_size):
    voxel_min_bound = pcd.get_min_bound() - voxel_size * 0.5
    voxel_max_bound = pcd.get_max_bound() + voxel_size * 0.5
    pcd_points = np.asarray(pcd.points)

    pcd_index = {}
    for i in range(len(pcd_points)):
        ref_coord = (pcd_points[i] - voxel_min_bound) / voxel_size
        index_x = int(math.floor(ref_coord[0]))
        index_y = int(math.floor(ref_coord[1]))
        index_z = int(math.floor(ref_coord[2]))
        if (index_x,index_y,index_z) not in pcd_index:
            pcd_index[(index_x,index_y,index_z)] = [i]
        else:
            pcd_index[(index_x,index_y,index_z)].append(i)

    pcd_down_points = []
    pcd_down_normals = []
    pcd_down_colors = []
    pcd_down_covariances = []
    # add point and average
    for key in pcd_index.keys():
        tmp_color = []
        tmp_points = np.zeros(3)
        tmp_normals = np.zeros(3)
        tmp_covariances = np.zeros((3,3))
        for i, index in enumerate(pcd_index[key]):
            tmp_points += np.asarray(pcd.points)[index]
            tmp_color.append(np.asarray(pcd.colors)[index])
            if pcd.has_normals():
                tmp_normals += np.asarray(pcd.normals)[index]
            if pcd.has_covariances():
                tmp_covariances += np.asarray(pcd.covariances)[index]

        tmp_points /= len(pcd_index[key])
        pcd_down_points.append(tmp_points)
        if pcd.has_normals():
            tmp_normals /= len(pcd_index[key])
            pcd_down_normals.append(tmp_normals)
        if pcd.has_covariances():
            tmp_covariances /= len(pcd_index[key])
            pcd_down_covariances.append(tmp_covariances)

        color_dict = {}
        for color in tmp_color:
            color = tuple(color.tolist())
            if color not in color_dict:
                color_dict[color] = 1
            else:
                color_dict[color] += 1
        
        max_num = 0
        for key in color_dict:
            if color_dict[key]>max_num:
                max_num = color_dict[key]
                max_color = np.asarray(key)

        pcd_down_colors.append(max_color)
    
    # assign to pcd_down
    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(np.asarray(pcd_down_points))
    pcd_down.colors = o3d.utility.Vector3dVector(np.asarray(pcd_down_colors))
    if pcd.has_normals():
        pcd_down.normals = o3d.utility.Vector3dVector(np.asarray(pcd_down_normals))
    if pcd.has_covariances():
        pcd_down.covariances = o3d.utility.Matrix3dVector(np.asarray(pcd_down_covariances))

    return pcd_down

def preprocess_point_cloud(pcd, voxel_size):
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = custom_voxel_down(pcd, voxel_size)
    
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size*1.5  # 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def local_icp_algorithm_o3d(source, target, init_trans, voxel_size):
    '''
    origin name is refine_registration
    '''
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def best_fit_transform(p, q):
    '''
    reference prove link: https://zhuanlan.zhihu.com/p/107218828?utm_id=0&fbclid=IwAR01v2O0yjiI7-cQI7cNL_Rari89D1m2C_J59AMUed0ivesceEmNRKbBAEM
    reference code link: https://www.796t.com/content/1548524898.html?fbclid=IwAR23KWasUAWxo_If4McOL2vCJZWP8mr_zs0BIP7UbquPS20biyBLWmaIRBY
    p is (N,3)
    q is (M,3)
    '''
    mean_p = np.mean(p, axis=0)  # (N,3)
    mean_q = np.mean(q, axis=0)  # (M,3)
    x = p - mean_p
    y = q - mean_q
    # because tr(ABC)=tr(CAB)=tr(BCA) => tr((Y.T)RX)= tr(RX(Y.T)), make S = X(Y.T)
    # need attention shape
    S = np.dot(x.T, y)

    # do SVD
    U, sigma, VT = np.linalg.svd(S)
    R = np.dot(VT.T, U.T)  # get rotation matrix

    # special reflection case
    if np.linalg.det(R) < 0:
        VT[:, 2] *= -1
        R = np.dot(VT.T, U.T)

    t = mean_q - np.dot(R, mean_p.T).T  # get translate matrix (1,3)

    trans = np.zeros((4, 4))        # homogeneous
    trans[3, 3] = 1
    trans[0:3, 0:3] = R
    trans[0:3, 3] = t

    return trans

if __name__ == "__main__":
    floor = "floor2"
    width = 512
    height = 512
    fx = 256
    fy = 256
    cx = 256
    cy = 256
    depth_scale = 1000  # 1000
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=512, height=512, fx=256, fy=256, cx=256, cy=256)
    voxel_size = 0.05
    trans_o3d = np.identity(4)
    trans_list.append(trans_o3d)

    trans_by_o3d = []
    trans_by_o3d.append(np.identity(4))

    # get length of file in rgb
    DIR = "hw1_data/"+floor+"/images"
    size = len([name for name in os.listdir(DIR) if os.path.isfile(
        os.path.join(DIR, name))])

    print("read file to get pcd")
    for i in tqdm(range(size)):
        path_rgb = "hw1_data/"+floor+"/images/rgb_"+str(i)+".png"
        path_semantic = "hw1_data/"+floor+"/result_semantic_apartment0/semantic_"+str(i)+".png"
        path_depth = "hw1_data/"+floor+"/depth/depth_"+str(i)+".png"

        depth_image_to_point_cloud_o3d(path_semantic, path_depth, intrinsic)

    print("o3d icp to get transformation")
    for i in tqdm(range(size)):
        if i > 0:
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
                pcd_list[i], pcd_list[i-1], voxel_size)
            result_ransac = execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)
            '''
            o3d local icp algorithm
            '''
            result_icp = local_icp_algorithm_o3d(
               source_down, target_down, result_ransac.transformation, voxel_size)
            trans_list.append(result_icp.transformation)

            if i == 1:
                pcd_down_list.append(target_down)
                pcd_down_list.append(source_down)
            else:
                pcd_down_list.append(source_down)

    # add all pcd together
    for i in range(1, len(trans_list)):
        trans_list[i] = trans_list[i-1] @ trans_list[i]

    print("add all pcd together to get final pcd and estimated trajectory")
    for i in tqdm(range(len(trans_list))):
        if i == 0:
            final_pcd = pcd_down_list[i]
            #final_pcd = pcd_list[i]
        else:
            final_pcd = final_pcd + \
                pcd_down_list[i].transform(trans_list[i])
            #final_pcd = final_pcd + \
            #   pcd_list[i].transform(trans_list[i])

    o3d.visualization.draw_geometries([final_pcd])
    print("end")
