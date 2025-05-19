import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise IOError(f"Unable to read the point cloud file: {file_path}")
    return pcd

def normalize_point_cloud(pcd):

    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points -= centroid  
    max_range = np.max(np.linalg.norm(points, axis=1))
    points /= max_range
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, max_range

def preprocess_point_cloud_with_scale(pcd, voxel_size=0.0):
    """
    Preprocessing of point clouds:
    - Optional voxel downsampling
    - Normalization (returning the scaling factor simultaneously)
    - Normal vector estimation, normalization and direction unification 
    """
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd, scale_factor = normalize_point_cloud(pcd)
    
    # Estimation of normal vector
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(k=30)
    
    # Ensure that the normal vector points outward
    center = pcd.get_center()
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    vectors_to_center = center - points
    inward_mask = (normals * vectors_to_center).sum(axis=1) > 0
    normals[inward_mask] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd, scale_factor
