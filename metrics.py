import numpy as np
import open3d as o3d

def compute_chamfer_distance(pcd1, pcd2):
    """
    Calculate the Chamfer distance between two point clouds
    - symmetric average nearest neighbor distance
    """
    pcd1_points = np.asarray(pcd1.points)
    pcd2_points = np.asarray(pcd2.points)
    
    tree_pcd2 = o3d.geometry.KDTreeFlann(pcd2)
    sum_dist1 = 0.0
    for p in pcd1_points:
        [_, _, dists] = tree_pcd2.search_knn_vector_3d(p, 1)
        sum_dist1 += np.sqrt(dists[0])
    avg_dist1 = sum_dist1 / len(pcd1_points)
    
    tree_pcd1 = o3d.geometry.KDTreeFlann(pcd1)
    sum_dist2 = 0.0
    for p in pcd2_points:
        [_, _, dists] = tree_pcd1.search_knn_vector_3d(p, 1)
        sum_dist2 += np.sqrt(dists[0])
    avg_dist2 = sum_dist2 / len(pcd2_points)
    
    return avg_dist1 + avg_dist2

def compute_hausdorff_distance(pcd1, pcd2):
    """
    Calculate the Hausdorff distance between two point clouds 
    - symmetric maximum nearest neighbor distance
    """
    pcd1_points = np.asarray(pcd1.points)
    pcd2_points = np.asarray(pcd2.points)
    
    tree_pcd2 = o3d.geometry.KDTreeFlann(pcd2)
    max_dist1 = 0.0
    for p in pcd1_points:
        [_, _, dists] = tree_pcd2.search_knn_vector_3d(p, 1)
        max_dist1 = max(max_dist1, np.sqrt(dists[0]))
    
    tree_pcd1 = o3d.geometry.KDTreeFlann(pcd1)
    max_dist2 = 0.0
    for p in pcd2_points:
        [_, _, dists] = tree_pcd1.search_knn_vector_3d(p, 1)
        max_dist2 = max(max_dist2, np.sqrt(dists[0]))
    
    return max(max_dist1, max_dist2)