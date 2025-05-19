import numpy as np
import open3d as o3d

def poisson_reconstruction(pcd, depth=9, width=0, scale=1.1, linear_fit=False):
    print(f"Poisson Parameters: depth={depth}, width={width}, scale={scale}, linear_fit={linear_fit}")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=width,
        scale=scale,
        linear_fit=linear_fit
    )

    # Convert to a NumPy array and check if it is empty
    densities = np.asarray(densities)
    if densities.size == 0:
        print("Warning: densities 数组为空")
        return mesh

    # Remove the abnormal areas using the 5% - 95% percentile range of the density
    lower_bound = np.quantile(densities, 0.05)
    upper_bound = np.quantile(densities, 0.95)
    valid_mask = (densities >= lower_bound) & (densities <= upper_bound)
    mesh.remove_vertices_by_mask(~valid_mask)

    # Remove floating debris
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Re-calculate the normal vector
    mesh.compute_vertex_normals()
    return mesh

def mesh_to_point_cloud(mesh, number_of_points=10000):
    pcd = mesh.sample_points_poisson_disk(number_of_points)
    return pcd

