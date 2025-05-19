import os
import tempfile
import pandas as pd
import open3d as o3d
import pymeshlab
from numpy.random import uniform
from bayes_opt import BayesianOptimization
from data_utils import load_point_cloud, preprocess_point_cloud_with_scale
from poisson_reconstruction import poisson_reconstruction
from volume_estimation import estimate_volume
from metrics import compute_chamfer_distance, compute_hausdorff_distance
from seg import generate_GT_if_not_exists

def close_holes_with_meshlab(mesh, max_hole_size=20000):
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_input_file:
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_output_file:
            o3d.io.write_triangle_mesh(temp_input_file.name, mesh)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_input_file.name)
            ms.meshing_close_holes(maxholesize=max_hole_size)
            ms.meshing_remove_unreferenced_vertices()
            ms.save_current_mesh(temp_output_file.name)
            mesh_closed = o3d.io.read_triangle_mesh(temp_output_file.name)
    os.remove(temp_input_file.name)
    os.remove(temp_output_file.name)
    return mesh_closed

def process_single_kidney(pcd_path, output_prefix, gt_volume):
    pcd = load_point_cloud(pcd_path)
    pcd = pcd.voxel_down_sample(voxel_size=1.0)
    pcd, scale_factor = preprocess_point_cloud_with_scale(pcd)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(pcd.points) < 3000:
        print(f"The point cloud is too sparse. The current number of points is: {len(pcd.points)}. Adjusting downsampling strategy...")
        pcd = pcd.uniform_down_sample(1)

    history = []

    def objective(depth, scale):
        depth = int(round(depth))
        scale = float(scale)
        try:
            mesh = poisson_reconstruction(pcd, depth=depth, scale=scale)
            if not mesh.has_triangles():
                return -9999
            mesh.paint_uniform_color([0.5, 0.5, 0.5])
            mesh = mesh.filter_smooth_simple(number_of_iterations=5)
            volume_norm = estimate_volume(mesh)
            volume_real = volume_norm * (scale_factor ** 3)
            mesh_sampled = mesh.sample_points_poisson_disk(10000)
            chamfer = compute_chamfer_distance(pcd, mesh_sampled)
            hausdorff = compute_hausdorff_distance(pcd, mesh_sampled)
            volume_diff = abs(volume_real - gt_volume) / gt_volume
            loss = 0.85 * volume_diff + 0.15 * (chamfer + hausdorff)
            history.append({"depth": depth, "scale": scale, "volume_diff": volume_diff, "chamfer": chamfer, "hausdorff": hausdorff, "loss": loss})
            return -loss
        except Exception as e:
            print(f"Fail to parameters combination: {e}")
            return -9999

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={"depth": (7, 12), "scale": (0.8, 1.6)},
        random_state=42, verbose=0
    )

    tried = set()
    while len(tried) < 40:
        depth = round(uniform(7, 12))
        scale = round(uniform(0.8, 1.6), 3)
        key = (depth, scale)
        if key in tried:
            continue
        tried.add(key)
        optimizer.probe(params={"depth": depth, "scale": scale}, lazy=False)

    optimizer.maximize(init_points=0, n_iter=0)
    best = optimizer.max["params"]
    best_depth = int(round(best["depth"]))
    best_scale = best["scale"]

    mesh = poisson_reconstruction(pcd, depth=best_depth, scale=best_scale)
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    mesh = close_holes_with_meshlab(mesh)

    volume_norm = estimate_volume(mesh)
    volume_estimated = volume_norm * (scale_factor ** 3)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5]) 
    o3d.io.write_triangle_mesh(f"{output_prefix}_reconstructed.ply", mesh)

    mesh_sampled = mesh.sample_points_poisson_disk(10000)
    chamfer = compute_chamfer_distance(pcd, mesh_sampled)
    hausdorff = compute_hausdorff_distance(pcd, mesh_sampled)
    volume_diff_ratio = abs(volume_estimated - gt_volume) / gt_volume * 100

    final_result = {
        "Estimation": round(volume_estimated, 2),
        "Chamfer": round(chamfer, 6),
        "Hausdorff": round(hausdorff, 6),
        "depth": best_depth,
        "scale": round(best_scale, 3),
        "比例": f"{volume_diff_ratio:.2f}%"
    }

    if volume_diff_ratio > 10:
        print("Initiate volume priority remediation optimization...")

        def volume_only_objective(depth, scale):
            depth = int(round(depth))
            scale = float(scale)
            try:
                mesh = poisson_reconstruction(pcd, depth=depth, scale=scale)
                if not mesh.has_triangles():
                    return -9999
                mesh = mesh.filter_smooth_simple(number_of_iterations=5)
                mesh = close_holes_with_meshlab(mesh)
                volume_norm = estimate_volume(mesh)
                volume_real = volume_norm * (scale_factor ** 3)
                volume_diff = abs(volume_real - gt_volume) / gt_volume
                print(f"[VOLUME ONLY] depth={depth}, scale={scale:.3f}, volume_diff={volume_diff*100:.2f}%")
                return -volume_diff
            except Exception as e:
                print(f"Fail to fallback: {e}")
                return -9999

        optimizer = BayesianOptimization(
            f=volume_only_objective,
            pbounds={"depth": (7, 12), "scale": (0.8, 1.6)},
            random_state=99, verbose=0
        )
        optimizer.maximize(init_points=5, n_iter=10)
        best = optimizer.max["params"]
        best_depth = int(round(best["depth"]))
        best_scale = best["scale"]

        mesh = poisson_reconstruction(pcd, depth=best_depth, scale=best_scale)
        mesh = mesh.filter_smooth_simple(number_of_iterations=5)
        mesh = close_holes_with_meshlab(mesh)
        volume_norm = estimate_volume(mesh)
        volume_estimated = volume_norm * (scale_factor ** 3)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5]) 
        o3d.io.write_triangle_mesh(f"{output_prefix}_reconstructed.ply", mesh)
        mesh_sampled = mesh.sample_points_poisson_disk(10000)
        chamfer = compute_chamfer_distance(pcd, mesh_sampled)
        hausdorff = compute_hausdorff_distance(pcd, mesh_sampled)
        volume_diff_ratio = abs(volume_estimated - gt_volume) / gt_volume * 100

        final_result = {
            "Estimation": round(volume_estimated, 2),
            "Chamfer": round(chamfer, 6),
            "Hausdorff": round(hausdorff, 6),
            "depth": best_depth,
            "scale": round(best_scale, 3),
            "ratio": f"{volume_diff_ratio:.2f}%"
        }

    pd.DataFrame(history).to_csv(f"{output_prefix}_optimization_history.csv", index=False)
    return final_result

def main():
    base_dir = "/home/visitor/Case"
    generate_GT_if_not_exists(base_dir)

    gt_table = pd.read_csv(os.path.join(base_dir, "volume_GT.csv"))
    gt_dict = dict(zip(gt_table["ID"], gt_table["Volume"]))

    results = []

    for case in sorted(os.listdir(base_dir)):
        case_path = os.path.join(base_dir, case)
        if not os.path.isdir(case_path):
            continue

        print(f"\n Processing {case} ...")
        for side in ["left", "right"]:
            organ_id = f"{case}_{'L' if side == 'left' else 'R'}"
            if organ_id not in gt_dict:
                print(f"No GT data:{organ_id}")
                continue

            ply_path = os.path.join(case_path, f"{side}_kidney.ply")
            if not os.path.exists(ply_path):
                print(f"Lack of point cloud doc:{ply_path}")
                continue

            output_prefix = os.path.join(case_path, f"{side}")
            result = process_single_kidney(ply_path, output_prefix, gt_dict[organ_id])

            if result:
                row = {
                    "ID": organ_id,
                    "Volume": gt_dict[organ_id],
                    **result
                }
                results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(base_dir, "volume_results.csv"), index=False)
    print("All results saved in volume_results.csv")

if __name__ == "__main__":
    main()