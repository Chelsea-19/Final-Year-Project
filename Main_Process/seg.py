import os
import nibabel as nib
import numpy as np
import open3d as o3d
import pandas as pd
from skimage import measure
from scipy.ndimage import label, center_of_mass


def load_segmentation(path):
    nii = nib.load(path)
    mask = nii.get_fdata()
    spacing = nii.header.get_zooms()
    return mask, spacing


def compute_volume(mask, spacing, target_label=1):
    voxel_volume = np.prod(spacing)
    region_voxels = np.sum(mask == target_label)
    return region_voxels * voxel_volume


def mask_to_surface_mesh(mask, spacing, target_label=1):
    binary_mask = (mask == target_label).astype(np.uint8)
    verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5, spacing=spacing)
    return verts, faces


def split_kidneys_by_connected_components(mask, target_label=1):
    binary_mask = (mask == target_label).astype(np.uint8)
    labeled_mask, num_features = label(binary_mask)

    if num_features < 2:
        raise ValueError("Only one connected area was detected. This may indicate that the kidney is not separated or is missing.")

    component_sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
    largest_indices = np.argsort(component_sizes)[-2:]
    idx1, idx2 = largest_indices + 1

    com1 = center_of_mass(labeled_mask == idx1)
    com2 = center_of_mass(labeled_mask == idx2)

    if com1[2] < com2[2]:
        left_id, right_id = idx1, idx2
    else:
        left_id, right_id = idx2, idx1

    left_mask = np.zeros_like(mask)
    right_mask = np.zeros_like(mask)
    left_mask[labeled_mask == left_id] = target_label
    right_mask[labeled_mask == right_id] = target_label

    return left_mask, right_mask


def process_all_cases(root_dir, output_summary=True):
    summary = []

    for case in sorted(os.listdir(root_dir)):
        case_path = os.path.join(root_dir, case)
        if not os.path.isdir(case_path):
            continue

        nii_files = [f for f in os.listdir(case_path) if f.endswith('.nii') or f.endswith('.nii.gz')]
        if not nii_files:
            print(f"No NIfTI file was found in {case}, skipped")
            continue

        nii_path = os.path.join(case_path, nii_files[0])
        print(f"Processing {case} - Doc: {nii_path}")

        try:
            mask, spacing = load_segmentation(nii_path)
            left_mask, right_mask = split_kidneys_by_connected_components(mask, target_label=1)

            left_volume = compute_volume(left_mask, spacing)
            right_volume = compute_volume(right_mask, spacing)

            # Extract and save the surface mesh points as a point cloud
            for side, organ_mask in zip(["left", "right"], [left_mask, right_mask]):
                verts, _ = mask_to_surface_mesh(organ_mask, spacing)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(verts)
                o3d.io.write_point_cloud(os.path.join(case_path, f"{side}_kidney.ply"), pcd)

            print(f"{case} Segmentation completed: Left kidney {left_volume:.2f}, Right kidney {right_volume:.2f}")
            summary.append({
                "ID": f"{case}_L", "Volume": round(left_volume, 2)
            })
            summary.append({
                "ID": f"{case}_R", "Volume": round(right_volume, 2)
            })

        except Exception as e:
            print(f"Fail to process {case}: {str(e)}")

    if output_summary and summary:
        pd.DataFrame(summary).to_csv(os.path.join(root_dir, "volume_GT.csv"), index=False)


def generate_GT_if_not_exists(root_dir):
    gt_path = os.path.join(root_dir, "volume_GT.csv")
    if not os.path.exists(gt_path):
        print("Not found volume_GT.csv，generating Ground Truth ...")
        process_all_cases(root_dir)
    else:
        print("Find out volume_GT.csv，skipped")