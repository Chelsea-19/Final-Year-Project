import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

seg_path = '/home/visitor/KiTs 23/Case000/segmentation.nii' 
output_path = '/home/visitor/3DGS/output.png' 

# Loading segmentation doc
sg_img = nib.load(seg_path)
seg_data = seg_img.get_fdata()
seg_data = seg_data.astype(np.uint8)

# Basic information
print("Dimension（Z, Y, X）:", seg_data.shape)
print("Unique label value:", np.unique(seg_data))

# Display multiple slices
def show_slices(data, axis='z', num_slices=6, save_path='output.png'):
    assert axis in ['x', 'y', 'z']
    fig, axs = plt.subplots(1, num_slices, figsize=(15, 5))
    fig.suptitle(f"Segmentation slices along {axis}-axis", fontsize=16)

    total_slices = data.shape['zyx'.index(axis)]
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)

    for i, idx in enumerate(indices):
        if axis == 'z':
            img = data[idx, :, :]
        elif axis == 'y':
            img = data[:, idx, :]
        elif axis == 'x':
            img = data[:, :, idx]
        axs[i].imshow(img, cmap='nipy_spectral')
        axs[i].set_title(f'{axis.upper()} = {idx}')
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Image saved: {save_path}")

# Save result
show_slices(seg_data, axis='z', num_slices=6, save_path=output_path)
