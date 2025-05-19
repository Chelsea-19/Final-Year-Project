# Final-Year-Project

## -- Kidney Segmentation & 3D Reconstruction with Bayesian Optimization

This repository implements an end-to-end pipeline for kidney mesh reconstruction and volume estimation from NIfTI segmentation masks, integrating point-cloud processing, Poisson surface reconstruction, and automated hyperparameter tuning via Bayesian Optimization.

---

## ⚙️ Features

- **Dataset**  
  – KiTs 2023 challenge: 50 abdominal CT cases, 96 kidney samples -- download from [KiTs23 Dataset](https://github.com/neheller/kits23/tree/main/dataset) (In this work, we use from Case0000 to Case0049, the format is `.nii.gz`)  
  – NIfTI dimensions: `611 × 512 × 512` voxels; labels `0 = background`, `1 = kidney`, `2 = tumor`  

- **Segmentation & Ground-Truth Volume**  
  – Connected-component separation of left/right kidneys  
  – Segmentation success rate: `96%`  
  – Volume error distribution (N = 96): `mean 2.88%`, `median 1.56%`, `std ±3.15%`, `min 0.05%`, `max 17.59%`  
  – `85.7%` of samples error < `5%`, `95%` < `10%`  

- **Point-Cloud Extraction & Processing**  
  – Marching Cubes → high-density PLY point clouds  
  – Statistical outlier removal, voxel downsampling (`1 mm³`), normal estimation & orientation correction  

- **Poisson Surface Reconstruction**  
  – Adaptive `depth ∈ [7,12]`, `scale ∈ [0.8,1.6]` via Bayesian Optimization (average `40` iterations)  
  – Loss function balances volume error, Chamfer distance, Hausdorff distance  

- **Mesh Post-Processing**  
  – Density-based face filtering (`5th–95th` percentiles)  
  – Hole closing with PyMeshLab, recompute normals  

- **Evaluation Metrics**  
  – Chamfer distance: `mean 0.0295 ± 0.0102` (min `0.0178`, max `0.0808`)  
  – Hausdorff distance: `mean 0.1543 ± 0.0716` (min `0.0669`, max `0.3667`)  
  – Pearson correlation: `CD vs. volume error r = 0.41` (p < 0.005), `HD vs. volume error r = 0.34` (p < 0.05)  

- **Performance**  
  – Average processing time per case: `≈15 min` (vs. `30 min` traditional)  

---

## 📥 Installation

### Prerequisites
- **Hardware & OS**  
  Ubuntu 20.04, CUDA 11.8  
  NVIDIA GPU (e.g., TITAN RTX) recommended for accelerated Poisson reconstruction  

### Setup
```bash
# Clone the repository
git clone https://github.com/Chelsea-19/Final-Year-Project.git
cd Final-Year-Project

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```  

---

## 📂 Directory Structure
```text
Final-Year-Project/
├── Main_Process/
│   ├── data_utils.py           # Functions for loading and preprocessing point clouds
│   ├── main.py                 # Main script for processing kidneys and generating results
│   ├── metrics.py              # Functions for computing evaluation metrics
│   ├── poisson_reconstruction.py # Functions for Poisson surface reconstruction
│   ├── seg.py                  # Functions for segmentation and volume calculation
│   └── volume_estimation.py    # Function for estimating the volume of a mesh
├── Resluts_Analysis/
│   ├── stats.py                # Script for statistical analysis and visualization of results
│   └── visualization.py        # Script for visualizing segmentation slices
└── README.md                   # This README file
```

---


## 🤝 Ackownledgement
 - Supervisor: Prof. Fei Ma
 - Dataset: KiTS23 Challenge Organizers
 - Computing Support: Dr. Zixun Lan
 - Open-Source Tools: Open3D, PyMeshLab, BayesianOptimization

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

---

## ✉️ Contact
If you have any questions or feedback, please feel free to contact the project maintainer at [Felix.Liang24atoutlook.com].

