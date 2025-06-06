# ActLoc Best Single Viewpoint Selection Pipeline

This repository provides an end-to-end pipeline for predicting optimal camera viewing directions at given waypoints in 3D scenes and capturing images at those predicted viewpoints.

## Environment Setup

1. **Create the conda environment from the provided environment file:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate flash_env
   ```

## Overview

The pipeline consists of two main components:

1. **`inference.py`** - Predicts best viewing angles for each waypoint using ActLoc Model
2. **`capture_images_at_best_viewing_directions.py`** - Captures images at the predicted optimal viewpoints

## Pipeline Workflow

```
Input: SfM Reconstruction + Waypoints
         ↓
    inference.py
         ↓
    Best Viewing Angles
         ↓
capture_images_at_best_viewing_directions.py
         ↓
    Output: Images at Optimal Viewpoints
```

## Example Data Download
You can download the example data from [here](https://drive.google.com/drive/folders/1BunuI_wIVeL1oZ1zWxxAfu7HSop7uMMi?usp=sharing) and put it in the root folder of this repo to run the pipeline.

## Quick Start (with the example data)

### Step 1: Run Inference

Predict the best viewing angles for your waypoints:

```bash
python inference.py \
    --sfm-dir ./example_data/00005_reference_sfm \
    --waypoints-file ./example_data/sampled_viewpoints.txt \
    --checkpoint ./checkpoints/actloc_binary_best.pth \
    --output-angles ./example_data/best_viewing_angles.txt
```

### Step 2: Capture Images

Capture images at the predicted optimal viewpoints:

```bash
python capture_images_at_best_viewing_directions.py \
    --mesh-file ./example_data/yPKGKBCyYx8.glb \
    --waypoints-file ./example_data/sampled_viewpoints.txt \
    --angles-file ./example_data/best_viewing_angles.txt \
    --output-folder ./example_data/best_viewpoint_images
```

## Input Requirements

### For Inference (`inference.py`)

- **SfM Reconstruction**: COLMAP reconstruction folder containing:
  - `cameras.bin/txt` - Camera intrinsics
  - `images.bin/txt` - Camera poses and image information
  - `points3D.bin/txt` - 3D point cloud with colors
- **Waypoints File**: Text file with 3D coordinates (N×3 format)
  ```
  x1 y1 z1
  x2 y2 z2
  ...
  ```
- **Model Checkpoint**: Trained ActLoc model weights

### For Image Capture (`capture_images_at_best_viewing_directions.py`)

- **Mesh File**: 3D scene mesh (`.glb`, etc.)
- **Waypoints File**: Same waypoints used for inference
- **Angles File**: Output from inference step (N×2 format)
  ```
  x_angle1 y_angle1
  x_angle2 y_angle2
  ...
  ```

## Camera Coordinate System & Rotation Convention

⚠️ **Important**: This pipeline uses a specific camera coordinate system that differs from common conventions.

### Coordinate System
- **X-axis rotation (elevation)**: Positive angles rotate the camera **downward**
- **Y-axis rotation (azimuth)**: Positive angles rotate the camera **leftward**

### Rotation Order
Rotations are applied in this specific order:
1. **Y-axis rotation first** (azimuth/horizontal rotation)
2. **X-axis rotation second** (elevation/vertical rotation)

This is implemented as:
```python
rotation_x = R.from_euler('x', np.deg2rad(x_angle)).as_matrix()
rotation_y = R.from_euler('y', np.deg2rad(y_angle)).as_matrix()
combined_rotation = rotation_x @ rotation_y @ initial_rotation
```

### Angle Discretization
The model predicts viewing directions on a discrete grid:
- **X-axis**: 6 intervals covering [-60°, +40°] with 20° steps
- **Y-axis**: 18 intervals covering [-180°, +160°] with 20° steps

## Detailed Usage

### Inference Script (`inference.py`)

```bash
python inference.py [OPTIONS]
```

**Key Arguments:**
- `--sfm-dir`: Path to COLMAP SfM reconstruction folder
- `--waypoints-file`: Path to waypoints text file
- `--checkpoint`: Path to trained model checkpoint [Only the ActLoc-Bin weights are provided for now]
- `--output-angles`: Output file for best viewing angles
- `--num-classes`: Number of model output classes (2 or 4)
- `--error-threshold`: Reprojection error threshold for point filtering (default: 0.5)

**Output:**
- Text file with best viewing angles for each waypoint
- Console output showing prediction grids and probabilities

### Image Capture Script (`capture_images_at_best_viewing_directions.py`)

```bash
python capture_images_at_best_viewing_directions.py [OPTIONS]
```

**Key Arguments:**
- `--mesh-file`: Path to 3D scene mesh
- `--waypoints-file`: Path to waypoints text file
- `--angles-file`: Path to predicted viewing angles
- `--output-folder`: Output directory for captured images

**Output:**
- Images named: `waypoint_00001_x20_y-160.jpg`
- `best_viewpoints_info.txt`: Camera information summary

## Example Results

After running the pipeline, you'll have:
```
example_data/
├── best_viewing_angles.txt          # Predicted angles
└── best_viewpoint_images/
    ├── waypoint_00001_x20_y-160.jpg # Captured images
    ├── waypoint_00002_x0_y40.jpg
    ├── ...
    └── best_viewpoints_info.txt     # Camera metadata
```

## Full Dataset 
### Download
You can download the full dataset, including both training and test data, from [here](https://drive.google.com/drive/folders/1vsMV2CI144ihui4oJHrykwCkq0xKWJrJ?usp=sharing). After downloading, you can put the two zip files into a folder called `full_actloc_data` under the root folder and then unzip the two files there to obtain the `training_data` and `test_data` subfolders.

### Data Folder Structure Explanatio
The `training_data` and `test_data` folders follow the same layout. Below is an example of how the `training_data` folder is organized:

```
training_data
├── raw_images
│   ├── <scene_1>
│   │   ├── scene_reconstruction
│   │   │   ├── images
│   │   │   ├── img_nm_to_colmap_cam.txt
│   │   │   └── img_name_to_colmap_Tcw.txt
│   │   ├── waypoint_1
│   │   │   ├── images
│   │   │   ├── img_nm_to_colmap_cam.txt
│   │   │   └── img_name_to_colmap_Tcw.txt
│   │   ├── waypoint_2
│   │   │   └── …
│   │   ├── <scene_1>.glb
│   │   ├── sampled_viewpoints.txt
│   │   └── ...
│   ├── <scene_2>
│   │   └── ...
│   └── ...
└── sfm_and_localization_results
    ├── <scene_1>
    │   ├── scene_reconstruction
    │   │   ├── cameras.bin
    │   │   ├── images.bin
    │   │   └── points3D.bin
    │   ├── waypoint_1
    │   │   ├── scene_reconstruction
    │   │   │   ├── results.txt
    │   │   │   └── pose_errors.txt
    │   ├── waypoint_2
    │   │   └── …
    │   └── ...
    ├── <scene_2>
    │   └── ...
    └── ...
```

#### raw_images

This folder contains all the data needed for Structure-from-Motion (SfM) and for setting up visual localization. Each subfolder under `raw_images` represents a single scene. Inside each scene folder, you will find:

- `scene_reconstruction/`
  - **images/**  
    A set of images used to run SfM on this scene.
  - **img_nm_to_colmap_cam.txt**  
    Camera intrinsics for each image, stored in the COLMAP output format.
  - **img_name_to_colmap_Tcw.txt**  
    World-to-camera poses (ground truth) for the SfM images, also in COLMAP output format.

- `waypoint_i/` (where `i` is a number starting from 1)
  - **images/**  
    A set of images captured from different viewpoints at this waypoint. These images are used for visual localization.
  - **img_nm_to_colmap_cam.txt**  
    Camera intrinsics for the waypoint images, in COLMAP output format.
  - **img_name_to_colmap_Tcw.txt**  
    Ground truth poses for the waypoint images, in COLMAP output format.

- **<scene_name>.glb**  A 3D mesh of the scene stored in GLB format. The filename matches the name of the scene folder.

- **sampled_viewpoints.txt**  A file that contains the world coordinates of all the sampled waypoints used in the scene.

#### sfm_and_localization_results

This folder holds the output of SfM and the results of visual localization. Each subfolder under `sfm_and_localization_results` corresponds to one scene. Inside each scene folder, you will find:

- `scene_reconstruction/`
  - **cameras.bin**  
    Binary file containing camera intrinsics for the sparse reconstruction (COLMAP format).
  - **images.bin**  
    Binary file containing image poses for the sparse reconstruction (COLMAP format).
  - **points3D.bin**  
    Binary file containing 3D point cloud data for the sparse reconstruction (COLMAP format).

- `waypoint_i/scene_reconstruction` (where `i` is a number starting from 1)
  - **results.txt**  
    The visual localization results for each image captured at this waypoint. Each line typically includes the estimated pose for a given image.
  - **pose_errors.txt**  
    The error (difference) between each estimated pose in `results.txt` and the ground truth pose from `raw_images`. Each line usually shows the translation and rotation error for one image.

### Sample Data Parsing Script
A sample data parsing script is given in the `sample_data_parser.py`, which can help you to better understand the data. It can be easily tested with the command `python sample_data_parser.py` after having `test_data` in the `full_actloc_data` folder.

### Evaluation
To evaluate on a specific scene or a folder of scenes (e.g. `training_data` and `test_data`)，you can run the following commands:
```
# Evaluate on scene 00005-yPKGKBCyYx8
python evaluation_script.py --scenes "00005-yPKGKBCyYx8"

# Evaluate on scene 00005-yPKGKBCyYx8 with sparsification
python evaluation_script.py --scenes "00005-yPKGKBCyYx8" --enable-sparsification

# Evaluate on the test scenes by default
python evaluation_script.py 

# Evaluate on the test scenes by default with sparsification
python evaluation_script.py --enable-sparsification
```