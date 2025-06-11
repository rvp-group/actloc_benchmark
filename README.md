# Active Localization Benchmark

Active localization is the task of determining the most informative viewpoints to improve a robot’s pose estimation within a known map. Traditionally, a robot navigating an environment may rely on passive localization—simply pointing its camera forward while moving. However, this ignores the fact that not all viewpoints are equally informative. Consider a scenario where the robot faces a featureless white wall: looking straight ahead in this case offers little to no localization cues.

<!-- <p align="center">
  <img src="assets/forward1.png" width="30%" alt="passloc1"/>
  <img src="assets/forward2.png" width="30%" alt="passloc2"/>
  <img src="assets/forward3.png" width="30%" alt="passloc3"/>
</p> -->

Active localization instead aims to reason about the environment and proactively select viewpoints that maximize perceptual information, improving robustness and accuracy, especially in ambiguous or textureless areas. Heuristics in where to look can be achieved through optimiality criteria and Fisher information matrics, visibility, distribution of landmarks or deep learned techniques. 

<!-- <p align="center">
  <img src="images/placeholder.png" width="30%" alt="Image 1"/>
</p> -->

## How to Actively Localize?
The workflow is simple, you can start from any inputs SfM reconstruction, mesh, semantics image, the only **mandatory** thing are the waypoints, 3D locations (x, y, z) from which you want to provide the "best" viewing angles:

```
input: any <SfM model, mesh, semantics, etc..> + 3D waypoints (mandatory)
         ↓
    inference.py
         ↓
    best viewing angles (we assume for each 3D location you produce an orientation in form of quaternion)
```

This repository provides an end-to-end pipeline for predicting optimal camera viewing directions at given waypoints in 3D scenes and capturing images at those predicted viewpoints.

## Environment Setup

- **Clone the repo:**
   ```bash
   git clone --recursive https://github.com/rvp-group/actloc_benchmark
   ```

- **Create the conda environment:**
   ```bash
   conda create -n actloc_benchmark  python=3.11 && conda activate actloc_benchmark
   ```
- **Install:**
   ```bash
   pip install -r requirements.txt
   ```

### Download some sample data

Download sample data:
```bash
chmod +x download_sample_data.sh && ./download_sample_data.sh
```

## Overview
The pipeline consists of four main components:

1. **`inference.py`** - Predicts best viewing angles for each waypoint (this is where you should put your hands for the challenge),
here you need to output the orientations. 
2. **`capture_images_at_best_viewing_directions.py`** - Captures images at the predicted best orientations, this is required to evaluate the localization accuracy.
3. **`match_and_localize.py`** - Given poses and their corresponding images from step (1) and (2) you want to match and localize against the SfM model. For this purpose we employ [hloc](https://github.com/cvg/Hierarchical-Localization). For each image localized you will have an error.
4. **`evaluate.py`** - Given the errors from localization, this outputs the F1 score according to [learning-where-to-look](https://link.springer.com/chapter/10.1007/978-3-031-73016-0_12).

We now explain how each module work and what are I/O

### To work with a specific scene, navigate to its data folder:
```bash
cd actloc_benchmark/example_data/00005-yPKGKBCyYx8/
```


### `inference.py`
Predict the best viewing angles for your waypoints (here is where you should put your hands) for now the best angles heuristics is simply based on maximizing the visibility of 3d landmarks. The scripts output full estimate pose (waypoint + orientation) in COLMAP style:
```bash
python ../../inference.py \ 
    --waypoints-file sampled_waypoints.txt \ 
    --sfm-dir scene_reconstruction \ 
    --output-estimate estimate/estimate_poses.txt
```

### `capture_images_at_best_viewing_directions.py`
Capture images corresponding to previously estimated poses:
```bash
python ../../capture_images_at_best_viewing_directions.py \
    --pose-file ../00005-yPKGKBCyYx8/estimate/estimate_poses.txt \ 
    --mesh-file yPKGKBCyYx8.glb \ 
    --output-folder estimate/images
```

### `match_and_localize.py`
```bash
python ../../match_and_localize.py \
    --sfm_model_path scene_reconstruction \
    --ref_images_path scene_reconstruction/images \  
    --ref_features_fn scene_reconstruction/sfm_features.h5 \  
    --query_images_path estimate/images \
    --poses_fn estimate/estimate_poses.txt \ 
    --output_path estimate
```

### `evaluate.py`
TODO


<!-- ## Example Data Download
You can download the example data from [here](https://drive.google.com/drive/folders/1BunuI_wIVeL1oZ1zWxxAfu7HSop7uMMi?usp=sharing) and put it in the root folder of this repo to run the pipeline.

## Quick Start (with the example data)

### Step 1: Run Inference

Predict the best viewing angles for your waypoints:

```bash
python inference.py \
    --sfm-dir ./example_data/00005_reference_sfm \
    --waypoints-file ./example_data/sampled_viewpoints.txt \
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
- **Model Checkpoint**: (optional) if you use learning-based method

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
- `--output-angles`: Output file for best viewing angles

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


## How to Run Your Own Method:

TBA

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
``` -->