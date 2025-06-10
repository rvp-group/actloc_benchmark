import numpy as np
import open3d as o3d


def create_cylinder_between_points(
    point1, point2, radius=0.01, resolution=20, color=(0, 0, 0)
):
    """
    Create a cylinder between two 3D points.
    return: An Open3D TriangleMesh representing the cylinder.
    """
    # Compute the vector between the two points
    vec = point2 - point1
    length = np.linalg.norm(vec)
    if length == 0:
        return None

    # Create a cylinder oriented along the z-axis
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution
    )
    cylinder.paint_uniform_color(color)
    cylinder.compute_vertex_normals()

    # Compute rotation to align the cylinder with the vector
    z_axis = np.array([0, 0, 1])  # default cylinder direction
    axis = np.cross(z_axis, vec)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0:
        axis = axis / axis_len
        angle = np.arccos(np.dot(z_axis, vec) / length)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(R, center=(0, 0, 0))

    # Translate the cylinder to its correct position
    midpoint = (point1 + point2) / 2
    cylinder.translate(midpoint)

    return cylinder


def camera_vis_with_cylinders(
    cam_in_world: np.ndarray,
    wh_ratio: float = 4.0 / 3.0,
    scale: float = 1.0,
    fovx: float = 90.0,
    color=(1, 1, 0),  # default yellow color
    radius=0.01,
    return_mesh=False,
):
    """
    A camera frustum for visualization using cylinders with a yellow-to-red color gradient based on weight.
    return: A list of Open3D geometries representing the camera frustum.
    """
    if cam_in_world.shape != (4, 4):
        raise ValueError(f"Transform Matrix must be 4x4, but got {cam_in_world.shape}")

    # Compute frustum points
    pw = np.tan(np.deg2rad(fovx / 2.0)) * scale
    ph = pw / wh_ratio
    all_points = np.array(
        [
            [0.0, 0.0, 0.0],  # Frustum apex
            [pw, ph, scale],  # Top right
            [pw, -ph, scale],  # Bottom right
            [-pw, ph, scale],  # Top left
            [-pw, -ph, scale],  # Bottom left
        ]
    )

    # Define frustum edges by connecting points
    line_indices = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],  # Apex to corners
            [1, 2],
            [1, 3],
            [3, 4],
            [2, 4],  # Frustum base edges
        ]
    )

    # Create cylinders for each line segment in the frustum
    cylinders = []
    for start_idx, end_idx in line_indices:
        start_point = all_points[start_idx]
        end_point = all_points[end_idx]

        # Create the cylinder with the same color
        cylinder = create_cylinder_between_points(
            start_point, end_point, radius=radius, color=color
        )
        cylinders.append(cylinder)

    # Apply the transformation to all cylinders
    for cylinder in cylinders:
        cylinder.transform(cam_in_world)

    if return_mesh:
        m = cylinders[0]
        for c in cylinders[1:]:
            m += c
        return m
    return cylinders
