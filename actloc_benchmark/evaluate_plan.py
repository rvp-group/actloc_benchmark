import argparse
import numpy as np


def qconj(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z])

def qmul(a, b):
    aw,ax,ay,az = a; bw,bx,by,bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ])

def relative_angle_axis(q1, q2):
    # check if q1 and q2 are unit quaternions
    assert np.isclose(np.linalg.norm(q1), 1.0), "q1 is not a unit quaternion"
    assert np.isclose(np.linalg.norm(q2), 1.0), "q2 is not a unit quaternion"
    qrel = qmul(q2, qconj(q1))
    w,x,y,z = qrel
    vnorm = np.linalg.norm([x,y,z])
    angle = 2*np.arctan2(vnorm, np.clip(w, -1.0, 1.0))
    if vnorm < 1e-12:
        axis = np.array([1.0,0.0,0.0])  # arbitrary when angle≈0
    else:
        axis = np.array([x,y,z]) / vnorm

    if angle > np.pi:
        angle = 2*np.pi - angle
        axis = -axis

    return np.degrees(angle), axis

def load_error_data(filepath):
    """
    Load translation and rotation errors from a file.
    Each line should have: <image_name> <translation_error> <rotation_error>
    """
    errors = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            _, t_err, r_err = parts
            errors.append((float(t_err), float(r_err)))
    return np.array(errors)  # shape (N, 2)

def load_selected_poses(filepath):
    """
    Load selected pose indices from a file.
    Each line should have: <image_name> <qw, qx, qy, qz, tx, ty, tz> 
    skip lines starting with #
    """
    selected_pose = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            # only add quaternion and translation, ignore image name
            _, qw, qx, qy, qz, tx, ty, tz = parts
            selected_pose.append([float(qw), float(qx), float(qy), float(qz), float(tx), float(ty), float(tz)])
    return np.array(selected_pose)  # shape (N, 7)

def calculate_rotations(poses):
    # for each pose, calcuate it rotations from last poses, in degrees
    rotations = []
    # frist set it to 0
    rotations.append(0.0)
    for i in range(1, poses.shape[0]):
        q1 = poses[i-1, :4]
        q2 = poses[i, :4]
        angle, _ = relative_angle_axis(q1, q2)
        rotations.append(angle)
    return np.array(rotations)  # shape (N,)

def evaluate_thresholds(errors, rotations, thresholds):
    """
    Count how many errors are within each (translation, rotation, rotation_from_last) threshold.
    """
    total = errors.shape[0]
    results = []

    for t_thresh, r_thresh, rr_thresh in thresholds:
        count = np.sum((errors[:, 0] <= t_thresh) & (errors[:, 1] <= r_thresh) & (rotations <= rr_thresh))
        percentage = (count / total) * 100
        results.append((t_thresh, r_thresh, rr_thresh, count, percentage))  
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pose errors in visual localization."
    )
    parser.add_argument(
        "--error-file",
        type=str,
        required=True,
        help="Path to file containing translation and rotation errors.",
    )
    parser.add_argument(
        "--selected-pose-file",
        type=str,
        required=True,
        help="Path to file containing selected pose indices.",
    )
    args = parser.parse_args()
    thresholds = [(0.25, 2.0, 30)] # translation in meters, rotation in degrees, rotation from last pose in degrees

    errors = load_error_data(args.error_file)
    selected_poses = load_selected_poses(args.selected_pose_file)

    assert errors.shape[0] == selected_poses.shape[0], "Number of error entries must match number of selected poses"

    rotations = calculate_rotations(selected_poses)

    results = evaluate_thresholds(errors, rotations, thresholds)

    print("\nEvaluation Summary:")
    for t_thresh, r_thresh, rr_thresh, count, percent in results:
        print(
            f"≤ {t_thresh:.2f}m / {r_thresh:.1f}° / {rr_thresh:.1f}°: {count}/{len(errors)} predictions ({percent:.2f}%)"
        )


if __name__ == "__main__":
    main()
