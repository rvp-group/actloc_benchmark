import numpy as np

def ypr_xyz_to_extrinsic(ypr_xyz):
    """
    Convert yaw, pitch, roll, x, y, z to World-to-Camera extrinsic matrices.

    Args:
        ypr_xyz (np.ndarray): (N, 6) array. Each row is [yaw, pitch, roll, x, y, z].

    Returns:
        np.ndarray: (N, 4, 4) Extrinsic matrices (World to Camera).
    """
    N = ypr_xyz.shape[0]
    extrinsics = np.zeros((N, 4, 4), dtype=np.float32)

    for i in range(N):
        yaw, pitch, roll, x, y, z = ypr_xyz[i]

        # Compute rotation matrices
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R_yaw = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [ 0,   0, 1]
        ])

        R_pitch = np.array([
            [ cp, 0, sp],
            [  0, 1,  0],
            [-sp, 0, cp]
        ])

        R_roll = np.array([
            [1,   0,    0],
            [0,  cr, -sr],
            [0,  sr,  cr]
        ])

        # Final rotation: yaw -> pitch -> roll
        R_c2w = R_yaw @ R_pitch @ R_roll  # Camera-to-World rotation

        # apply another rotation to get the z-forward camera convention
        R_c2w = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) @ R_c2w

        t_c2w = np.array([x, y, z], dtype=np.float32)

        # Invert to get World-to-Camera (extrinsic)
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        extrinsics[i] = extrinsic

    return extrinsics

def vps_to_extrinsic(vps, xyzs, x_length=6, y_length=18, interval=20):
    # Convert view points to extrinsic matrix
    extrinsic = []
    for i in range(vps.shape[0]):
        vp = vps[i]
        xyz = xyzs[i]
        roll = -1 * (vp[0] - (x_length // 2)) * interval * np.pi / 180
        pitch = -1 * (vp[1] - (y_length // 2)) * interval * np.pi / 180
        yaw = 0
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        extrinsic.append(ypr_xyz_to_extrinsic(np.array([[yaw, pitch, roll, x, y, z]])))
    extrinsic = np.array(extrinsic)
    return extrinsic


def hybrid_cost_vp_selection(pred_map: np.ndarray, lamda: float = 0.02
              ):
    """
    Greedily select view points from a sequence of probability maps.

    Rules:
      1) For the first section, choose the global argmax over the entire H×W map.
      2) For each subsequent section i, choose argmax of:
            gain = pred_map[i] - lamda * dist
         where dist is manhattan-like with wrap-around on the column (yaw) axis:
            dist(x,y) = |x - x_last| + min(|y - y_last|, W - |y - y_last|)

    Args:
        pred_map: Array with shape (N, H, W) or (H, W) (treated as N=1).
                  Values are probabilities (higher is better).
        lamda:    Distance penalty weight.

    Returns:
        vps:   list of (row, col) indices (length N)
        probs: list of probabilities at those indices (length N)
    """
    pm = np.asarray(pred_map)
    if pm.ndim == 2:
        pm = pm[None, ...]  # (1, H, W)
    assert pm.ndim == 3, "pred_map must have shape (N,H,W) or (H,W)"

    N, H, W = pm.shape

    # ---- first VP: global argmax on the first map ----
    first_flat = int(np.argmax(pm[0]))
    first_r, first_c = divmod(first_flat, W)
    vps = [(first_r, first_c)]
    probs = [float(pm[0, first_r, first_c])]

    # Pre-build grid for vectorized distances
    rows = np.arange(H)[:, None]   # (H,1)
    cols = np.arange(W)[None, :]   # (1,W)

    for i in range(1, N):
        last_r, last_c = vps[-1]

        # distance with wrap on columns (yaw)
        dx = np.abs(rows - last_r)              # (H,1)
        dy_raw = np.abs(cols - last_c)          # (1,W)
        dy = np.minimum(dy_raw, W - dy_raw)     # wrap-around
        dist = dx + dy                           # (H,W)

        gain = pm[i] - lamda * dist
        best_flat = int(np.argmax(gain))
        r, c = divmod(best_flat, W)

        vps.append((r, c))
        probs.append(float(pm[i, r, c]))

    return vps, probs
