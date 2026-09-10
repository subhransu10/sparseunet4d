import os
import numpy as np
import matplotlib.pyplot as plt

SEQ = "/media/suba/Expansion/4D_Sparse_Conv_Ntwrk/my_kitti_dataset/dataset/sequences/08"
VELODYNE_DIR = os.path.join(SEQ, "velodyne")
LABEL_DIR = os.path.join(SEQ, "labels")

OUT_DIR = os.path.join(SEQ, "gt_mos_png_3d")
os.makedirs(OUT_DIR, exist_ok=True)

FRAMES = ["000046", "000075", "000081"]

MOVING_IDS = {
    252,  # moving-car
    253,  # moving-bicyclist
    254,  # moving-person
    255,  # moving-motorcyclist
    256,  # moving-on-rails
    257,  # moving-bus
    258,  # moving-truck
    259,  # moving-other-vehicle
}

for frame in FRAMES:
    scan_path = os.path.join(VELODYNE_DIR, frame + ".bin")
    label_path = os.path.join(LABEL_DIR, frame + ".label")

    points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(label_path, dtype=np.uint32)

    semantic = labels & 0xFFFF
    moving = np.isin(semantic, list(MOVING_IDS))

    xyz = points[:, :3]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Static points
    ax.scatter(
        xyz[~moving, 0],
        xyz[~moving, 1],
        xyz[~moving, 2],
        s=0.35,
        c="0.55",
        alpha=0.55,
        linewidths=0
    )

    # Moving points
    ax.scatter(
        xyz[moving, 0],
        xyz[moving, 1],
        xyz[moving, 2],
        s=0.55,
        c="red",
        alpha=1.0,
        linewidths=0
    )

    # Better publication view
    ax.view_init(elev=20, azim=-105)

    # Crop around useful scene region
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_zlim(-3, 5)

    # Reduce distorted 3D proportions
    ax.set_box_aspect((1, 1, 0.25))

    ax.set_axis_off()

    plt.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1
    )

    out_path = os.path.join(
        OUT_DIR,
        f"{frame}_gt_mos_3d.png"
    )

    plt.savefig(
        out_path,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    print(f"Saved: {out_path}")