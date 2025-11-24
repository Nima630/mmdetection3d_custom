# extract_layers.py

import os
import torch
import numpy as np
from pathlib import Path
from nuscenes.utils.data_classes import LidarPointCloud
from mmengine.registry import init_default_scope
from mmdet3d.apis import init_model

# ==================================================
# CONFIG
# ==================================================
LIDAR_DIR = Path("/home/draiman/Desktop/Nima/Lavis_blip2/lavis/New_Model/dataset_processing/dataset")
OUT_DIR   = Path("/home/draiman/Desktop/Nima/Lavis_blip2/lavis/New_Model/dataset_processing/features_CP_layers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = "/home/draiman/Desktop/Nima/mmdetection3d_CP/configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CKPT   = "/home/draiman/Desktop/Nima/mmdetection3d_CP/checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth"

# ==================================================
# INIT
# ==================================================
init_default_scope("mmdet3d")
model = init_model(CONFIG, CKPT, device="cuda:0")
model.eval()


def main():

    lidar_files = ["n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin"]

    for fname in lidar_files:
        lidar_path = LIDAR_DIR / fname

        # -----------------------------
        # Load LiDAR
        # -----------------------------
        pc = LidarPointCloud.from_file(str(lidar_path))
        pts = pc.points.T  # [N,4]

        if pts.shape[1] == 4:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1))])

        pts = torch.tensor(pts, dtype=torch.float32).cuda()

        with torch.no_grad():

            # -----------------------------------------
            # 1. Voxel Encoder (early feature)
            # -----------------------------------------
            voxels, coors, num_points = model.data_preprocessor.voxel_layer(pts)
            voxel_feats = model.pts_voxel_encoder(voxels, num_points, coors)

            # -----------------------------------------
            # 2. Middle Encoder → "spatial"
            # -----------------------------------------
            batch_index = torch.zeros((coors.shape[0], 1), device=coors.device, dtype=torch.long)
            coors_batched = torch.cat([batch_index, coors], dim=1)

            spatial = model.pts_middle_encoder(voxel_feats, coors_batched, batch_size=1)

            # -----------------------------------------
            # 3. Backbone → C1 and C2
            # -----------------------------------------
            backbone_feats = model.pts_backbone(spatial)

            bev_early = backbone_feats[0]   # C1 (128 ch)
            bev_deep  = backbone_feats[1]   # C2 (256 ch)

        # -----------------------------------------
        # SAVE (for Step 1)
        # -----------------------------------------
        save_path = OUT_DIR / "benign_early_layers.pt"

        torch.save({
            "voxel_feats": voxel_feats.cpu(),
            "spatial": spatial.cpu(),
            "bev_c1": bev_early.cpu(),
            "bev_c2": bev_deep.cpu(),  # optional
        }, save_path)

        print(f"Saved layers → {save_path}")

if __name__ == "__main__":
    main()
