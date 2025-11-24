
















# #!/usr/bin/env python

# import os
# import torch
# import numpy as np
# from pathlib import Path
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud
# from mmengine.registry import init_default_scope
# from mmdet3d.apis import init_model


# # ==================================================
# # USER CONFIG
# # ==================================================
# # LIDAR_DIR = Path("/home/draiman/Desktop/Datasets_nuscenes/testing/benign_raw/LIDAR_TOP")
# LIDAR_DIR = Path("/home/draiman/Desktop/Datasets_nuscenes/testing/attacks_raw/lidar/drop")
# # LIDAR_DIR = Path("/home/draiman/Desktop/Datasets_nuscenes/testing/attacks_raw/lidar/shift")

# # Output dirs
# # OUT_DIR   = Path("/home/draiman/Desktop/Datasets_nuscenes/testing/features_benign/lidar")
# OUT_DIR = Path("/home/draiman/Desktop/Datasets_nuscenes/testing/features_lidar_drop")
# # OUT_DIR = Path("/home/draiman/Desktop/Datasets_nuscenes/testing/features_lidar_shift")

# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # CenterPoint config + checkpoint
# CONFIG = "/home/draiman/Desktop/Nima/mmdetection3d_CP/configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
# CKPT   = "/home/draiman/Desktop/Nima/mmdetection3d_CP/checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth"

# # NuScenes pseudo-test split
# NUSC_ROOT = "/home/draiman/Desktop/datasets/nuscenes_pseudo_test"


# # ==================================================
# # INIT MODEL + NuScenes
# # ==================================================
# init_default_scope("mmdet3d")
# model = init_model(CONFIG, CKPT, device="cuda:0")
# model.eval()

# nusc = NuScenes(version="v1.0-pseudo-test", dataroot=NUSC_ROOT, verbose=False)


# # ==================================================
# # MAIN
# # ==================================================
# def main():
#     lidar_files = sorted(f for f in os.listdir(LIDAR_DIR) if f.endswith(".pcd.bin"))

#     for idx, fname in enumerate(lidar_files):
#         lidar_path = LIDAR_DIR / fname

#         # Extract timestamp
#         try:
#             timestamp = int(fname.split("__")[-1].replace(".pcd.bin", ""))
#         except ValueError:
#             print(f"⚠️ Invalid timestamp in {fname}")
#             continue

#         # Match to sample_token
#         sample_token = None
#         for sd in nusc.sample_data:
#             if sd["channel"] == "LIDAR_TOP" and sd["timestamp"] == timestamp:
#                 sample_token = sd["sample_token"]
#                 break

#         if not sample_token:
#             print(f"⚠️ Sample token not found for {fname}")
#             continue

#         # Load point cloud
#         try:
#             pc = LidarPointCloud.from_file(str(lidar_path))
#         except Exception as e:
#             print(f"⚠️ Failed to load {fname}: {e}")
#             continue

#         pts = pc.points.T  # [N, 4]

#         # Pad to 5 dims (required by CP voxel layer)
#         if pts.shape[1] == 4:
#             pts = np.hstack([pts, np.zeros((pts.shape[0], 1))])

#         pts = torch.tensor(pts, dtype=torch.float32).cuda()

#         # --------------------------------------------------
#         # VG / Voxel Encoder / Middle Encoder / Backbone
#         # --------------------------------------------------
#         with torch.no_grad():
#             voxels, coors, num_points = model.data_preprocessor.voxel_layer(pts)

#             if voxels.numel() == 0:
#                 print(f"⚠️ No voxels for {fname}, skipping")
#                 continue

#             voxel_feats = model.pts_voxel_encoder(voxels, num_points, coors)

#             batch_index = torch.zeros((coors.shape[0], 1), device=coors.device, dtype=torch.long)
#             coors_batched = torch.cat([batch_index, coors], dim=1)

#             spatial = model.pts_middle_encoder(voxel_feats, coors_batched, batch_size=1)
#             backbone_feats = model.pts_backbone(spatial)

#         bev_feat = backbone_feats[1].cpu()  # [1, 256, H, W]

#         # --------------------------------------------------
#         # SAVE (TRAINING STYLE)
#         # --------------------------------------------------
#         save_path = OUT_DIR / f"{sample_token}.pt"
#         torch.save(
#             {
#                 "feat": bev_feat,
#                 "meta": {
#                     "sample_token": sample_token
#                 }
#             },
#             save_path
#         )

#         print(f"[{idx+1}/{len(lidar_files)}] Saved → {save_path}")


# if __name__ == "__main__":
#     main()




















































# #!/usr/bin/env python

# import os
# import torch
# import numpy as np
# from pathlib import Path
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud
# from mmengine.registry import init_default_scope
# from mmdet3d.apis import init_model


# # ==================================================
# # USER CONFIG
# # ==================================================
# LIDAR_DIR = Path("/home/draiman/Desktop/Nima/Lavis_blip2/lavis/New_Model/dataset_processing/dataset")
# # lidar_sample = "/home/draiman/Desktop/Nima/Lavis_blip2/lavis/New_Model/dataset_processing/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin.pcd.bin"

# # Output dirs
# OUT_DIR = Path("/home/draiman/Desktop/Nima/Lavis_blip2/lavis/New_Model/dataset_processing/features")

# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # CenterPoint config + checkpoint
# CONFIG = "/home/draiman/Desktop/Nima/mmdetection3d_CP/configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
# CKPT   = "/home/draiman/Desktop/Nima/mmdetection3d_CP/checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth"

# # NuScenes pseudo-test split
# NUSC_ROOT = "/home/draiman/Desktop/datasets/nuscenes_pseudo_test"


# # ==================================================
# # INIT MODEL + NuScenes
# # ==================================================
# init_default_scope("mmdet3d")
# model = init_model(CONFIG, CKPT, device="cuda:0")
# model.eval()

# nusc = NuScenes(version="v1.0-pseudo-test", dataroot=NUSC_ROOT, verbose=False)


# # ==================================================
# # MAIN
# # ==================================================
# def main():
#     # lidar_files = sorted(f for f in os.listdir(LIDAR_DIR) if f.endswith(".pcd.bin"))
#     lidar_files = ["n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin.pcd.bin"]

#     for idx, fname in enumerate(lidar_files):
#         lidar_path = LIDAR_DIR / fname

#         # Extract timestamp
#         try:
#             timestamp = int(fname.split("__")[-1].replace(".pcd.bin", ""))
#         except ValueError:
#             print(f"⚠️ Invalid timestamp in {fname}")
#             continue

#         # Match to sample_token
#         sample_token = None
#         for sd in nusc.sample_data:
#             if sd["channel"] == "LIDAR_TOP" and sd["timestamp"] == timestamp:
#                 sample_token = sd["sample_token"]
#                 break

#         if not sample_token:
#             print(f"⚠️ Sample token not found for {fname}")
#             continue

#         # Load point cloud
#         try:
#             pc = LidarPointCloud.from_file(str(lidar_path))
#         except Exception as e:
#             print(f"⚠️ Failed to load {fname}: {e}")
#             continue

#         pts = pc.points.T  # [N, 4]

#         # Pad to 5 dims (required by CP voxel layer)
#         if pts.shape[1] == 4:
#             pts = np.hstack([pts, np.zeros((pts.shape[0], 1))])

#         pts = torch.tensor(pts, dtype=torch.float32).cuda()

#         # --------------------------------------------------
#         # VG / Voxel Encoder / Middle Encoder / Backbone
#         # --------------------------------------------------
#         with torch.no_grad():
#             voxels, coors, num_points = model.data_preprocessor.voxel_layer(pts)

#             if voxels.numel() == 0:
#                 print(f"⚠️ No voxels for {fname}, skipping")
#                 continue

#             voxel_feats = model.pts_voxel_encoder(voxels, num_points, coors)

#             batch_index = torch.zeros((coors.shape[0], 1), device=coors.device, dtype=torch.long)
#             coors_batched = torch.cat([batch_index, coors], dim=1)

#             spatial = model.pts_middle_encoder(voxel_feats, coors_batched, batch_size=1)
#             backbone_feats = model.pts_backbone(spatial)

#         bev_feat = backbone_feats[1].cpu()  # [1, 256, H, W]

#         # --------------------------------------------------
#         # SAVE (TRAINING STYLE)
#         # --------------------------------------------------
#         save_path = OUT_DIR / f"{sample_token}.pt"
#         torch.save(
#             {
#                 "feat": bev_feat,
#                 "meta": {
#                     "sample_token": sample_token
#                 }
#             },
#             save_path
#         )

#         print(f"[{idx+1}/{len(lidar_files)}] Saved → {save_path}")


# if __name__ == "__main__":
#     main()






















# file_1.py

import os
import torch
import numpy as np
from pathlib import Path
from nuscenes.utils.data_classes import LidarPointCloud
from mmengine.registry import init_default_scope
from mmdet3d.apis import init_model

# ==================================================
# USER CONFIG
# ==================================================
LIDAR_DIR = Path("/home/draiman/Desktop/Nima/Lavis_blip2/lavis/New_Model/dataset_processing/dataset")

# Output dirs
OUT_DIR = Path("/home/draiman/Desktop/Nima/Lavis_blip2/lavis/New_Model/dataset_processing/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CenterPoint config + checkpoint
CONFIG = "/home/draiman/Desktop/Nima/mmdetection3d_CP/configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CKPT   = "/home/draiman/Desktop/Nima/mmdetection3d_CP/checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth"

# ==================================================
# INIT MODEL
# ==================================================
init_default_scope("mmdet3d")
model = init_model(CONFIG, CKPT, device="cuda:0")
model.eval()

# ==================================================
# MAIN
# ==================================================
def main():
    # Process only one specific file
    # lidar_files = ["n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin"]
    lidar_files = ["synthetic_n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin"]

    for idx, fname in enumerate(lidar_files):
        lidar_path = LIDAR_DIR / fname

        # Load point cloud
        try:
            pc = LidarPointCloud.from_file(str(lidar_path))
        except Exception as e:
            print(f"⚠️ Failed to load {fname}: {e}")
            continue

        pts = pc.points.T  # [N, 4]

        # Pad to 5 dims (required by CP voxel layer)
        if pts.shape[1] == 4:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1))])

        pts = torch.tensor(pts, dtype=torch.float32).cuda()

        # --------------------------------------------------
        # VG / Voxel Encoder / Middle Encoder / Backbone
        # --------------------------------------------------
        with torch.no_grad():
            voxels, coors, num_points = model.data_preprocessor.voxel_layer(pts)

            if voxels.numel() == 0:
                print(f"⚠️ No voxels for {fname}, skipping")
                continue

            voxel_feats = model.pts_voxel_encoder(voxels, num_points, coors)

            batch_index = torch.zeros((coors.shape[0], 1), device=coors.device, dtype=torch.long)
            coors_batched = torch.cat([batch_index, coors], dim=1)

            spatial = model.pts_middle_encoder(voxel_feats, coors_batched, batch_size=1)
            backbone_feats = model.pts_backbone(spatial)

        bev_feat = backbone_feats[1].cpu()  # [1, 256, H, W]

        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------
        # save_path = OUT_DIR / "file_1.pt"
        save_path = OUT_DIR / "file_synth.pt"
        torch.save({"feat": bev_feat}, save_path)
        print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()