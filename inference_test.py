# import os
# import torch
# import numpy as np
# from mmengine import Config
# from mmengine.registry import init_default_scope
# from mmdet3d.apis import init_model
# from nuscenes.utils.data_classes import LidarPointCloud

# # --- Config & Model Init ---
# config_file = 'configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
# checkpoint_file = 'checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth'

# init_default_scope('mmdet3d')
# model = init_model(config_file, checkpoint_file, device='cuda:0')
# model.eval()

# # --- Input & Output paths ---
# lidar_dir = '/home/draiman/Desktop/Nima/BEVFormer/data/nuscenes/samples/LIDAR_TOP'
# output_dir = '/data/features_bevformer/lidar'
# os.makedirs(output_dir, exist_ok=True)

# # --- Loop over all LiDAR files ---
# lidar_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith('.pcd.bin'))

# for idx, lidar_filename in enumerate(lidar_files):
#     frame_id = lidar_filename.replace('.pcd.bin', '')
#     lidar_path = os.path.join(lidar_dir, lidar_filename)

#     try:
#         pc = LidarPointCloud.from_file(lidar_path)
#     except Exception as e:
#         print(f"⚠️ Skipping {lidar_filename} due to error: {e}")
#         continue

#     points = pc.points.T
#     if points.shape[1] == 4:
#         points = np.hstack([points, np.zeros((points.shape[0], 1))])  # Pad to 5D
#     points = points[:20000]
#     points_tensor = torch.tensor(points, dtype=torch.float32).cuda()

#     with torch.no_grad():
#         voxels, coors, num_points = model.data_preprocessor.voxel_layer(points_tensor)
#         voxel_feats = model.pts_voxel_encoder(voxels, num_points, coors)
#         if len(voxel_feats.shape) == 3:
#             voxel_feats = voxel_feats.mean(dim=1)

#         batch_indices = torch.zeros((coors.shape[0], 1), dtype=torch.int, device=coors.device)
#         coors_with_batch = torch.cat([batch_indices, coors], dim=1)

#         spatial_feats = model.pts_middle_encoder(voxel_feats, coors_with_batch, batch_size=1)
#         backbone_feats = model.pts_backbone(spatial_feats)

#     selected_feat = backbone_feats[1]  # [1, 256, 90, 90]
#     save_path = os.path.join(output_dir, f'{frame_id}.pt')
#     torch.save({'feat': selected_feat.cpu(), 'meta': {'frame_id': frame_id}}, save_path)
#     print(f"[{idx + 1}/{len(lidar_files)}] ✅ Saved: {save_path}")





































# # import os
# # import torch
# # import numpy as np
# # from mmengine import Config
# # from mmengine.registry import init_default_scope
# # from mmdet3d.apis import init_model
# # from nuscenes.utils.data_classes import LidarPointCloud

# # # --- Config & Paths ---
# # config_file = 'configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
# # checkpoint_file = 'checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth'
# # lidar_filename = 'n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151061547455.pcd.bin'
# # lidar_path = f'/home/draiman/Desktop/Nima/BEVFormer/data/nuscenes/samples/LIDAR_TOP/{lidar_filename}'

# # # Extract frame ID for naming
# # frame_id = lidar_filename.replace('.pcd.bin', '')  # cleaner than splitext for .pcd.bin

# # # Output base directory (same as BEVFormer camera features)
# # base_dir = '/data/features_bevformer/lidar'
# # os.makedirs(base_dir, exist_ok=True)

# # # --- Load model ---
# # init_default_scope('mmdet3d')
# # model = init_model(config_file, checkpoint_file, device='cuda:0')
# # model.eval()

# # # --- Load and preprocess point cloud ---
# # pc = LidarPointCloud.from_file(lidar_path)
# # points = pc.points.T  # Shape: (N, 5)
# # print("Loaded point cloud shape:", points.shape)

# # if points.shape[1] == 4:
# #     points = np.hstack([points, np.zeros((points.shape[0], 1))])  # Pad to 5D

# # MAX_POINTS = 20000
# # points = points[:MAX_POINTS]
# # points_tensor = torch.tensor(points, dtype=torch.float32).cuda()

# # # --- Inference ---
# # with torch.no_grad():
# #     voxels, coors, num_points = model.data_preprocessor.voxel_layer(points_tensor)
# #     voxel_feats = model.pts_voxel_encoder(voxels, num_points, coors)
# #     if len(voxel_feats.shape) == 3:
# #         voxel_feats = voxel_feats.mean(dim=1)

# #     batch_indices = torch.zeros((coors.shape[0], 1), dtype=torch.int, device=coors.device)
# #     coors_with_batch = torch.cat([batch_indices, coors], dim=1)

# #     spatial_feats = model.pts_middle_encoder(voxel_feats, coors_with_batch, batch_size=1)
# #     backbone_feats = model.pts_backbone(spatial_feats)

# # # --- Save selected feature map ---
# # selected_feat = backbone_feats[1]  # [1, 256, 90, 90]
# # print("Selected feature map shape:", selected_feat.shape)

# # save_path = os.path.join(base_dir, f'{frame_id}.pt')
# # torch.save({'feat': selected_feat.cpu(), 'meta': {'frame_id': frame_id}}, save_path)
# # print(f"✅ Saved LiDAR features to {save_path}")



import os
import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from mmengine.registry import init_default_scope
from mmdet3d.apis import init_model

# --- Config & Model Init ---
config_file = 'configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
checkpoint_file = 'checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth'

init_default_scope('mmdet3d')
model = init_model(config_file, checkpoint_file, device='cuda:0')
model.eval()

# --- Paths ---
lidar_dir = '/home/draiman/Desktop/Nima/BEVFormer/data/nuscenes/samples/LIDAR_TOP'
output_dir = '/data/features_bevformer/lidar'
os.makedirs(output_dir, exist_ok=True)

# --- NuScenes Metadata ---
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/draiman/Desktop/Nima/BEVFormer/data/nuscenes')

# --- Process LiDAR Files ---
lidar_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith('.pcd.bin'))

for idx, lidar_filename in enumerate(lidar_files):
    frame_id = lidar_filename.replace('.pcd.bin', '')
    timestamp_str = frame_id.split('__')[-1]
    try:
        timestamp = int(timestamp_str)
    except ValueError:
        print(f"⚠️ Invalid timestamp in {lidar_filename}")
        continue

    # Match timestamp to sample_token
    sample_token = None
    for sd in nusc.sample_data:
        if sd['channel'] == 'LIDAR_TOP' and sd['timestamp'] == timestamp:
            sample_token = sd['sample_token']
            break

    if not sample_token:
        print(f"⚠️ Could not find sample_token for {lidar_filename}")
        continue

    lidar_path = os.path.join(lidar_dir, lidar_filename)
    try:
        pc = LidarPointCloud.from_file(lidar_path)
    except Exception as e:
        print(f"⚠️ Skipping {lidar_filename}: {e}")
        continue

    # Process point cloud
    points = pc.points.T
    if points.shape[1] == 4:
        points = np.hstack([points, np.zeros((points.shape[0], 1))])  # Pad to 5D
    points = points[:20000]
    points_tensor = torch.tensor(points, dtype=torch.float32).cuda()

    # Model forward
    with torch.no_grad():
        voxels, coors, num_points = model.data_preprocessor.voxel_layer(points_tensor)
        voxel_feats = model.pts_voxel_encoder(voxels, num_points, coors)
        if voxel_feats.ndim == 3:
            voxel_feats = voxel_feats.mean(dim=1)
        batch_indices = torch.zeros((coors.shape[0], 1), dtype=torch.int, device=coors.device)
        coors_with_batch = torch.cat([batch_indices, coors], dim=1)
        spatial_feats = model.pts_middle_encoder(voxel_feats, coors_with_batch, batch_size=1)
        backbone_feats = model.pts_backbone(spatial_feats)

    # Save feature
    selected_feat = backbone_feats[1]  # [1, 256, 90, 90]
    save_path = os.path.join(output_dir, f'{sample_token}.pt')
    torch.save({'feat': selected_feat.cpu(), 'meta': {'sample_token': sample_token}}, save_path)
    print(f"[{idx + 1}/{len(lidar_files)}] ✅ Saved: {save_path}")






