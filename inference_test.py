# import os
# import torch
# import numpy as np
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud
# from mmengine.registry import init_default_scope
# from mmdet3d.apis import init_model

# # --- Config & Model Init ---
# config_file = 'configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
# checkpoint_file = 'checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth'

# init_default_scope('mmdet3d')
# model = init_model(config_file, checkpoint_file, device='cuda:0')
# model.eval()
# # print(model)

# # --- Paths ---
# # lidar_dir = '/home/draiman/Desktop/datasets/nuscenes/samples/LIDAR_TOP'
# # output_dir = '/home/draiman/Desktop/Datasets_nuscenes/lidar'

# lidar_dir = '/home/draiman/Desktop/Datasets_nuscenes/scripts/output_script/validation_attacks_samples/LIDAR_TOP_injected'
# output_dir = '/home/draiman/Desktop/Datasets_nuscenes/lidar_attack'
# os.makedirs(output_dir, exist_ok=True)

# # --- NuScenes Metadata ---
# nusc = NuScenes(version='v1.0-trainval', dataroot='/home/draiman/Desktop/datasets/nuscenes')

# # --- Process LiDAR Files ---
# lidar_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith('.pcd.bin'))

# for idx, lidar_filename in enumerate(lidar_files):
#     # if idx >= 5: # for testing 
#     #     break
#     frame_id = lidar_filename.replace('.pcd.bin', '')
#     timestamp_str = frame_id.split('__')[-1]
#     try:
#         timestamp = int(timestamp_str)
#     except ValueError:
#         print(f"⚠️ Invalid timestamp in {lidar_filename}")
#         continue

#     # Match timestamp to sample_token
#     sample_token = None
#     for sd in nusc.sample_data:
#         if sd['channel'] == 'LIDAR_TOP' and sd['timestamp'] == timestamp:
#             sample_token = sd['sample_token']
#             break

#     if not sample_token:
#         print(f"⚠️ Could not find sample_token for {lidar_filename}")
#         continue

#     lidar_path = os.path.join(lidar_dir, lidar_filename)
#     try:
#         pc = LidarPointCloud.from_file(lidar_path)
#     except Exception as e:
#         print(f"⚠️ Skipping {lidar_filename}: {e}")
#         continue

#     # Process point cloud
#     points = pc.points.T
#     if points.shape[1] == 4:
#         points = np.hstack([points, np.zeros((points.shape[0], 1))])  # Pad to 5D
#     # points = points[:20000]
#     points_tensor = torch.tensor(points, dtype=torch.float32).cuda()

#     # Model forward
#     with torch.no_grad():
#         voxels, coors, num_points = model.data_preprocessor.voxel_layer(points_tensor)
#         voxel_feats = model.pts_voxel_encoder(voxels, num_points, coors)
#         # if voxel_feats.ndim == 3:
#         #     voxel_feats = voxel_feats.mean(dim=1)
#         assert voxel_feats.ndim == 2, f"Unexpected voxel_feats shape: {voxel_feats.shape}"

#         batch_indices = torch.zeros((coors.shape[0], 1), dtype=torch.int, device=coors.device)
#         coors_with_batch = torch.cat([batch_indices, coors], dim=1)
#         spatial_feats = model.pts_middle_encoder(voxel_feats, coors_with_batch, batch_size=1)
#         backbone_feats = model.pts_backbone(spatial_feats)

#     # Save feature
#     selected_feat = backbone_feats[1]  # [1, 256, 90, 90]
#     save_path = os.path.join(output_dir, f'{sample_token}.pt')
#     torch.save({'feat': selected_feat.cpu(), 'meta': {'sample_token': sample_token}}, save_path)
#     if not idx % 200:
#         print(f"[{idx + 1}/{len(lidar_files)}] ✅ Saved: {save_path}")

















# from mmcv import Config
# config_file = 'configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'

# cfg = Config.fromfile(config_file)
# print("pc_range:", cfg.point_cloud_range)
# print("voxel_size:", cfg.voxel_size)



from mmengine.config import Config
config_file = 'configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
cfg = Config.fromfile(config_file)
print("pc_range:", cfg.point_cloud_range)
print("voxel_size:", cfg.voxel_size)






import os
import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from mmengine.registry import init_default_scope
from mmdet3d.apis import init_model


import nuscenes.utils.data_classes
print(nuscenes.utils.data_classes.__file__)
print("-----------------------------------------------")


# --- Config & Model Init ---
config_file = 'configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
checkpoint_file = 'checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth'

init_default_scope('mmdet3d')
model = init_model(config_file, checkpoint_file, device='cuda:0')
model.eval()
# print(model)

# --- Paths ---
# lidar_dir = '/home/draiman/Desktop/datasets/nuscenes/samples/LIDAR_TOP'
# output_dir = '/home/draiman/Desktop/Datasets_nuscenes/lidar'

# lidar_dir = '/home/draiman/Desktop/Datasets_nuscenes/scripts/output_script/validation_attacks_samples/LIDAR_TOP_injected'
# output_dir = '/home/draiman/Desktop/Datasets_nuscenes/lidar_attack_samples_features/lidar_injected_attack'


# lidar_dir  = '/home/draiman/Desktop/Datasets_nuscenes/scripts/output_script/validation_attacks_samples/LIDAR_TOP_frustum_attack_ann_shift'
# output_dir = '/home/draiman/Desktop/Datasets_nuscenes/lidar_attack_samples_features/lidar_attack_shift'

# lidar_dir  = '/home/draiman/Desktop/Datasets_nuscenes/scripts/output_script/validation_attacks_samples/LIDAR_TOP_frustum_attack_ann_point_drop'
# output_dir = '/home/draiman/Desktop/Datasets_nuscenes/lidar_attack_samples_features/lidar_attack_point_drop'



# short attack list 
# lidar_dir  = '/home/draiman/Desktop/Datasets_nuscenes/scripts/output_script/validation_attacks_samples/short_attack_test_list/drop'
# output_dir = '/home/draiman/Desktop/Datasets_nuscenes/lidar_attack_samples_features/lidar_att_short_test_list_point_drop'

# lidar_dir  = '/home/draiman/Desktop/Datasets_nuscenes/scripts/output_script/validation_attacks_samples/short_attack_test_list/shift'
# output_dir = '/home/draiman/Desktop/Datasets_nuscenes/lidar_attack_samples_features/lidar_att_short_test_list_shift'

# single attack list 

lidar_dir  = '/home/draiman/Desktop/Datasets_nuscenes/scripts/output_script/validation_attacks_samples/single_attack_test/drop'
output_dir = '/home/draiman/Desktop/Datasets_nuscenes/lidar_attack_samples_features/single_test_point_drop'



os.makedirs(output_dir, exist_ok=True)

# --- NuScenes Metadata ---
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/draiman/Desktop/datasets/nuscenes')

# --- Process LiDAR Files ---
lidar_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith('.pcd.bin'))

for idx, lidar_filename in enumerate(lidar_files):
    # if idx >= 5: # for testing 
    #     break
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
    # points = points[:20000]
    points_tensor = torch.tensor(points, dtype=torch.float32).cuda()

    # Model forward
    with torch.no_grad():
        voxels, coors, num_points = model.data_preprocessor.voxel_layer(points_tensor)
        if voxels.shape[0] == 0 or coors.shape[0] == 0:
            print(f"⚠️ No voxels found for {lidar_filename}, skipping")
            continue
        voxel_feats = model.pts_voxel_encoder(voxels, num_points, coors)
        # if voxel_feats.ndim == 3:
        #     voxel_feats = voxel_feats.mean(dim=1)
        assert voxel_feats.ndim == 2, f"Unexpected voxel_feats shape: {voxel_feats.shape}"

        batch_indices = torch.zeros((coors.shape[0], 1), dtype=torch.int, device=coors.device)
        coors_with_batch = torch.cat([batch_indices, coors], dim=1)


        # print("points_tensor shape:", points_tensor.shape)
        # print("voxels shape:", voxels.shape)
        # print("coors shape:", coors.shape)
        # print("num_points shape:", num_points.shape)
        # print("voxel_feats shape:", voxel_feats.shape)
        # print("coors_with_batch shape:", coors_with_batch.shape)

        spatial_feats = model.pts_middle_encoder(voxel_feats, coors_with_batch, batch_size=1)
        backbone_feats = model.pts_backbone(spatial_feats)

    # Save feature
    selected_feat = backbone_feats[1]  # [1, 256, 90, 90]
    save_path = os.path.join(output_dir, f'{sample_token}.pt')
    torch.save({'feat': selected_feat.cpu(), 'meta': {'sample_token': sample_token}}, save_path)
    if not idx % 200:
        print(f"[{idx + 1}/{len(lidar_files)}] ✅ Saved: {save_path}")





















