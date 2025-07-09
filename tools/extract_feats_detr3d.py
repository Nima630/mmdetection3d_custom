import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmengine.registry import DefaultScope

import os

# Create output folder if it doesn't exist
output_dir = 'output_feats'
os.makedirs(output_dir, exist_ok=True)

# 🔧 STEP 1: Set paths
cfg_path = 'projects/DETR3D/configs/detr3d_r101_gridmask.py'
ckpt_path = 'checkpoints/detr3d_r101_gridmask.pth'

# 🔧 STEP 2: Load config
cfg = Config.fromfile(cfg_path)
cfg.model.pretrained = None  # avoid loading ImageNet pretrained weights
cfg.model.pts_bbox_head = None  # remove detection head
cfg.model.train_cfg = None  # remove training-specific logic
cfg.model.test_cfg = None

# 🔧 STEP 3: Build model
DefaultScope.get_instance("extract_feats", scope_name="mmdet3d")
model = MODELS.build(cfg.model)
model.eval()

# 🔧 STEP 4: Load checkpoint
load_checkpoint(model, ckpt_path, map_location='cpu')

# 🔧 STEP 5: Load dummy images (simulate 6 cameras)
dummy_imgs = torch.rand(1, 6, 3, 600, 800)  # (B, N=6, C, H, W)
metainfo = [{
    'img_shape': (600, 800),
    'cam2img': [torch.eye(3).tolist() for _ in range(6)],
    'lidar2cam': [torch.eye(4).tolist() for _ in range(6)]
}]

# 🔧 STEP 6: Extract features
with torch.no_grad():
    feats = model.extract_img_feat(dummy_imgs, metainfo)

# 🔧 STEP 7: Save feature maps
for i, feat_tensor in enumerate(feats):
    # torch.save(f, f'bev_feats_lvl{i}.pt')
    torch.save(feat_tensor, os.path.join(output_dir, f'bev_feats_lvl{i}.pt'))


print("✅ Features saved as .pt files.")
