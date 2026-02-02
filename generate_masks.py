"""
    1. Render multi-view images and compute SAM masks for each view
"""
import os
import numpy as np
from tqdm import tqdm
import argparse

import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from src.config import load_cfg_from_cfg_file
from src.mesh_utils import load_mesh_uv, load_mesh_vertex
from src.render import render_images_orthographic, save_images, save_masks

def load_sam(device):
    sam = sam_model_registry["vit_h"](checkpoint='./checkpoints/sam_vit_h_4b8939.pth').to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )
    return mask_generator

def main(cfg, mask_generator, device):
    dataset = cfg.dataset
    mesh_name = cfg.mesh_name
    mesh_path = cfg.mesh_path
    texture_path = cfg.texture_path
    out_root =  cfg.output_dir
    num_views = cfg.num_views
    num_samples = cfg.num_samples
    
    out_dir = os.path.join(out_root, dataset+'_'+mesh_name)
    os.makedirs(out_dir, exist_ok=True)
    # load mesh
    if texture_path is None:
        mesh, verts, verts_norm, points, points_norm = load_mesh_vertex(mesh_path, num_samples=num_samples)
    else:
        mesh, verts, verts_norm, points, points_norm = load_mesh_uv(mesh_path, texture_path, num_samples=num_samples)
    mesh = mesh.to(device)
    points_norm = points_norm.to(device)
    
    print("Rendering multi-view images...")
    step_size = 360 / num_views
    azim = torch.arange(-180, 180, step_size)

    img_list, uv_list, view_mask_list, sil_list = render_images_orthographic(device, mesh, points_norm, azim, elev=0, dist=1, k=1)
    save_images(img_list, out_dir)

    print("Computing SAM masks for each view...")
    area_thresh = 10
    masks2D_list, masks3D_list = [], []
    for i in tqdm(range(len(img_list))):
        view_dir = os.path.join(out_dir, 'view_{}'.format(i))
        os.makedirs(view_dir, exist_ok=True)
        
        img = img_list[i]
        uv = uv_list[i]
        view_mask = view_mask_list[i].cuda()
        sil = sil_list[i]
        with torch.no_grad():
            sam_masks = mask_generator.generate(img)
        num_masks = len(sam_masks)
        masks = []
        for j in range(num_masks):
            mask = sam_masks[j]['segmentation']
            mask_ = mask * sil
            area = np.sum(mask_)
            if area > area_thresh:
                masks.append(mask_)
        masks2D_list.append(masks)
        
        save_masks(masks, view_dir)
            
        # project 2D SAM masks to 3D
        masks2D = np.stack(masks, axis=0)
        masks2D = torch.from_numpy(masks2D).unsqueeze(0).cuda()
        masks3D = torch.nn.functional.grid_sample(masks2D.float(), uv, mode='nearest', align_corners=True)
        masks3D = masks3D.squeeze()
        view_mask = view_mask.unsqueeze(0)
        # set mask of invisible vertices to 0
        masks3D = masks3D * view_mask
        masks3D_list.append(masks3D)
    
    masks = torch.cat(masks3D_list, dim=0).cpu()
    save_data = {
        'images': img_list,
        'view_mask': view_mask_list,
        'silhouette': sil_list,
        'masks2D': masks2D_list,
        'masks3D': masks,
        'points': points
    }
    torch.save(save_data, out_dir+'/save_data.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference for a sequence of 3D meshes")
    parser.add_argument("--config", type=str, default=True, help="Path to the config file")
    args = parser.parse_args()
    
    cfg = load_cfg_from_cfg_file(args.config)
    
    device = torch.device('cuda')
    mask_generator = load_sam(device)
    main(cfg, mask_generator, device)