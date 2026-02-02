"""
    2. Compute CLIP embeddings for each mask with HumanCLIP
"""
import os
from PIL import Image
import numpy as np
import glob
import argparse

import torch
from torchvision import transforms

from AlphaCLIP import alpha_clip
from src.config import load_cfg_from_cfg_file

def get_mask_embedding(mask, image, mask_transform, model):
    if len(mask.shape) == 2: binary_mask = (mask == 255)
    if len(mask.shape) == 3: binary_mask = (mask[:, :, 0] == 255)
    alpha = mask_transform((binary_mask * 255).astype(np.uint8))
    alpha = alpha.half().cuda().unsqueeze(dim=0)
    with torch.no_grad():
        mask_feature = model.visual(image, alpha)
    return mask_feature

def compute_mask_embeddings(save_dir, masks, silhouettes, model, mask_transform, preprocess, num_views=8):
    mask_embeddings_list = []
    keep_idx = []
    idx = 0
    for i in range(num_views):
        view_dir = os.path.join(save_dir, 'view_{}'.format(i))
        img_path = os.path.join(save_dir, 'img_{}.png'.format(i))
        image = Image.open(img_path).convert('RGB')
        image = preprocess(image).unsqueeze(0).half().to(device)

        sil = silhouettes[i]
        area_sil = np.sum(sil)

        masks_list = sorted(glob.glob(view_dir+'/mask_*.png'))
        for mask_path in masks_list:
            mask = np.array(Image.open(mask_path))
            mask_ = mask  / 255.
            overlap = np.sum(mask_*sil)
            overlap_ratio = overlap / area_sil
            # remove masks that cover the entire person
            if overlap_ratio < 0.8:
                keep_idx.append(idx)
            mask_embedding = get_mask_embedding(mask, image, mask_transform, model)
            mask_embeddings_list.append(mask_embedding)
            idx += 1

    masks = masks[keep_idx]
    mask_embedding = torch.cat(mask_embeddings_list, dim=0)
    mask_embedding = mask_embedding[keep_idx]
    mask_embedding = mask_embedding / mask_embedding.norm(dim=1, keepdim=True)
    return mask_embedding, masks

def main(save_dir, model, mask_transform, preprocess, device):
    save_data_path = os.path.join(save_dir, 'save_data.pt')
    save_data = torch.load(save_data_path)
    silhouettes = save_data['silhouette']
    masks = save_data['masks3D']
    masks = masks.to(device)
    mask_embeddings, masks = compute_mask_embeddings(save_dir, masks, silhouettes, model, mask_transform, preprocess)
    
    masks = masks.cpu()
    mask_embeddings = mask_embeddings.cpu()
    save_data['masks3D'] = masks
    save_data['mask_embeddings'] = mask_embeddings
    torch.save(save_data, save_dir+'/save_data.pt')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference for a sequence of 3D meshes")
    parser.add_argument("--config", type=str, default=True, help="Path to the config file")
    args = parser.parse_args()
    
    cfg = load_cfg_from_cfg_file(args.config)
    
    device = torch.device('cuda')
    model, preprocess = alpha_clip.load("ViT-L/14", alpha_vision_ckpt_pth="./checkpoints/humanclip.pth", device=device)
    mask_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((224, 224)),
        transforms.Normalize(0.5, 0.26)
    ])
    
    save_dir = os.path.join(cfg.output_dir, cfg.dataset+'_'+cfg.mesh_name)
    main(save_dir, model, mask_transform, preprocess, device)