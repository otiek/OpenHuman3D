"""
    3. Segment 3D points based on the input texts
"""
import os
import pymeshlab as ml
import open3d as o3d
from sklearn.neighbors import KDTree
import numpy as np
import argparse

import torch

from AlphaCLIP import alpha_clip
from src.config import load_cfg_from_cfg_file

color_palette = {
    'MGN': [(255, 0, 0), (0, 255, 0), (0, 8, 255)],
    'SIZER': [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191), (153, 153, 153)],
    'CTD': [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2), (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108)],
    'THuman': [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128)],
    'PosedPro': [(0, 0, 0), (255, 229, 109), (0, 255, 21), (245, 0, 255), (0, 231, 255), (91, 0, 255), (203, 255, 0), (255, 0, 140), (154, 255, 0), (9, 157, 0), (200, 79, 27), (150, 0, 0), (108, 121, 88), (133, 104, 84), (0, 210, 255), (127, 66, 0), (197, 165, 211), (140, 125, 109), (238, 255, 1)]
}

def convert_label(label, colormap):
    num_classes = colormap.shape[0]
    label_color = np.zeros((len(label), 3))
    for n in range(num_classes):
        label_color[label==n] = colormap[n]
    return label_color

def visualize(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=500, height=800)
    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    vis.get_render_option().point_size = 7.0
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    vis.run()
    
def segment_points3D(mask_embedding, masks, points, class_texts, model, device):
    class_tokens = alpha_clip.tokenize(class_texts).to(device)
    text_embedding = model.encode_text(class_tokens)
    text_embedding = text_embedding / text_embedding.norm(dim=1, keepdim=True)

    logits = mask_embedding.float() @ text_embedding.float().t()
    logits = (100.0 * logits).softmax(dim=-1) 
    seg_pred = masks.t() @ logits
    pred_choice = seg_pred.contiguous().data.max(1)[1].cpu()
    
    total_masks = torch.sum(masks, dim=0).cpu()
    points = points[total_masks!=0]
    pred_choice = pred_choice[total_masks!=0]
    
    label = pred_choice.numpy()
    return label, points

def main(save_dir, mesh_path, class_texts, model, device):
    save_data_path = os.path.join(save_dir, 'save_data.pt')
    save_data = torch.load(save_data_path)
    masks = save_data['masks3D'].to(device)
    mask_embeddings = save_data['mask_embeddings'].to(device)
    points = save_data['points']
    label, points = segment_points3D(mask_embeddings, masks, points, class_texts, model, device)
    
    ms = ml.MeshSet()
    ms.load_new_mesh(mesh_path)
    ml_mesh = ms.current_mesh()
    vertices = ml_mesh.vertex_matrix()
    
    tree = KDTree(points)
    _, indices = tree.query(vertices, k=1)
    indices = indices[:, 0]
    label = label[indices]
    
    # save labels
    output = np.concatenate((vertices, label[:, None]), axis=1)
    output_path = os.path.join(save_dir, 'labels.txt')
    np.savetxt(output_path, output, fmt="%6f")
    
    # visualize segmentation results
    colormap = np.array(color_palette['THuman']) / 255
    vis_label = convert_label(label, colormap)
    visualize(vertices, vis_label)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference for a sequence of 3D meshes")
    parser.add_argument("--config", type=str, default=True, help="Path to the config file")
    args = parser.parse_args()
    
    cfg = load_cfg_from_cfg_file(args.config)
    
    device = torch.device('cuda')
    model, preprocess = alpha_clip.load("ViT-L/14", alpha_vision_ckpt_pth="./checkpoints/humanclip.pth", device=device)
    
    save_dir = os.path.join(cfg.output_dir, cfg.dataset+'_'+cfg.mesh_name)
    mesh_path = cfg.mesh_path
    class_texts = cfg.class_texts
    main(save_dir, mesh_path, class_texts, model, device)