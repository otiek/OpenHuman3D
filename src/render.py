import numpy as np
from PIL import Image

import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    BlendParams
)

def save_images(img_list, save_dir):
    for i, img in enumerate(img_list):
        img = Image.fromarray(img)
        img.save(save_dir+'/img_{}.png'.format(i))   

def save_masks(masks_list, save_dir):
    for i, mask in enumerate(masks_list):
        mask = np.uint8(mask*255)
        mask = Image.fromarray(mask)
        mask.save(save_dir+'/mask_{:02d}.png'.format(i))        

def get_visibility_points(points_rasterizer, points, k=1):
    batch_size, num_points, _ = points.shape
    pc = Pointclouds(points=points, features=torch.ones_like(points)).to(points.device)
    view_mask = torch.zeros((batch_size*num_points)).to(points.device)
    fragments = points_rasterizer(pc)
    closest_idx= fragments.idx[:, :, :, :k]
    closest_idx = torch.unique(closest_idx)[1:]
    view_mask[closest_idx.long()] = 1
    return view_mask

def render_images_orthographic(device, mesh, points, azim, elev, dist=1, k=1):
    num_views = len(azim)
    vertices = mesh.verts_packed()
    vertices = vertices.unsqueeze(0)
    points = points.float()

    mesh_raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0, 
        faces_per_pixel=1
    )
    silhouette_raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0,
        faces_per_pixel=50
    )
    pc_raster_settings = PointsRasterizationSettings(
                image_size=512, 
                radius=0.01,
                points_per_pixel=10
            )
    blend_params = BlendParams(background_color=[1, 1, 1])

    img_list, uv_list, visibility_list, sil_list = [], [], [], []
    for i in range(num_views):
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim[i])
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=mesh_raster_settings
        )
        image_renderer = MeshRenderer(
                        rasterizer=rasterizer,
                        shader=SoftPhongShader(
                            device=device, 
                            cameras=cameras
                        )
                    )
        silhouette_rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=silhouette_raster_settings
        )
        silhouette_renderer = MeshRenderer(
                        rasterizer=silhouette_rasterizer,
                        shader=SoftSilhouetteShader()
                    )
        points_rasterizer = PointsRasterizer(cameras=cameras, raster_settings=pc_raster_settings)
        img = image_renderer(mesh, blend_params=blend_params)
        img = img[:, :, :, :3]
        img = img.squeeze().cpu().numpy()
        img = np.uint8(img*255)
        img_list.append(img)

        silhouette = silhouette_renderer(mesh)
        silhouette = silhouette[:, :, :, 3]
        silhouette = silhouette.squeeze().cpu().numpy()
        silhouette_ = np.zeros_like(silhouette)
        silhouette_[silhouette > 0.5] = 1
        sil_list.append(silhouette_)

        visibility_map = get_visibility_points(points_rasterizer, points, k=k)

        visibility_list.append(visibility_map)
    
        xyz = cameras.transform_points(points)
        xyz = xyz.permute(0, 2, 1)
        uv = xyz[:, :2, :]
        uv = -uv.transpose(1, 2)
        uv = uv.unsqueeze(2)
        uv_list.append(uv)
    
    return img_list, uv_list, visibility_list, sil_list