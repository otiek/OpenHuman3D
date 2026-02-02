import numpy as np
import open3d as o3d
from PIL import Image

import torch
import torchvision.transforms.functional as TF

from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
    TexturesUV
)

def normalize_verts_points(verts, points):
    centroid = np.mean(verts, axis=0)
    verts = verts - centroid
    m = np.max(np.sqrt(np.sum(verts ** 2, axis=1)))
    verts = verts / m
    points = points - centroid
    points = points / m
    return verts, points

# load mesh with vertex colors
def load_mesh_vertex(mesh_path, num_samples=100000):
    mesh = o3d.io.read_triangle_mesh(mesh_path, True)
    verts = np.asarray(mesh.vertices)
    # sample points
    pc = mesh.sample_points_poisson_disk(number_of_points=num_samples)
    points = np.asarray(pc.points)
    # normalize vertices/points
    verts_norm, points_norm = normalize_verts_points(verts, points)
    
    mesh_faces = np.asarray(mesh.triangles)
    mesh_verts_color = np.asarray(mesh.vertex_colors)
    mesh_verts = torch.Tensor(verts_norm)
    mesh_faces = torch.Tensor(mesh_faces)
    mesh_verts_color = torch.Tensor(mesh_verts_color).unsqueeze(0)
    textures = TexturesVertex(verts_features=mesh_verts_color)
    mesh = Meshes(
        verts = [mesh_verts],
        faces = [mesh_faces],
        textures=textures
    )
    verts_norm = mesh_verts.unsqueeze(0)
    points_norm = torch.from_numpy(points_norm).unsqueeze(0)
    return mesh, verts, verts_norm, points, points_norm

# loading mesh with UV texture map
def load_mesh_uv(mesh_path, texture_path=None, num_samples=100000):
    mesh = o3d.io.read_triangle_mesh(mesh_path, True)
    # sample points
    pc = mesh.sample_points_poisson_disk(number_of_points=num_samples)
    points = np.asarray(pc.points)
    
    mesh = load_objs_as_meshes([mesh_path])
    if texture_path is not None:
        texture = Image.open(texture_path)
        texture = TF.pil_to_tensor(texture) / 255.
        texture = texture.permute(1, 2, 0).unsqueeze(0)
        verts, faces, aux = load_obj(mesh_path)
        verts_uvs = aux.verts_uvs[None, ...]
        faces_uvs = faces.textures_idx[None, ...]
        texture = TexturesUV(maps=texture,
                            faces_uvs=faces_uvs,
                            verts_uvs=verts_uvs)
        mesh.textures = texture
    verts = mesh.verts_packed().cpu().numpy()
    # normalize vertices/points
    verts_norm, points_norm = normalize_verts_points(verts, points)
    verts_norm = torch.from_numpy(verts_norm).unsqueeze(0)
    points_norm = torch.from_numpy(points_norm).unsqueeze(0)
    mesh = mesh.update_padded(verts_norm)
    return mesh, verts, verts_norm, points, points_norm