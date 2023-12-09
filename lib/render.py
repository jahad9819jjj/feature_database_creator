import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    OpenGLOrthographicCameras,
    look_at_view_transform,
    look_at_rotation,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
    HardPhongShader,
    PointLights,
    Textures
    )
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class RenderObj:
    """
    A dataclass to store the properties of a rendered object.
    This includes vertices, faces, normals, texture coordinates, material colors, texture images, texture atlas, and bounding box size.
    """
    verts: torch.Tensor = None # indices of vertices
    faces: torch.Tensor = None # indices of faces
    normals: Optional[torch.Tensor] = None # vertex normals
    verts_uvs: Optional[torch.Tensor] = None # texture coordinates
    material_colors: Optional[Dict[str, Dict[str, torch.Tensor]]] = None # color of material
    texture_images: Optional[Dict[str, torch.Tensor]] = None # texture images
    texture_atlas: Optional[torch.Tensor] = None # texture atlas
    bbox_size: Optional[torch.Tensor] = None # size of bounding box
    
class Renderer:
    """
    A class for rendering 3D objects from OBJ files using PyTorch3D.
    It supports rendering in silhouette and Phong shading modes and can be configured to use GPU acceleration if available.    
    """
    def __init__(self, accelerate=False):
        """
        Initializes the Renderer with optional GPU acceleration.
        
        Args:
            accelerate (bool): If set to True, will use GPU acceleration if available.
        """
        self.render_obj = None
        self.accelerate = accelerate
        
        if (self.accelerate and torch.cuda.is_available()):
            self.device = torch.device("cuda:0") 
            torch.cuda.set_device(self.device)
        elif (self.accelerate and torch.backends.mps.is_available()):
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.cameras = OpenGLPerspectiveCameras(device=self.device)
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        self.raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=np.log(1. / 1e-4 - 1.) * self.blend_params.sigma, 
            faces_per_pixel=100, 
            bin_size=0
            )
        self.silhouette_shader = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings
                ), 
            shader=SoftSilhouetteShader(blend_params=self.blend_params)
            )
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings
                ),
            shader=HardPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
            )
        )
            
        
    def load_obj(self, obj_path):
        """
        Loads an OBJ file and calculates its bounding box size. Stores the object's properties in a RenderObj instance.
        
        Args:
            obj_path (str): File path to the OBJ file.
        
        Returns:
            Meshes: A PyTorch3D Meshes object representing the loaded 3D object.
        """
        self.render_obj = RenderObj()
        verts, faces_idx, aux = load_obj(obj_path, device=self.device) # [verts, Faces, Properties]
        self.render_obj.verts = verts
        self.render_obj.faces = faces_idx.verts_idx
        self.render_obj.normals = aux.normals if 'normals' in aux else None
        self.render_obj.verts_uvs = aux.verts_uvs if 'verts_uvs' in aux else None
        self.render_obj.material_colors = aux.material_colors if 'material_colors' in aux else None
        self.render_obj.texture_images = aux.texture_images if 'texture_images' in aux else None
        
        min_vals, _ = torch.min(self.render_obj.verts, dim=0)
        max_vals, _ = torch.max(self.render_obj.verts, dim=0)
        self.render_obj.bbox_size = torch.max(max_vals - min_vals)
        
        textures = None
        if self.render_obj.texture_images is not None:
            texture_map = next(iter(self.render_obj.texture_images.values()))
            textures = Textures(verts_uvs=[self.render_obj.verts_uvs], maps=[texture_map])
        else:
            verts_rgb = torch.ones_like(self.render_obj.verts)[None]  # (1, V, 3)
            textures = Textures(verts_rgb=verts_rgb.to(self.device))

        mesh = Meshes(
            verts=[self.render_obj.verts.to(self.device)], 
            faces=[self.render_obj.faces.to(self.device)],
            textures=textures  # テクスチャ情報を追加
        )
        return mesh

    
    def rendering(self, 
                  mesh, 
                  distance=None,
                  distance_scale=3.0,
                  elevation=0.0, 
                  azimuth=0.0, 
                  mode='silhouette'
                  ):
        """
        Renders the given mesh from a specified viewpoint and mode. Automatically adjusts the camera distance based on the bounding box size if distance is not provided.

        Args:
            mesh (Meshes): A PyTorch3D Meshes object to render.
            distance (float, optional): Distance of the camera from the object. If None, automatically calculated from bounding box size.
            elevation (float): Elevation angle of the camera.
            azimuth (float): Azimuth angle of the camera.
            mode (str): Rendering mode, either 'silhouette' or 'phong'.

        Returns:
            np.ndarray: The rendered image as a NumPy array.
        """
        if distance is None:
            bbox_max_size = torch.max(self.render_obj.bbox_size).item()
            distance = bbox_max_size * distance_scale
        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.device)
        if mode == 'silhouette':
            image = self.silhouette_shader(meshes_world=mesh, R=R, T=T)
        elif mode == 'phong':
            image = self.phong_renderer(meshes_world=mesh, R=R, T=T)
        else:
            raise ValueError('mode must be silhouette or phong')
        image = image.squeeze().cpu().numpy()
        return image
    
    
# main
if __name__ == '__main__':
    renderer = Renderer(accelerate=False)
    mesh = renderer.load_obj('data/stanford-bunny_translatd_decimate036.obj')
    
    # image_ref = renderer.rendering(mesh, distance=0.6, elevation=0.0, azimuth=0.0, mode='phong')
    # sillouette = renderer.rendering(mesh, distance=0.6, elevation=0.0, azimuth=0.0, mode='silhouette')
    
    image_ref = renderer.rendering(mesh,  distance_scale=3.5, elevation=30.0, azimuth=90.0, mode='phong')
    sillouette = renderer.rendering(mesh, distance_scale=3.5, elevation=30.0, azimuth=90.0, mode='silhouette')
    def show_image(image):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.show()
    show_image(image_ref)
    # show_image(sillouette[..., 3])