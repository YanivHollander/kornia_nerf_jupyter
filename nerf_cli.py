#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'Documents/Git/kornia/'))

import click
import torch
from torchvision.io import read_image
from typing import List

from kornia.nerf.colmap_parser import parse_colmap_output, parse_colmap_points_3d
from kornia.nerf.nerf_solver import NerfSolver, NerfParams
from kornia.nerf.rays import analyze_points_3d


@click.command('cli', context_settings={'show_default': True})
@click.option('--device', default='cpu', help='Device to run on (cpu/gpu)')
@click.option('--params', default='None', help='Path to .json file with NeRF params for training')
@click.option('--scene_dir', default='./', help='Scene directory with images and camera models')
@click.option('--image_dir', default='images', help='Image directory (relative to scene_dir)')
@click.option('--colmap_cameras', default='cameras.txt', help='Colmap camera model filename (relative to scene_dir)')
@click.option('--colmap_images', default='images.txt', help='Colmap image-camera relation filename (relative to '
                                                            'scene_dir)')
@click.option('--colmap_points3d', default='points3D.txt', help='Colmap 3d point cloud filename (relative to scene_dir)')
def cli(device, params, scene_dir, image_dir, colmap_cameras, colmap_images, colmap_points3d):
    image_dir = os.path.join(scene_dir, image_dir)
    colmap_cameras = os.path.join(scene_dir, colmap_cameras)
    colmap_images = os.path.join(scene_dir, colmap_images)
    colmap_points3d = os.path.join(scene_dir, colmap_points3d)

    print('NeRF training')
    print('=============')
    print(f'Scene directory: {os.path.abspath(scene_dir)}')
    print(f'Image directory: {os.path.abspath(image_dir)}')
    print(f'Colmap camera file path: {os.path.abspath(colmap_cameras)}')
    print(f'Colmap image-camera relation file path: {os.path.abspath(colmap_images)}')
    print(f'Colmap 3d point cloud file path: {os.path.abspath(colmap_points3d)}')
    print(f'Computation device: {device}')

    print('Phase 1: parse Colmap camera and image-camera relation files')
    img_names, cameras = parse_colmap_output(
            cameras_path=colmap_cameras, 
            images_path=colmap_images, 
            device=device,
            dtype=torch.float32, 
            sort_by_image_names=True)
    print(f'Number of scene cameras: {cameras.batch_size}')
    print('Image files associated with scene cameras: ')
    for img_name in img_names:
        print(img_name)

    print('Pahse 2: parse and analyze Colmap 3D point cloud')
    points_3d_sparse_cloud = parse_colmap_points_3d(colmap_points3d, device, torch.float32)
    print(f'Total number of 3D point cloud: {points_3d_sparse_cloud.shape[0]}')
    min_depth, max_depth = analyze_points_3d(points_3d_sparse_cloud, cameras)
    print(f'Point cloud minimum depth tensor: \n{min_depth}')
    print(f'Point cloud maximum depth tensor: \n{max_depth}')
    min_depth_cameras = min_depth.min().item()
    max_depth_cameras = max_depth.max().item()
    print(f'Closest and farthest points: {min_depth_cameras, max_depth_cameras}')

    print('Pahse 3: load images')
    imgs: List[torch.tensor] = []
    for img_name in img_names:
        img_path = os.path.join(image_dir, img_name)
        img = read_image(img_path)
        img = img[:3, ...]    # FIXME: This is a hack until I understand how to work with the alpha channel
        imgs.append(img.to(device))

    print('Phase 4: run NeRF training on scene images')
    nerf_params = NerfParams(max_depth=max_depth_cameras, batch_size=4096)  # FIXME: Add parsing of input Json for user defined params
    nerf_obj = NerfSolver(device=device, dtype=torch.float32, params=nerf_params)
    nerf_obj.set_cameras_and_images_for_training(cameras, imgs)
    nerf_obj.run(num_iters=10000)
    


if __name__ == '__main__':
    cli()
 