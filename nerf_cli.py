#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'Documents/Git/kornia/'))

import click
import json
import logging
import torch
from torchvision.io import read_image
from typing import List

from kornia.nerf import logger
from kornia.nerf.colmap_parser import parse_colmap_output, parse_colmap_points_3d
from kornia.nerf.nerf_solver import NerfSolver, NerfParams
from kornia.nerf.rays import analyze_points_3d


@click.command('cli', context_settings={'show_default': True})
@click.option('--device', default='cpu', help='Device to run on (cpu/gpu)')
@click.option('--scene_dir', default='./', help='Scene directory with images and camera models')
@click.option('--json_params', default=None, 
              help='Path to .json file with NeRF parameters for training (relative to scene_dir)')
@click.option('--image_dir', default='images', help='Image directory (relative to scene_dir)')
@click.option('--colmap_cameras', default='cameras.txt', help='Colmap camera model filename (relative to scene_dir)')
@click.option('--colmap_images', default='images.txt', help='Colmap image-camera relation filename (relative to '
                                                            'scene_dir)')
@click.option('--colmap_points3d', default=None, 
              help='Colmap 3d point cloud filename (relative to scene_dir)')
@click.option('--output_dir', help='Directory to output run products (relative to scene_dir)', required=True)
@click.option('--logger_path', default='logger.log', help='Path to log file (relative to output_dir)')
@click.option("--suppress_terminal_logs", is_flag=True, show_default=True, default=False, 
              help="Avoid output log messages to the terimnal")
@click.option('--checkpoint_save_dir', default=None, 
              help='Directory to save model checkpoints (relative to output_dir)')
@click.option('--checkpoint_load_path', default=None, 
              help='Path to model checkpoint file to load (relative to scene_dir)')
@click.option('--num_iters', default=10000, help='Number of training iterations')
def cli(device, scene_dir, json_params, image_dir, colmap_cameras, colmap_images, colmap_points3d, output_dir, 
        logger_path, suppress_terminal_logs, checkpoint_save_dir, checkpoint_load_path, num_iters):
    
    # Assign input/output paths
    if json_params is not None:
        json_params = os.path.join(scene_dir, json_params)
    image_dir = os.path.join(scene_dir, image_dir)
    colmap_cameras = os.path.join(scene_dir, colmap_cameras)
    colmap_images = os.path.join(scene_dir, colmap_images)
    if colmap_points3d is not None:
        colmap_points3d = os.path.join(scene_dir, colmap_points3d)
    output_dir = os.path.join(scene_dir, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger_path = os.path.join(output_dir, logger_path)
    if checkpoint_save_dir is not None:
        checkpoint_save_dir = os.path.join(output_dir, checkpoint_save_dir)
    if checkpoint_load_path is not None:
        checkpoint_load_path = os.path.join(scene_dir, checkpoint_load_path)

    # Define logger
    fh = logging.FileHandler(logger_path, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if suppress_terminal_logs:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler) 

    # Start training
    logger.info('NeRF training')
    logger.info('=============')
    logger.info(f'Scene directory: {os.path.abspath(scene_dir)}')
    if json_params is not None:
        logger.info(f'Parameter path: {os.path.abspath(json_params)}')
    logger.info(f'Image directory: {os.path.abspath(image_dir)}')
    logger.info(f'Colmap camera file path: {os.path.abspath(colmap_cameras)}')
    logger.info(f'Colmap image-camera relation file path: {os.path.abspath(colmap_images)}')
    if colmap_points3d is not None:
        logger.info(f'Colmap 3d point cloud file path: {os.path.abspath(colmap_points3d)}')
    logger.info(f'Directory to output run products: {os.path.abspath(output_dir)}')
    logger.info(f'Run logs will be output to: {os.path.abspath(logger_path)}')
    if checkpoint_save_dir is not None:
        logger.info(f'Model checkpoints will be written to: {os.path.abspath(checkpoint_save_dir)}')
    if checkpoint_load_path is not None:
        logger.info(f'Checkpoint file will be loaded from: {os.path.abspath(checkpoint_load_path)}')
    logger.info(f'Computation device: {device}')

    logger.info('Phase 1: parse Colmap camera and image-camera relation files')
    img_names, cameras = parse_colmap_output(
            cameras_path=colmap_cameras, 
            images_path=colmap_images, 
            device=device,
            dtype=torch.float32, 
            sort_by_image_names=True)
    logger.info(f'Number of scene cameras: {cameras.batch_size}')
    logger.info('Image files associated with scene cameras: ')
    for i, img_name in enumerate(img_names):
        logger.info(f'Image name: {img_name}, shape: {cameras.width[i].item(), cameras.height[i].item()}')

    logger.info('Pahse 2: parse and analyze Colmap 3D point cloud')
    if colmap_points3d is not None:
        points_3d_sparse_cloud = parse_colmap_points_3d(colmap_points3d, device, torch.float32)
        logger.info(f'Total number of 3D point cloud: {points_3d_sparse_cloud.shape[0]}')
        min_depth, max_depth = analyze_points_3d(points_3d_sparse_cloud, cameras)
        logger.info(f'Point cloud minimum depth tensor: \n{min_depth}')
        logger.info(f'Point cloud maximum depth tensor: \n{max_depth}')
        min_depth_cameras = min_depth.min().item()
        max_depth_cameras = max_depth.max().item()
        logger.info(f'Closest and farthest points: {min_depth_cameras, max_depth_cameras}')
        logger.info('These values will be assigned as min/man depths')
    else:
        logger.info('3D point cloud file was not supplied. Will use min/max depth input parameters')

    logger.info('Pahse 3: load images')
    imgs: List[torch.tensor] = []
    for img_name in img_names:
        img_path = os.path.join(image_dir, img_name)
        img = read_image(img_path)
        img = img[:3, ...]    # FIXME: This is a hack until I understand how to work with the alpha channel
        imgs.append(img.to(device))

    logger.info('Phase 4: define NeRF parameters')
    nerf_params = NerfParams()
    if json_params is not None:
        logger.info('Json parameter file will be loaded')
        with open(json_params, 'r') as f:
            json_nerf_params = f.read()
            logger.info(f'Use the following non-default training parameters read from Json: {json_nerf_params}')
            nerf_params = NerfParams.from_json(json_nerf_params)
        read_params = json.loads(json_nerf_params)
    else:
        logger.info('Json parameter file was not supplied. Using default parameters')
    if colmap_points3d is not None:
        logger.info('Min/max depth values will be taken from point cloud analysis, or, if supplied directly, from the Json '
             'param file')
        if json_params is not None and '_min_depth' in read_params:
            logger.info('Overwriting min depth value from point cloud analysis by its value from Json param file')
            nerf_params._min_depth = read_params['_min_depth']
        else:
            logger.info('Taking min depth value from point cloud analysis')
            nerf_params._min_depth = min_depth_cameras
        if json_params is not None and '_max_depth' in read_params:
            logger.info('Overwriting max depth value from point cloud analysis by its value from Json param file')
            nerf_params._max_depth = read_params['_max_depth']
        else:
            logger.info('Taking max depth value from point cloud analysis')
            nerf_params._max_depth = max_depth_cameras
    logger.info(f'NeRF parameters: {nerf_params.__str__(indent=True)}')
    
    logger.info('Phase 5: run NeRF training on scene images')
    nerf_obj = NerfSolver(device=device, dtype=torch.float32, params=nerf_params, 
                          checkpoint_save_dir=checkpoint_save_dir, checkpoint_save_niter=2000)
    if checkpoint_load_path is not None:
        logger.info('Loading model checkpoint file')
        nerf_obj.load_checkpoint(checkpoint_load_path)
        logger.info(f'Training will continue from iteration {nerf_obj.iter}')
    nerf_obj.set_cameras_and_images_for_training(cameras, imgs)
    nerf_obj.run(num_iters=num_iters)
    

if __name__ == '__main__':
    cli()
 