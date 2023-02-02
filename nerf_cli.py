#!/usr/bin/env python3
import os
import sys
import click
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'Documents/Git/kornia/'))

import kornia.nerf.nerf_solver

@click.command('cli', context_settings={'show_default': True})
@click.option('--device', default='cpu', help='Device to run on (cpu/gpu)')
@click.option('--params', default='None', help='Path to .json file with NeRF params for training')
@click.option('--scene_dir', default='./', help='Scene directory with images and camera models')
@click.option('--image_dir', default='images', help='Image directory (relative to scene_dir)')
@click.option('--colmap_cameras', default='cameras.txt', help='Colmap camera model filename (relative to scene_dir)')
@click.option('--colmap_images', default='images.txt', help='Colmap image-camera relation filename (relative to '
                                                            'scene_dir)')
def cli(device, params, scene_dir, image_dir, colmap_cameras, colmap_images):
    image_dir = os.path.join(scene_dir, image_dir)
    colmap_cameras = os.path.join(scene_dir, colmap_cameras)
    colmap_images = os.path.join(scene_dir, colmap_images)

    print('NeRF training')
    print('=============')
    print(f'Scene directory: {os.path.abspath(scene_dir)}')
    print(f'Image directory: {os.path.abspath(image_dir)}')
    print(f'Colmap camera file path: {os.path.abspath(colmap_cameras)}')
    print(f'Colmap image-camera relation file path: {os.path.abspath(colmap_images)}')
    print(f'Computation device: {device}')


if __name__ == '__main__':
    cli()
 