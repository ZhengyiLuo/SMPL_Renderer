import os
import sys
import pdb
sys.path.append(os.getcwd())

import torch
import pickle as pk
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import torch

from zen_renderer.dataloaders.dataset_amass import Dataset_AMASS
from zen_renderer.renderer.smpl_renderer import SMPL_Renderer
from zen_renderer.utils.transform_utils import *
from zen_renderer.utils.image_utils import *
import time
from multiprocessing import Pool


def render_job_videos(job_list, image_size, gpu_index):
    device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    smpl_renderer = SMPL_Renderer(camera_mode="look_at", device = device, image_size = image_size)
    for curr_pose, out_file_name in tqdm(job_list):
        if not os.path.isfile(out_file_name):
            curr_pose = torch.tensor(curr_pose)
            smpl_renderer.render_pose_vid(curr_pose, out_file_name, random_camera=False, random_shape=True, frame_chunk=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--job', type=str, default=150)
    args = parser.parse_args()

    ''' confg '''
    image_size = args.image_size
    gpu_index = args.gpu_index
    job_list = pk.load(open(args.job, "rb"))
    render_job_videos(job_list, image_size, gpu_index)