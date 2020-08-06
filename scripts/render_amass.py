import os
import sys
import pdb
sys.path.append(os.getcwd())

import torch
import pickle as pk
import cv2
import numpy as np
import argparse

from zen_renderer.smpl_parser import SMPL_Parser
from zen_renderer.dataloaders.dataset_amass import Dataset_AMASS
from zen_renderer.renderer.smpl_renderer import SMPL_Renderer
from zen_renderer.utils.transform_utils import *
from zen_renderer.utils.image_utils import *
import time
from multiprocessing import Pool


def render_amass_videos(job_list, image_size, gpu_index):
    
    device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    smpl_p = SMPL_Parser(device = device)
    smpl_renderer = SMPL_Renderer(device = device, image_size = image_size)
    # smpl_renderer.set_render_angles(3, 89, 0)
    
    for curr_pose, out_file_name in job_list:
        import time
        t_s = time.time()
        curr_pose = vertizalize_smpl_root(curr_pose)
        images, _ = smpl_renderer._render_pose_images(curr_pose, smpl_p)
        write_frames_to_video(images, out_file_name)
        
        dt = time.time() - t_s

        print(out_file_name, "Time {:.4f}".format(dt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seq_length', type=int, default=150)
    parser.add_argument('--out_dir', type=str, default=150)
    args = parser.parse_args()

    ''' confg '''
    out_dir = args.out_dir
    seq_length = args.seq_length
    video_out = os.path.join(out_dir, "videos")
    image_out = os.path.join(out_dir, "images")
    if not os.path.isdir(out_dir): os.makedirs(out_dir)
    if not os.path.isdir(image_out): os.makedirs(image_out)
    if not os.path.isdir(video_out): os.makedirs(video_out)

    dtype = torch.FloatTensor
    
    #################### AMASS Data split ####################
    
    amass_data = pk.load(open("/hdd/zen/data/ActBound/AMASS/amass_take3.pkl", "rb"))
    job_list = []
    num_poses = 0
    all_data = list(amass_data.items())
    for idx, (k, v) in enumerate(all_data):
        poses = v['pose']
        vid_id = k.split("_")[0]
        out_file_name = os.path.join(video_out, "{}.mp4".format(vid_id))
        curr_pose = convert_orth_6d_to_aa(torch.tensor(poses).float())
        job_list.append((curr_pose, out_file_name))

    print("Number of jobs: ", len(job_list))
    num_jobs = 10
    chunk = np.ceil(len(job_list)/num_jobs).astype(int)
    job_list= [job_list[i:i + chunk] for i in range(0, len(job_list), chunk)]
    for i, arg in enumerate(job_list):
        pk.dump(arg, open("/hdd/zen/data/ActBound/AMASS/Rendering/jobs/amass_{}.pk".format(i), "wb"))
    #################### actually Parsing it ####################