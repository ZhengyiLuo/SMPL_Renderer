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

def label_to_class(label):
    # label_map = pk.load(open("/hdd/zen/data/ActBound/AMASS/label_names.pk", "rb"))
    # idx = np.argmax(label) - 50
    # label_map = pk.load(open("/hdd/zen/data/VIBE_NTU/ntu_class_labels.pk", "rb"))
    # idx = np.argmax(label)
    label_map = {
        0: "fake",
        1: "real"
    }
    idx = np.argmax(label)
    return label_map[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--image_size', type=int, default=400)
    args = parser.parse_args()

    output_dir = args.output_dir
    image_size = args.image_size
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    dtype = torch.FloatTensor
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    smpl_render = SMPL_Renderer(device = device, image_size = image_size, camera_mode="look_at")
    
    vae_res_path = "/hdd/zen/dev/ActMix/actmix/DataGen/MotionSyn/res_rf.pk"
    
    vae_ress = pk.load(open(vae_res_path, "rb"))
    gt_seqs = vae_ress["gt_seq"]
    fk_seqs = vae_ress["fake_seq"]
    rec_seqs = vae_ress['rec']
    rec_rf_seqs = vae_ress["rec_rf"]
    labels = vae_ress["label"]

    print(gt_seqs.shape, rec_seqs.shape, labels.shape, rec_rf_seqs.shape)

    ################## Rendering #################
    with torch.no_grad():
        for k in range(gt_seqs.shape[0]):
            gt_seq, rec_seq, rec_rf_seq, fk_seq = gt_seqs[k], rec_seqs[k], rec_rf_seqs[k], fk_seqs[k]


            ''' Calcualte Sequence diffs'''
            gt_rec_diff = (np.linalg.norm((gt_seq - rec_seq).reshape(-1, 24, 6), axis = 2) **2 ).sum(axis = 1).mean()
            gt_fk_diff = (np.linalg.norm((gt_seq - fk_seq).reshape(-1, 24, 6), axis = 2) **2 ).sum(axis = 1).mean()
            gt_rec_rf_diff = (np.linalg.norm((gt_seq - rec_rf_seq).reshape(-1, 24, 6), axis = 2) **2 ).sum(axis = 1).mean()

            label = labels[k]
            label_name = label_to_class(label)

            gt_seq, rec_seq, rec_rf_seq, fk_seq = convert_orth_6d_to_aa(torch.tensor(gt_seq).float()), convert_orth_6d_to_aa(torch.tensor(rec_seq).float()), convert_orth_6d_to_aa(torch.tensor(rec_rf_seq).float()), convert_orth_6d_to_aa(torch.tensor(fk_seq).float())
            gt_seq, rec_seq, rec_rf_seq, fk_seq = vertizalize_smpl_root(gt_seq), vertizalize_smpl_root(rec_seq), vertizalize_smpl_root(rec_rf_seq), vertizalize_smpl_root(fk_seq)
            print("Rendering: ", k, label_name)
            
            # gt_seq, rec_seq, rec_rf_seq, fk_seq = torch.tensor(gt_seq).to(device).type(dtype), torch.tensor(rec_seq).to(device).type(dtype), torch.tensor(rec_rf_seq).to(device).type(dtype), torch.tensor(fk_seq).to(device).type(dtype)
            

            gt_images, _ = smpl_render.render_pose_img(gt_seq, random_shape=False)
            # rec_images, _ = smpl_render.render_pose_img(rec_seq, random_shape=False)
            f2r_images, _ = smpl_render.render_pose_img(rec_rf_seq, random_shape=False)
            fk_images, _ = smpl_render.render_pose_img(fk_seq, random_shape=False)

            grid_size = [1,3]
            videos = [gt_images, fk_images, f2r_images]
            descriptions = ["Real", "Fake", "Fake_to_Real"]
            output_name = "{}/rf_{}_{}.mp4".format(output_dir, label_name, k)
            assemble_videos(videos, grid_size, descriptions, output_name)
