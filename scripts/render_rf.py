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
    
    vae_res_path = "/hdd/zen/dev/ActMix/actmix/DataGen/MotionSyn/vis_rf.pk"
    
    vae_ress = pk.load(open(vae_res_path, "rb"))
    gt_seqs = vae_ress["gt_seq"]
    fk_seqs = vae_ress["fake_seq"]
    rec_rf_seqs = vae_ress["rec_rf"]
    labels = vae_ress["label"]

    print(gt_seqs.shape, labels.shape, rec_rf_seqs.shape)

    ################## Rendering #################
    with torch.no_grad():
        for k in range(gt_seqs.shape[0]):
            gt_seq,  rec_rf_seq, fk_seq = gt_seqs[k],  rec_rf_seqs[k], fk_seqs[k]


            ''' Calcualte Sequence diffs'''
            gt_fk_diff = (np.linalg.norm((gt_seq - fk_seq).reshape(-1, 24, 6), axis = 2) **2 ).sum()
            gt_rec_rf_diff = (np.linalg.norm((gt_seq - rec_rf_seq).reshape(-1, 24, 6), axis = 2) **2 ).sum()

            label = labels[k]
            label_name = label_to_class(label)

            gt_seq, rec_rf_seq, fk_seq = convert_orth_6d_to_aa(torch.tensor(gt_seq).float()),  convert_orth_6d_to_aa(torch.tensor(rec_rf_seq).float()), convert_orth_6d_to_aa(torch.tensor(fk_seq).float())
            gt_seq, rec_rf_seq, fk_seq = vertizalize_smpl_root(gt_seq), vertizalize_smpl_root(rec_rf_seq), vertizalize_smpl_root(fk_seq)
            print("Rendering: ", k, label_name)
            

            gt_images, _ = smpl_render.render_pose_img(gt_seq, random_shape=False)
            rec_rf_images, _ = smpl_render.render_pose_img(rec_rf_seq, random_shape=False)
            fk_images, _ = smpl_render.render_pose_img(fk_seq, random_shape=False)

            grid_size = [1,3]
            videos = [gt_images, rec_rf_images, fk_images]
            # descriptions = ["Real", "TCN  L2_vs_gt: {:.4f}".format(gt_rec_rf_diff), "Fake L2_vs_gt: {:.4f}".format(gt_fk_diff)]
            descriptions = ["Real", "Fake", "Real_to_fake"]
            output_name = "{}/rf_{}_{}.mp4".format(output_dir, label_name, k)
            assemble_videos(videos, grid_size, descriptions, output_name)

            # y_shape, x_shape, _ = gt_images[0].shape
            # canvas = np.zeros((y_shape, x_shape * 3, 3)).astype(np.uint8)

            # output_name = "{}/rf_{}_{}.mp4".format(output_dir, label_name, k)
            # print(output_name)
            # out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'FMP4'), 30, (x_shape*3, y_shape))
            # color = (255, 255, 255)
            # for i in range(len(gt_images)):
            #     canvas[:y_shape,:x_shape, :] = gt_images[i]
            #     canvas[:y_shape,x_shape:x_shape*2, :] = rec_rf_images[i]
            #     canvas[:y_shape,x_shape*2:x_shape*3, :] =   fk_rf_images[i]


            #     cv2.putText(canvas, "AMASS", (0, 20), 2, 0.5, color)
            #     cv2.putText(canvas, "TCN  L2_vs_gt: {:.4f}".format(gt_rec_rf_diff) , (x_shape, 20), 2, 0.5, color)
            #     cv2.putText(canvas, "Fake L2_vs_gt: {:.4f}".format(gt_fk_diff), (x_shape* 2,20), 2, 0.5, color)

            #     out.write(canvas)
            # out.release()
            # # brea```