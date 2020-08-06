import sys
import os
sys.path.append(os.getcwd())

import torch
import pickle as pk
import cv2
import numpy as np
import argparse

from zen_renderer.smpl_parser import SMPL_Parser
from zen_renderer.dataloaders.dataset_amass import Dataset_AMASS
from zen_renderer.renderer.smpl_renderer import SMPL_Renderer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    dtype = torch.FloatTensor

    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    smpl_p = SMPL_Parser(device = device) 
    smpl_render = SMPL_Renderer(device = device)
    smpl_render.set_render_angles(3, 150, 0)
    
    dsf_res_path = "/hdd/zen/dev/ActMix/actmix/DataGen/MotionSyn/gen.pk"
    dsf_ress = pk.load(open(dsf_res_path, "rb"))
    org_seq = dsf_ress["org_seq"]
    gen_seq =dsf_ress["gen_seq"]
    label = np.argmax(dsf_ress["label"], 1)
    gen_seq = gen_seq.reshape(org_seq.shape[0], gen_seq.shape[0]//org_seq.shape[0], org_seq.shape[1], org_seq.shape[2])
    
    print(org_seq.shape)
    print(gen_seq.shape)
    print(label.shape)

    for i in range(org_seq.shape[0]):
        org_seqs = org_seq[i]
        curr_label = label[i]
        for j in range(gen_seq.shape[1]):
            rec_seqs = gen_seq[i][j]
            # print(rec_seqs.shape)
            org_seqs[:, :3] = 0
            rec_seqs[:, :3] = 0
            org_seqs, rec_seqs = torch.tensor(org_seqs).to(device).type(dtype), torch.tensor(rec_seqs).to(device).type(dtype)
            org_verts, _ = smpl_p.get_vert_from_pose(org_seqs)
            rec_verts, _ = smpl_p.get_vert_from_pose(rec_seqs)

            org_images, _ = smpl_render._render_verts_seq(org_verts, random_texture=False, render_mask=False)
            org_images = (np.transpose(org_images.cpu().detach().numpy(), (0, 2,3,1)) * 256).astype(np.uint8)

            rec_images, _ = smpl_render._render_verts_seq(rec_verts, random_texture=False, render_mask=False)
            rec_images = (np.transpose(rec_images.cpu().detach().numpy(), (0, 2,3,1)) * 256).astype(np.uint8)

            y_shape, x_shape, _ = org_images[0].shape
            place_holder = np.zeros((y_shape, x_shape*2, 3)).astype(np.uint8)


            out = cv2.VideoWriter(os.path.join("output/vae/train_rec_{}_{}_lb:{}.mp4".format(i, j, curr_label)), cv2.VideoWriter_fourcc(*'FMP4'), 30, (x_shape*2, y_shape))
            for i in range(len(org_images)):
                place_holder[:y_shape,:x_shape, :] = org_images[i]
                place_holder[:y_shape,x_shape:, :] = rec_images[i]
                out.write(place_holder)
            out.release()
            print(curr_label)
            break