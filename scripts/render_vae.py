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

def label_to_class(label):
    # label_map = pk.load(open("/hdd/zen/data/ActBound/AMASS/label_names.pk", "rb"))
    label_map = pk.load(open("/hdd/zen/data/VIBE_NTU/ntu_class_labels.pk", "rb"))
    idx = np.argmax(label)
    return label_map[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    dtype = torch.FloatTensor

    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    smpl_p = SMPL_Parser(device = device) 
    smpl_render = SMPL_Renderer(device = device)
    smpl_render.set_render_angles(3, 150, 0)
    
    vae_res_path = "/hdd/zen/dev/ActMix/actmix/DataGen/MotionSyn/res.pk"
    
    vae_ress = pk.load(open(vae_res_path, "rb"))
    org_seqs = vae_ress["org_seq"]
    rec_seqs = vae_ress["rec_seq"]
    labels = vae_ress["label"]
    print(org_seqs.shape, rec_seqs.shape, labels.shape)
    
    # print(amass_dataset[0][0]["trans"])
    
    for k in range(org_seqs.shape[0]):
        org_seq = org_seqs[k]
        rec_seq = rec_seqs[k]
        label = labels[k]
        label_name = label_to_class(label)
        org_seq[:, :3] = 0
        rec_seq[:, :3] = 0
        print("Rendering: ", k, label_name)
        
        org_seq, rec_seq = torch.tensor(org_seq).to(device).type(dtype), torch.tensor(rec_seq).to(device).type(dtype)
        org_verts, _ = smpl_p.get_vert_from_pose(org_seq)
        rec_verts, _ = smpl_p.get_vert_from_pose(rec_seq)

        org_images, _ = smpl_render._render_verts_seq(org_verts, random_texture=False, render_mask=False)
        org_images = (np.transpose(org_images.cpu().detach().numpy(), (0, 2,3,1)) * 256).astype(np.uint8)

        rec_images, _ = smpl_render._render_verts_seq(rec_verts, random_texture=False, render_mask=False)
        rec_images = (np.transpose(rec_images.cpu().detach().numpy(), (0, 2,3,1)) * 256).astype(np.uint8)

        y_shape, x_shape, _ = org_images[0].shape
        place_holder = np.zeros((y_shape, x_shape*2, 3)).astype(np.uint8)


        out = cv2.VideoWriter(os.path.join("output/vae/cnd_{}_{}.mp4".format(label_name, k)), cv2.VideoWriter_fourcc(*'FMP4'), 30, (x_shape*2, y_shape))
        for i in range(len(org_images)):
            place_holder[:y_shape,:x_shape, :] = org_images[i]
            place_holder[:y_shape,x_shape:, :] = rec_images[i]
            out.write(place_holder)
        out.release()
        # break