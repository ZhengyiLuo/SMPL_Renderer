import os
import sys
sys.path.append(os.getcwd())

import torch
import pickle as pk
import cv2
import numpy as np
import argparse
from PIL import Image
from PIL import ImageDraw

from zen_renderer.smpl_parser import SMPL_Parser
from zen_renderer.dataloaders.dataset_amass import Dataset_AMASS
from zen_renderer.renderer.smpl_renderer import SMPL_Renderer
from zen_renderer.utils.transform_utils import *
from zen_renderer.utils.image_utils import *


def render_pose_img(poses, smpl_p, smpl_renderer, file_name):
    poses = torch.tensor(poses).to(device).type(dtype)
    verts, Jtr = smpl_p.get_vert_from_pose(poses)
    images, _ = smpl_renderer.render_verts_seq(verts)
    images = np.transpose(images.cpu().detach().numpy(), (0, 2,3,1))
    for i in range(len(images)):
        cv2.imwrite("test/{}_{:06d}.png".format(file_name,i), images[i])

def render_pose_vid(poses, smpl_p, smpl_renderer, file_name):
    poses = torch.tensor(poses).to(device).type(dtype)
    verts, Jtr = smpl_p.get_vert_from_pose(poses)
    images, _ = smpl_renderer.render_verts_seq(verts)
    images = np.transpose(images.cpu().detach().numpy(), (0, 2,3,1))
    y_shape, x_shape, _ = images[0].shape
    out = cv2.VideoWriter(os.path.join("output/smpl/{}.mp4".format(file_name)), cv2.VideoWriter_fourcc(*'FMP4'), 30, (x_shape, y_shape))
    for i in range(len(images)):
        out.write(( images[i] * 256).astype(np.uint8))
    out.release()

def label_to_class(label):
    # label_map = pk.load(open("/hdd/zen/data/ActBound/AMASS/label_names.pk", "rb"))
    # idx = np.argmax(label) - 50
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
    smpl_renderer = SMPL_Renderer(device = device)
    smpl_renderer.set_render_angles(2, 89, 0)

    ################## Load BG images ##################
    # bg_base = "/hdd/zen/data/lsun/imgs/"
    # bg_imgs = [ os.path.join(bg_base, i) for i in os.listdir(bg_base) if i.endswith("jpg")]
    bg_imgs = [
        "/hdd/zen/data/NTU/images/S010C002P025R001A030_rgb/frame000001.jpg", 
        "/hdd/zen/data/NTU/images/S012C003P017R001A048_rgb/frame000001.jpg", 
        "/hdd/zen/data/NTU/images/S006C003P007R001A041_rgb/frame000001.jpg", 
    ]

    data_path = "/hdd/zen/data/ActBound/Aug/gen_aug_6d.pk"
    
    pose_data = pk.load(open(data_path, "rb"))
    for k, v in pose_data.items():
        curr_pose = v['pose']
        label = v['label']
        label_name = label_to_class(label)

        curr_pose = convert_orth_6d_to_aa(torch.tensor(curr_pose).float())    
        curr_pose = vertizalize_smpl_root(curr_pose)
        print("Rendering: ", k, label_name)
        
        curr_pose = torch.tensor(curr_pose).to(device).type(dtype)
        curr_verts, _ = smpl_p.get_vert_from_pose(curr_pose)

        images, masks = smpl_renderer._render_verts_seq(curr_verts, random_texture=False, render_mask=True)
        # images_rgba = torch.cat((images, image_alpha.view(image_alpha.shape[0], 1, image_alpha.shape[1], image_alpha.shape[2])/256), 1)
        images = (np.transpose(images.cpu().detach().numpy(), (0, 2,3,1)) * 256).astype(np.uint8)
        masks = np.transpose(masks.cpu().detach().numpy(), (0, 2,3,1))

        _,  x_shape, y_shape, _ = images.shape
        output_name = "output/vae/cnd_{}_{}.mp4".format(label_name, k)
        
        print(output_name, x_shape, y_shape)

        out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'FMP4'), 30, (x_shape, y_shape))
        for i in range(len(images)):
            img = images[i]
            bg_img = crop_side(cv2.imread(bg_imgs[0]), 256, 256)

            img[masks[i] == 0] = bg_img[masks[i] == 0]
            img = cv2.GaussianBlur(img,(3,3),0)
            out.write(img)
        out.release()
        # break