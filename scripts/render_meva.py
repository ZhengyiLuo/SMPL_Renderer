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
    out = cv2.VideoWriter(os.path.join("output/meva/{}.mp4".format(file_name)), cv2.VideoWriter_fourcc(*'FMP4'), 30, (x_shape, y_shape))
    for i in range(len(images)):
        out.write(( images[i] * 256).astype(np.uint8))
    out.release()

def label_to_class(label):
    label_map = pk.load(open("/hdd/zen/data/ActBound/AMASS/label_names.pk", "rb"))
    idx = np.argmax(label) - 50
    # label_map = pk.load(open("/hdd/zen/data/VIBE_NTU/ntu_class_labels.pk", "rb"))
    # idx = np.argmax(label)
    return label_map[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    dtype = torch.FloatTensor

    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    smpl_p = SMPL_Parser(device = device)
    smpl_renderer = SMPL_Renderer(device = device, image_size=512)
    smpl_renderer.set_render_angles(2, 89, 0)

    ################## Load BG images ##################
    # bg_base = "/hdd/zen/data/lsun/imgs/"
    # bg_imgs = [ os.path.join(bg_base, i) for i in os.listdir(bg_base) if i.endswith("jpg")]
    bg_imgs = [
        "/hdd/zen/data/MEVA/images_org/Vehicle_Stopping_82/frame000000.jpg", 
        "/hdd/zen/data/MEVA/images_org/Vehicle_U-Turn_40/frame000000.jpg", 
        "/hdd/zen/data/MEVA/images_org/Vehicle_U-Turn_0/frame000000.jpg", 
        "/hdd/zen/data/MEVA/images_org/Riding_1/frame000000.jpg", 
        "/hdd/zen/data/MEVA/images_org/Vehicle_Picks_Up_Person_11/frame000103.jpg", 
        "/hdd/zen/data/MEVA/images_org/Person_Sets_Down_Object_37/frame000000.jpg"
    ]

    amass_6d = pk.load(open("/hdd/zen/data/ActBound/AMASS/amass_ntu_6d.pkl", "rb"))
    
    for k, v in amass_6d.items():
        # curr_pose = v['pose'][:150, :]
        curr_pose = v['pose'][:30, :]
        label = v['label']
        if label > 50:
            print(label)
            # label_name = label_to_class(label)

            curr_pose = convert_orth_6d_to_aa(torch.tensor(curr_pose).float())    
            curr_pose = vertizalize_smpl_root(curr_pose)
            print("Rendering: ", k)
            
            curr_pose = torch.tensor(curr_pose).to(device).type(dtype)
            # curr_verts, _ = smpl_p.get_vert_from_pose(curr_pose)
            images, masks = smpl_renderer._render_pose_images(curr_pose, smpl_p, render_mask=True)
            
            # images, masks = smpl_renderer.render_verts_seq(curr_verts, random_texture=False, render_mask=True)
            # images_rgba = torch.cat((images, image_alpha.view(image_alpha.shape[0], 1, image_alpha.shape[1], image_alpha.shape[2])/256), 1)

            _,  x_shape, y_shape, _ = images.shape
            output_name = "output/meva/cnd_{}.mp4".format(label)
            
            print(output_name, x_shape, y_shape)

            out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'FMP4'), 30, (x_shape, y_shape))
            for i in range(len(images)):
                img = images[i]
                bg_img = crop_center(cv2.imread(bg_imgs[0]), x_shape, y_shape)
                img[masks[i] == 0] = bg_img[masks[i] == 0]
                img = cv2.GaussianBlur(img,(3,3),0)
                out.write(img)
            out.release()
            # break