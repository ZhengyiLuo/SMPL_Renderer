import os
import sys
sys.path.append(os.getcwd())

import torch
import pickle as pk
import cv2
import numpy as np
import argparse
import shutil
from PIL import Image

from zen_renderer.smpl_parser import SMPL_Parser
from zen_renderer.dataloaders.dataset_amass import Dataset_AMASS
from zen_renderer.renderer.smpl_renderer import SMPL_Renderer
from zen_renderer.utils.image_utils import *




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--seq_length', type=int, default=150)
    args = parser.parse_args()

    dtype = torch.FloatTensor
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

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

    ################## MOCAP poses ##################
    amass_dataset = Dataset_AMASS("/hdd/zen/data/ActBound/AMASS/amass.pkl", fixed_length=args.seq_length)
    seq_generator = amass_dataset.sequence_generator_by_class("walk")

    # genders = ["male", "female"]
    # angles = [89, 260]
    genders = ["female", "male"]
    angles = [240, 245, 250, 255, 260, 265]
    identities = []
    for gender in genders:
        for i in  np.linspace(-5, 5, 4):
            for j in  np.linspace(-5, 5, 4):
                for k in  np.linspace(-5, 5, 3):
                    identities.append((torch.tensor([i, j, k]), gender))
                    # break
                # break
            # break
    np.random.shuffle(identities)
    texture_fixed_ids = identities[:int(len(identities)/2)]
    texture_varies_ids = identities[int(len(identities)/2):]
    pose_num = 5

    if os.path.isdir("output/rid/same_cloth"):
        shutil.rmtree("output/rid/same_cloth")
    os.makedirs("output/rid/same_cloth")
    if os.path.isdir("output/rid/diff_cloth"):
        shutil.rmtree("output/rid/diff_cloth")
    os.makedirs("output/rid/diff_cloth")

    jobs = [(idt[0], idt[1], "same_cloth", False) for idt in texture_fixed_ids] + [(idt[0], idt[1], "diff_cloth", True) for idt in texture_varies_ids]
    for i in range(len(jobs)):
        p_id, gender, out_dir, texture_fix = jobs[i]
        pid_str = "".join(["{:03d}".format(i), str(gender)[0]])
        print("Generating for: ", pid_str)

        smpl_parser = SMPL_Parser(device = device, gender=gender)
        smpl_render = SMPL_Renderer(device = device, gender=gender)
        
        pose_list= []
        bg_list = []
        for a in range(len(angles)):
            smpl_render.set_render_angles(2.5, angles[a], 0)
            counter = 0
            for posese, trans in seq_generator:
                # print(pose.shape, trans.shape)
                posese, trans = torch.tensor(posese).to(device).type(dtype), torch.tensor(trans).to(device).type(dtype)
                pose_list.append(posese[0])
                counter += 1
                bg_list.append((a, angles[a]))
                if counter >= pose_num:
                    break
        id_all_poses = torch.cat(pose_list, dim = 0)
        
        shape_params = torch.rand(1, 10).repeat(id_all_poses.shape[0], 1)
        shape_params[:,:3] = p_id
        verts, Jtr = smpl_parser.get_vert_from_pose(id_all_poses, th_betas=shape_params)

        images, masks = smpl_render._render_verts_seq(verts, random_texture=texture_fix, render_mask=True)

        images = np.transpose(images.cpu().detach().numpy(), (0, 2,3,1))
        masks = np.transpose(masks.cpu().detach().numpy(), (0, 2,3,1))
        
        for j in range(len(images)):
            # print(np.max(images[i]), np.min(images[i]))
            img = (images[j] * 256).astype(np.uint8)
            bg_img = crop_center(cv2.imread(bg_imgs[bg_list[j][0]]), 256, 256)
            
            # pasting background by mask..... may not be good but oh well...
            img[masks[j] == 0] = bg_img[masks[j] == 0]
            img = cv2.GaussianBlur(img,(3,3),0)
            
            cv2.imwrite("output/rid/{}/P{}V{}frame{:06d}.png".format(out_dir, pid_str, bg_list[j][1], j), img)