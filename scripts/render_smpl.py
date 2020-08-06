import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch
import pickle as pk
import cv2
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as sciR

from zen_renderer.smpl_parser import SMPL_Parser
from zen_renderer.dataloaders.dataset_amass import Dataset_AMASS
from zen_renderer.renderer.smpl_renderer import SMPL_Renderer
from zen_renderer.utils.transform_utils import *


def render_pose_vid_trans(poses, trans, smpl_p, smpl_renderer, file_name):
    poses = torch.tensor(poses).to(device).type(dtype)
    trans = torch.tensor(trans).to(device).type(dtype)
    verts, Jtr = smpl_p.get_vert_from_pose(poses, th_trans = trans)
    images, _ = smpl_renderer.render_verts_seq(verts)
    images = np.transpose(images.cpu().detach().numpy(), (0, 2,3,1))
    y_shape, x_shape, _ = images[0].shape
    out = cv2.VideoWriter(os.path.join("output/smpl/{}.mp4".format(file_name)), cv2.VideoWriter_fourcc(*'FMP4'), 30, (x_shape, y_shape))
    for i in range(len(images)):
        out.write(( images[i] * 256).astype(np.uint8))
    out.release()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seq_length', type=int, default=150)

    args = parser.parse_args()
    image_size = args.image_size
    dtype = torch.FloatTensor

    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

    #################### PKL ####################
    # data = pk.load(open("/hdd/zen/dev/ActMix/actmix/DataGen/MotionSyn/smpl_vis.pk", "rb"))
    # output_dir = "/hdd/zen/data/ActmixGenenerator/output/rf/vae_vis" 
    # if not os.path.isdir(output_dir): os.makedirs(output_dir)
    # for k, v in data:
    #     poses = v
    #     poses = convert_orth_6d_to_aa(torch.tensor(poses).float())
    #     curr_pose = vertizalize_smpl_root(poses)
    #     smpl_renderer.render_pose_vid(curr_pose, smpl_p,  os.path.join(output_dir, k + ".mp4"))

    #################### AMASS ####################
    # amass_dataset = Dataset_AMASS("/hdd/zen/data/ActBound/AMASS/amass.pkl", fixed_length=args.seq_length)
    # seq_generator = amass_dataset.sequence_generator(batch_size=1)
    # # print(amass_dataset[0][0]["trans"])
    # pose_list = []
    # num_poses = 0
    # for poses, trans in seq_generator:
    #     # print(pose.shape, trans.shape)
    #     curr_pose = vertizalize_smpl_root(poses[0])
    #     smpl_renderer.render_pose_vid(curr_pose, smpl_p,  "ouput/smpl/test.mp4")

    #     num_poses += 1
    #     if num_poses > 10:
    #         break
        
    #################### AMASS 6d ####################
    
    amass_6d = pk.load(open("/hdd/zen/data/ActBound/AMASS/amass_take3.pkl", "rb"))
    smpl_renderer = SMPL_Renderer(camera_mode="look_at", device = device, image_size = image_size)
    for idx, (k, v) in enumerate(amass_6d.items()):
        curr_pose = v['pose']
        curr_pose = convert_orth_6d_to_aa(torch.tensor(curr_pose).float())
        output_path = "/hdd/zen/data/ActmixGenenerator/output/amass/test{}.mp4".format(idx)
        smpl_renderer.render_pose_vid(curr_pose, output_path, random_camera=False, random_shape=True)

    #################### VIBE ####################
    # import joblib
    # ntu_data_path = "/hdd/zen/data/VIBE_NTU/A029_rnd/S008C001P033R001A029_rgb.pkl"
    # ntu_data = jobzen_renderer.load(open(ntu_data_path, "rb"))
    # key = list(ntu_data.keys())[0]
    # pose = ntu_data[key]['pose']
    # # pose[:, :3] = np.tile([1.20919958, -1.20919958, -1.20919958], [pose.shape[0], 1])
    # pose = vertizalize_smpl_root(torch.tensor(pose).double())
    # # pose[:, :3] = np.tile([-1.74393425e-16, -2.22144147e+00, -2.22144147e+00], [pose.shape[0], 1])
    # render_pose_vid(pose, smpl_p, smpl_renderer, "vibe")

    #################### VIBE Dataset ####################
    # ntu_data_path = "/hdd/zen/data/VIBE_NTU/train_daily_vibe_ntu.pkl"
    # ntu_data = pk.load(open(ntu_data_path, "rb"))
    # key = list(ntu_data.keys())[0]
    # pose = ntu_data[key]['pose']
    # label = ntu_data[key]['label']
    # print(label)
    # pose[:, :3] = 0
    # render_pose_vid(pose, smpl_p, smpl_renderer, "vibe-ntu")

    # #################### HMMR ####################
    # hmmr_path = "/hdd/zen/data/HMMR/results/ntu/A029/S008C003P032R002A029_rgb.avi/hmmr_output/hmmr_output.pkl"
    # hmmr_data = pk.load(open(hmmr_path, "rb"))
    # pose = smpl_mat_to_aa(hmmr_data["poses"]).reshape(-1, 72)
    # # pose[:, :3] = np.tile([-1.74393425e-16, -2.22144147e+00, -2.22144147e+00], [pose.shape[0], 1])
    # pose = vertizalize_smpl_root(torch.tensor(pose).double())
    # render_pose_vid(pose, smpl_p, smpl_renderer, "hmmr")

    #################### HMMR ####################
    # aug_data_gen = "/hdd/zen/data/ActBound/Aug/gen_aug_1.pk"
    # aug_data = pk.load(open(aug_data_gen, "rb"))
    # gen_id = "gen_7002"
    # pose = aug_data[gen_id]["pose"]
    # label = aug_data[gen_id]["label"]
    # print(pose.shape, label)
    # pose[:, :3] = 0
    # render_pose_vid(pose, smpl_p, smpl_renderer, "gen_aug_1")
    