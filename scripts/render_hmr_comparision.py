import os
import sys
sys.path.append(os.getcwd())

import torch
import pickle as pk
import cv2
import numpy as np
import argparse
from glob import glob
import os.path as osp

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


def render_pose_vid_trans(poses, trans, smpl_p, smpl_renderer, file_name):
    poses = torch.tensor(poses).to(device).type(dtype)
    trans = torch.tensor(trans).to(device).type(dtype)
    verts, Jtr = smpl_p.get_vert_from_pose(poses, th_trans = trans)
    images, _ = smpl_renderer.render_verts_seq(verts)
    images = np.transpose(images.cpu().detach().numpy(), (0, 2,3,1))
    write_frames_to_video(images, os.path.join("output/smpl/{}.mp4".format(file_name)))

def render_compare_video(smpl_renderer, vibe_data_path, hmmr_path, vid_path, ouput_dir, smpl_img_size = 512):
    print(vid_name)
    ##################### VIBE ####################
    import joblib
    ntu_data = jobzen_renderer.load(open(vibe_data_path, "rb"))
    key = list(ntu_data.keys())[0]
    pose = ntu_data[key]['pose']
    pose = vertizalize_smpl_root(torch.tensor(pose).double())
    vibe_images, _ = smpl_renderer.render_pose_images(pose, smpl_p, frame_chunk=30)

    # ##################### HMMR ####################
    if not os.path.isfile(hmmr_path):
        print("No hmmr file ", hmmr_path)
        return
    hmmr_data = pk.load(open(hmmr_path, "rb"))
    pose = smpl_mat_to_aa(hmmr_data["poses"]).reshape(-1, 72)
    pose = vertizalize_smpl_root(torch.tensor(pose).double())
    hmmr_images, _ = smpl_renderer.render_pose_images(pose, smpl_p, frame_chunk=30)

    
    ##################### Original Video ####################
    frames = read_video_frames(vid_path)
    print(len(frames), len(hmmr_images), len(vibe_images))
    y_shape, x_shape, _ = frames[0].shape
    target_x_shape = np.floor(x_shape/y_shape * smpl_img_size).astype(int)
    target_y_shape = smpl_img_size 
    resized_org_frames = resize_frames(frames, size_x  = target_x_shape, size_y = target_y_shape)
    output_frames = []

    # canvas = np.zeros((target_y_shape, target_x_shape + smpl_img_size, 3)).astype(np.uint8)
    # num_min_frames = min(len(frames), len(hmmr_images), len(vibe_images))
    # for i in range(num_min_frames):
    #     orig_img = resized_org_frames[i]
    #     vibe_img= vibe_images[i]
    #     hmmr_img = hmmr_images[i]
    #     canvas[:target_y_shape, :target_x_shape, :] = orig_img
    #     canvas[:smpl_img_size, target_x_shape:, :] = hmmr_img
    #     canvas[smpl_img_size:, target_x_shape:, :] = vibe_img

    #     color = (255, 255, 255)
    #     cv2.putText(canvas,"HMMR", (target_x_shape, 20), 2, 1, color)
    #     cv2.putText(canvas,"VIBE", (target_x_shape, smpl_img_size), 2, 1, color)

    #     output_frames.append(np.copy(canvas))
    # # import pdb       
    # # pdb.set_trace()
    # write_frames_to_video(output_frames, "output/{}/{}.mp4".format(base_path, vid_name[:-4]))
    # torch.cuda.empty_cache()

    canvas = np.zeros((target_y_shape, target_x_shape + smpl_img_size * 2, 3)).astype(np.uint8)
    num_min_frames = min(len(frames), len(hmmr_images), len(vibe_images))
    for i in range(num_min_frames):
        orig_img = resized_org_frames[i]
        vibe_img = vibe_images[i]
        hmmr_img = hmmr_images[i]
        canvas[:target_y_shape, :target_x_shape, :] = orig_img
        canvas[:target_y_shape,  target_x_shape:(target_x_shape + smpl_img_size), :] = hmmr_img
        canvas[:target_y_shape, (target_x_shape + smpl_img_size):, :] = vibe_img

        color = (255, 255, 255)
        cv2.putText(canvas,"AMASS", (0, 20), 2, 1, color)
        cv2.putText(canvas,"HMMR", (target_x_shape, 20), 2, 1, color)
        cv2.putText(canvas,"VIBE", (target_x_shape + smpl_img_size,20), 2, 1, color)

        output_frames.append(np.copy(canvas))
    # import pdb       
    # pdb.set_trace()
    write_frames_to_video(output_frames, ouput_dir)
    torch.cuda.empty_cache()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    dtype = torch.FloatTensor

    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    smpl_p = SMPL_Parser(device = device)
    smpl_renderer = SMPL_Renderer(device = device, image_size=512)
    smpl_renderer.set_render_angles(3, 89, 0)
    
    
    ################## NTU ##################
    # from glob import glob
    # class_num = "A009"
    # vid_names = glob("/hdd/zen/data/HMMR/results/ntu/{}/*.avi".format(class_num))
    # for vid_name in vid_names:
    #     vid_name = vid_name.split("/")[-1][:-4]
    #     vibe_data_path = "/hdd/zen/data/VIBE_NTU/{}/{}.pkl".format(class_num, vid_name)
    #     hmmr_path = "/hdd/zen/data/HMMR/results/ntu/{}/{}.avi/hmmr_output/hmmr_output.pkl".format(class_num, vid_name)
    #     vid_path = "/hdd/zen/data/NTU/videos/{}/{}.avi".format(class_num, vid_name)
    #     render_compare_video(vibe_data_path, hmmr_path, vid_path)
    
    ################## Self vids ##################
    # vid_name = "IMG_1554.mp4"
    # vibe_data_path = "/hdd/zen/dev/VIBE/output/{}/vibe_output.pkl".format(vid_name[:-4])
    # hmmr_path = "/hdd/zen/data/HMMR/results/rz_talk_phone/{}/hmmr_output/hmmr_output.pkl".format(vid_name)
    # vid_path = "/hdd/zen/data/HMMR/raw_videos/rz_talk_phone/{}".format(vid_name)
    # render_compare_video(smpl_renderer, vibe_data_path, hmmr_path, vid_path)

    ################## HMR 2 HMR vids ##################
    amass_input = "/hdd/zen/data/ActBound/AMASS/Rendering/take2/videos/"
    amass_videos = glob(osp.join(amass_input, "*"))
    choices = np.random.choice(amass_videos, 10)
    ouput_base = "/hdd/zen/data/ActmixGenenerator/output/hmr_compare/"
    for i in choices:
        vid_name = i.split("/")[-1]
        vibe_data_path = "/hdd/zen/data/VIBE/VIBE_AMASS/take2/{}.pkl".format(vid_name[:-4])
        hmmr_path = "/hdd/zen/data/HMMR/results/amass/take2/{}/hmmr_output/hmmr_output.pkl".format(vid_name)
        vid_path = i
        # print(vibe_data_path, vibe_data_path, hmmr_path)
        render_compare_video(smpl_renderer, vibe_data_path, hmmr_path, vid_path, osp.join(ouput_base, vid_name))