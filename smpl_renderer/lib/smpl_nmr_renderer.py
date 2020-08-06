import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as sciR

import neural_renderer as nr
from zen_renderer.dataloaders.dataset_surreal_texture import TextureDataset
from zen_renderer.utils.transform_utils import *
from zen_renderer.smpl_parser import SMPL_Parser
from zen_renderer.utils.image_utils import assemble_videos

class SMPL_Renderer(object):
    def __init__(self, camera_mode='projection', K = None, R = None, t = None,\
        smpl_model_path = "/hdd/zen/dev/ActMix/actmix/DataGen/MotionRender/ActmixTorchGenerator/smpl_models/",\
        texture_path =  "/hdd/zen/data/SURREAL/smpl_data/", dtype = torch.float32, gender = "male", image_size = 256, \
        device =(torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")), \
            background_color=[0,0,0],
        ):

        self.dtype = dtype
        self.camera_mode = camera_mode
        self.image_size = image_size
            
        r = sciR.from_euler('zxy', [0, 0, 0], degrees = True)
        vet_rot = r.as_matrix()
        fx = 500.0
        fy = 500.0
        cx = 512.0
        cy = 512.0
        def_K = np.array( [ [fx, 0., cx],
                        [ 0. ,fx, cy],
                        [0.,0.,1.]])

        def_K = def_K[None, :]
        def_R = vet_rot[None, :]
        def_t = np.array([0,0,1.5])

        if K is None:
            K = def_K
        if R is None:
            R = def_R
        if t is None:
            t = def_t

        
        if camera_mode == "look_at":
            self.renderer = nr.Renderer(camera_mode='look_at', viewing_angle = 30, image_size = image_size, background_color= background_color)
            self.set_render_angles(2.5, 0, 0)
        elif camera_mode == "projection":
            self.renderer = nr.Renderer(K = K, R = R, t = t, camera_mode='projection', image_size = image_size, light_intensity_ambient=1, background_color= background_color)
        

        self.renderer = self.renderer.to(device)
        
        self.gender = gender
        
        self.faces = np.load(os.path.join(smpl_model_path, "smpl_faces.npy"))[np.newaxis, :,:]
        self.device = device
        self.faces_uv = np.load(os.path.join(texture_path, 'final_faces_uv_mapping.npy'))
        self.uv_sampler = torch.from_numpy(self.faces_uv.reshape(-1, 2, 2, 2)).to(device).type(self.dtype)
        self.uv_sampler = self.uv_sampler.view(-1, 13776, 2*2, 2)


        self.male_texture_dataset = TextureDataset(texture_path, "male")
        self.male_smpl_p = SMPL_Parser(device = device, gender = "male")

        self.female_texture_dataset = TextureDataset(texture_path, "female")
        self.female_smpl_p = SMPL_Parser(device = device, gender = "female")

        self.smpl_p = self.male_smpl_p if self.gender == 'male' else self.female_smpl_p
        self.texture_dataset = self.male_texture_dataset if self.gender == 'male' else self.female_texture_dataset

        # self.set_render_angles(2, 0, 0)

    def random_sex(self):
        self.gender = np.random.choice(['male', 'female'])
        self.smpl_p = self.male_smpl_p if self.gender == 'male' else self.female_smpl_p
        self.texture_dataset = self.male_texture_dataset if self.gender == 'male' else self.female_texture_dataset
        

    def set_render_angles(self, distance, elevation, azimuth):
        self.renderer.eye = nr.get_points_from_angles(distance, elevation, azimuth)

    def get_random_vect_root(self):
        rot = np.random.random() * np.pi * 2
        pitch =  np.random.random() * np.pi/3 + np.pi
        r = sciR.from_rotvec([pitch, 0, 0])
        r2 = sciR.from_rotvec([0, rot, 0])
        root_vec = (r * r2).as_rotvec()
        return root_vec

    def get_random_shape(self, batch_size):
        shape_params = torch.rand(1, 10).repeat(batch_size, 1)
        s_id = torch.tensor(np.random.normal(scale = 1.5, size = (3)))
        shape_params[:,:3] = s_id
        return shape_params


    def get_textures(self, batch_size, texture_idx = -1):
        if texture_idx != -1:
            texture_images = torch.tensor(self.texture_dataset[texture_idx]).to(self.device).type(self.dtype)
            texture_images = texture_images.repeat(batch_size, 1,1,1).to(self.device).type(self.dtype)
        else:
            texture_images = torch.tensor(self.texture_dataset.random_sample(batch_size)).to(self.device).type(self.dtype)
        
        uv_sampler = self.uv_sampler.repeat(batch_size, 1, 1, 1) ##torch.Size([B, 13776, 4, 2])

        # sample using the uv map from the texture image, the pose does not matter
        textures = torch.nn.functional.grid_sample(texture_images, uv_sampler, align_corners=True) #torch.Size([N, 3, 13776, 4])
        textures = textures.permute(0, 2, 3, 1) #torch.Size([N, 13776, 4, 3])
        textures = textures.view(-1, 13776, 2, 2, 3) #torch.Size([N, 13776, 2, 2, 3])

        # for the texture_size * texture_size -> texture_size * texture_size * texture_size
        textures = textures.unsqueeze(dim=4).repeat(1, 1, 1, 1, 2, 1)/256 ##torch.Size([N, 13776, 2, 2, 2, 3])
        return textures.to(self.device)


    def look_at_verts(self, poses):
        verts, joints, joints_36m = self.smpl_p.get_all_from_pose(poses)

        verts_lookat = nr.look_at(verts, self.renderer.eye)
        verts_proj = nr.perspective(verts_lookat)
        verts_proj *= 1024/5
        verts_proj += self.image_size/2

        joints_lookat = nr.look_at(joints_36m, self.renderer.eye)
        joints_proj = nr.perspective(joints_lookat)
        joints_proj *= 1024/5
        joints_proj += self.image_size/2
        return verts_proj, joints_proj, verts, joints, joints_36m



    def _render_verts_seq(self, verts, texture_idx = -1, render_mask = False):

        batch_size = verts.shape[0]
        faces = np.repeat(self.faces, batch_size, axis = 0)
        faces = torch.tensor(faces.astype(int)).to(self.device)
        textures = self.get_textures(batch_size, texture_idx= texture_idx)
        # images, image_depth, image_alpha = self.renderer(verts, faces, textures)  # [batch_size, RGB, image_size, image_size]

        images, image_depth, image_alpha = self.renderer(verts, faces, textures)  # [batch_size, RGB, image_size, image_size]
        masks = None
        if render_mask:
            white_textures = torch.ones(batch_size, faces.shape[1], 2, 2, 2, 3, dtype=self.dtype).to(self.device)
            masks, _, _ = self.renderer(verts, faces, white_textures)  # [batch_size, RGB, image_size, image_size]

        if self.camera_mode == "look_at":
            images = images[:,:,list(reversed(range(images.shape[2]))),:] # batch_size x channel x H x W

        if render_mask:
            return images, masks
        else:
            return images, None

    def _render_pose_images(self, poses, th_betas=None,
                th_trans=None, batch_texture = 1, render_mask = True, frame_chunk = 30):
        with torch.no_grad():
            poses = poses.type(self.dtype).to(self.device)
            th_trans = torch.zeros(poses.shape[0], 3).to(self.device)
            th_betas = torch.zeros(poses.shape[0], 10).to(self.device)

            verts, Jtr = self.smpl_p.get_vert_from_pose(poses, th_betas = th_betas, th_trans= th_trans)
            if verts.shape[0] < frame_chunk:
                frame_chunk = verts.shape[0]
            num_jobs = verts.shape[0] // frame_chunk
            
            chunk = np.ceil(len(verts)/num_jobs).astype(int)
            verts_chunks= [verts[i:i + chunk] for i in range(0, len(verts), chunk)]
            all_images = []
            all_masks = []

            if batch_texture == -2:
                texture_idx = np.random.randint(0, len(self.texture_dataset))
            else:
                texture_idx = batch_texture
                

            for vert_chunk in verts_chunks:
                images, masks = self._render_verts_seq(vert_chunk, texture_idx=texture_idx, render_mask=render_mask)
                images_np = np.transpose(images.cpu().detach().numpy(), (0, 2,3,1))
                all_images.append(images_np)

                if render_mask:
                    masks_np = np.transpose(masks.cpu().detach().numpy(), (0, 2,3,1))
                    all_masks.append(masks_np)

                del images
                torch.cuda.empty_cache()
                
            if render_mask:
                all_masks = np.vstack(all_masks)
            else:
                all_masks = None

            all_images = np.vstack(all_images)
            all_images = (all_images * 256).astype(np.uint8)
        return all_images, all_masks


    def render_pose_vid(self, poses, out_file_name = "output.mp4", th_betas=torch.zeros(1),th_trans=torch.zeros(1), \
                random_camera = 0, random_shape = True, frame_chunk = 30, random_sex= True, batch_texture = -2):

        images, masks = self.render_pose_img(poses,th_betas = th_betas, th_trans = th_trans,\
             random_camera = random_camera, random_shape = random_shape, frame_chunk = frame_chunk, random_sex = random_sex \
                 , batch_texture = batch_texture)
        x_shape, y_shape, _ = images[0].shape
        # Jank
        images[images == 0] = 255

        out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'FMP4'), 30, (x_shape, y_shape))
        for i in range(len(images)):
            out.write(images[i])
        out.release()

    
    def render_pose_img(self, poses, th_betas=torch.zeros(1),th_trans=torch.zeros(1), \
            random_camera = 0, random_shape = True, frame_chunk = 30, random_sex= True, batch_texture = -2):
        # Takes in a batch of poses
        if random_sex:
            self.random_sex()
    
        if poses.shape[1] == 144:
            poses = convert_orth_6d_to_aa(torch.tensor(poses, dtype = self.dtype)).to(self.device)
            

        if random_camera == 1:
            root_vec = self.get_random_vect_root().to(self.device)
            poses = vertizalize_smpl_root(poses, root_vec).to(self.device)
        elif random_camera == 2:
            poses = vertizalize_smpl_root(poses).to(self.device)
        
        if random_shape:
            th_betas = self.get_random_shape(poses.shape[0]).to(self.device)
        
        images, masks = self._render_pose_images(poses, th_betas = th_betas, frame_chunk= frame_chunk, th_trans= th_trans, batch_texture = batch_texture)

        # Jank
        images[images == 0] = 255

        return images, masks
