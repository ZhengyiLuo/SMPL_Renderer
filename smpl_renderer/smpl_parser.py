import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from smplx.lbs import vertices2joints
import torch
import numpy as np

from zen_renderer.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from smplx.vertex_ids import vertex_ids as VERTEX_IDS
from smplx.vertex_joint_selector import VertexJointSelector

# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]

SMPL_DATA_DIR = "/hdd/zen/dev/ActMix/actmix/DataGen/MotionRender/ActmixTorchGenerator/smpl_models"
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}
JOINT_REGRESSOR_TRAIN_EXTRA = osp.join(SMPL_DATA_DIR, 'J_regressor_extra.npy')
JOINT_REGRESSOR = osp.join(SMPL_DATA_DIR, 'J_regressor_h36m.npy')
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]



class SMPL_Parser:
    def __init__(self, gender = "male", model_root = SMPL_DATA_DIR, \
        device =(torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))):
        self.smpl_layer = SMPL_Layer(
            center_idx=0,
            gender=gender,
            model_root=model_root, 
            device =  device)
        
        self.smpl_layer = self.smpl_layer.to(device)
        self.device = device
        self.J_regressor_extra = torch.tensor(np.load(JOINT_REGRESSOR_TRAIN_EXTRA)).float().to(device)
        self.J_regressor = torch.tensor(np.load(JOINT_REGRESSOR)).float().to(device)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.joint_map = torch.tensor(joints, dtype=torch.long).to(self.device)
        vertex_ids = VERTEX_IDS['smplh']
        self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids).to(self.device)


    
    def get_all_from_pose(self, poses, regress = True, th_betas=torch.zeros(1),
                th_trans=torch.zeros(1)):
        
        verts, Jtr = self.get_vert_from_pose(poses,th_betas= th_betas, th_trans=th_trans)
        Jtr = self.vertex_joint_selector(verts, Jtr)

        extra_joints = vertices2joints(self.J_regressor_extra, verts).to(self.device)
        joints = torch.cat([Jtr, extra_joints], dim=1).to(self.device)
        # print(joints.shape, self.joint_map, extra_joints.shape)
        joints = joints[:, self.joint_map, :]

        if regress:
            J_regressor_batch = self.J_regressor[None, :].expand(verts.shape[0], -1, -1).to(self.device)
            joints_36m = torch.matmul(J_regressor_batch, verts)
        # Joints has 49 joints, Joints_36m has 17 joints
        return verts.to(self.device), joints.to(self.device), joints_36m.to(self.device)



    def get_vert_from_pose(self, poses, th_betas=torch.zeros(1),
                th_trans=torch.zeros(1)):
        """[summary]
            Given SMPL Poses, generate SMPL vertices and joint locations
        Arguments:
            poses {[type]} -- [description]
        """
        poses_flat = poses.reshape(-1, 72)

        # verts, Jtr = self.smpl_layer(poses_flat, th_betas=th_betas.to(self.device), th_trans = th_trans.to(self.device))
        chunk_size = 300
        if poses_flat.shape[0] < chunk_size:
            verts, Jtr = self.smpl_layer(poses_flat, th_betas=th_betas.to(self.device), th_trans = th_trans.to(self.device))
        else:
            poses_flat_chunks = torch.split(poses_flat, int(poses_flat.shape[0]/chunk_size), dim=0)
            vert_list = []
            Jtr_list = []
            for pose_flat_c in poses_flat_chunks:
                verts_c, Jtr_c = self.smpl_layer(pose_flat_c, th_betas=th_betas.to(self.device), th_trans = th_trans.to(self.device))
                vert_list.append(verts_c)
                Jtr_list.append(Jtr_c)

            verts = torch.cat(vert_list, dim = 0)
            Jtr = torch.cat(Jtr_list, dim = 0)


        shape_curr = tuple([int(i) for i in (list(poses.shape)[:-1] + list(verts.shape[-2:]))])
        verts = verts.reshape(shape_curr)

        shape_curr = tuple([int(i) for i in (list(poses.shape)[:-1] + list(Jtr.shape[-2:]))])
        Jtr = Jtr.reshape(shape_curr)

        return verts.to(self.device), Jtr.to(self.device)
