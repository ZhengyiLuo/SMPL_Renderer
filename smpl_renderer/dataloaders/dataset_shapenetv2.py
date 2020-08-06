import torch
from torch.utils.data import Dataset
import pickle as pk
import numpy as np

SHAPENET_ID = {'04379243': 'table', '03211117': 'monitor', '04401088': 'phone',
                   '04530566': 'watercraft', '03001627': 'chair', '03636649': 'lamp',
                   '03691459': 'speaker', '02828884': 'bench', '02691156': 'plane',
                   '02808440': 'bathtub', '02871439': 'bookcase', '02773838': 'bag',
                   '02801938': 'basket', '02880940': 'bowl', '02924116': 'bus',
                   '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car',
                   '03207941': 'dishwasher', '03337140': 'file', '03624134': 'knife',
                   '03642806': 'laptop', '03710193': 'mailbox', '03761084': 'microwave',
                   '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
                   '04004475': 'printer', '04099429': 'rocket', '04256520': 'sofa',
                   '04554684': 'washer', '04090263': 'rifle', '02946921': 'can'}

class Dataset_ShapenetV2(Dataset):
    def __init__(self, shapenet_base):
        
        return

    def __len__(self):
        return len(self.amass_data_list)

    def __getitem__(self, index):
        # load the texture image
        data_entry = self.amass_data_list[index]
        return data_entry
    
    def sequence_generator_by_class(self, class_name, batch_size = 1):
        


    def sequence_generator(self, batch_size=8):
        """[summary]
        
        Arguments:
            index {[type]} -- [description]
        """
        for i in range(self.seq_len // batch_size):
            pose_batch = []
            trans_batch = []
            for j in range(batch_size):
                data_dict = self.amass_data_list[i * batch_size + j][1]
                poses, trans = data_dict["poses"], data_dict["trans"]
                if self.fixed_length == None:
                    pose_batch.append(poses)
                    trans_batch.append(trans)
                else:
                    pose_batch.append(self.extract_random_fixed_length(poses, self.fixed_length))
                    trans_batch.append(self.extract_random_fixed_length(trans, self.fixed_length))
            pose_batch = np.array(pose_batch)
            trans_batch = np.array(trans_batch)
            yield pose_batch, trans_batch
    

    def extract_random_fixed_length(self, pose, fixed_length):
        total_length = pose.shape[0]
        random_start = np.random.randint(0, total_length - fixed_length)
        random_end = random_start + fixed_length
        return pose[random_start:random_end, :]
        

    def class_sequence_generator(self, index):
        """        
        
        Arguments:
            index {[type]} -- [description]
        """
        pass