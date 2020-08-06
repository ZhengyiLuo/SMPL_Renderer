import torch
from torch.utils.data import Dataset
import pickle as pk
import numpy as np
class Dataset_AMASS(Dataset):
    def __init__(self, amass_file_path, fixed_length = -1):
        print("******* Reading AMASS Data ***********")
        self.amass_data = pk.load(open(amass_file_path, "rb"))
        self.fixed_length = fixed_length
        if fixed_length == -1:
            self.amass_data_list = list(self.amass_data.items())
        else:
            self.amass_data_list = [i for i in list(self.amass_data.items()) if i[1]["poses"].shape[0] > self.fixed_length]
        self.seq_len = len(self.amass_data_list)

        
        print("Dataset Num Sequences: ", self.seq_len)
        print("Fixed Length: ", self.fixed_length)
        print("******* Finished Reading AMASS Data ***********")
        return

    def __len__(self):
        return len(self.amass_data_list)

    def __getitem__(self, index):
        # load the texture image
        data_entry = self.amass_data_list[index]
        return data_entry
    
    def sequence_generator_by_class(self, class_name, batch_size = 1):
        class_data_list = [i for i in self.amass_data_list if i[1]["poses"].shape[0] > self.fixed_length and class_name in i[1]["class"]]
        for i in range(self.seq_len // batch_size):
            pose_batch = []
            trans_batch = []
            for j in range(batch_size):
                data_dict = class_data_list[(i * batch_size + j) % len(class_data_list) ][1]
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