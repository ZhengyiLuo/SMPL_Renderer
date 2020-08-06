from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from skimage import io

import torch
from torch.utils.data import Dataset
import cv2
# ----------------------------------------------------------------
# class to access texture images as batch tensors
## TODO: implement female rendering
class TextureDataset(Dataset):
    def __init__(self, assets_dir, gender='male'):
        self.assets_dir = assets_dir
        self.texture_dir = os.path.join(self.assets_dir, 'textures', gender)
        self.textures = [os.path.join(self.texture_dir, x) for x in os.listdir(self.texture_dir) if x.startswith("nongrey")]
        # print(self.textures)
        return

    def __len__(self):
        return len(self.textures)

    def __getitem__(self, index):
        texture_image = cv2.imread(self.textures[index])
        return np.transpose(texture_image, (2,0,1))

    def random_sample(self, batch_size):
        images = []
        for i in range(batch_size):
            images.append(cv2.imread(np.random.choice(self.textures, replace=False)))
        images = np.array(images)
        return  np.transpose(images, (0, 3, 1, 2))




if __name__ == "__main__":
    pass