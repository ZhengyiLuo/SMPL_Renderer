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

from smpl_renderer.lib.smpl_parser import SMPL_49, SMPL_24


if __name__ == "__main__":
    SMPL_MODEL_DIR = "data/smpl_models"
    smpl = SMPL_49(
                model_path = SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False, 
            )
