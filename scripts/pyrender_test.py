import pyrender
import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import ortho_group

fuze_trimesh = trimesh.load('/hdd/zen/data/Reallite/ShapeNetCore.v2/02958343/7d4fd8ed77355364fa98472c1d231070/models/model_normalized.obj')


scene = pyrender.Scene.from_trimesh_scene(fuze_trimesh)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
s = np.sqrt(2)/2
camera_pose = np.array([
        [1.0, 0,   0,  0],
        [0,  1.0, 0.0, 0],
        [0.0,  0,   1,   1],
        [0.0,  0.0, 0.0, 1.0],])
# camera_pose[:3,:3] = curr_rot.as_matrix()
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=5.0,
                            innerConeAngle=np.pi/6.0,
                            outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
# color, depth = r.render(scene)
# plt.imshow(color)
# plt.show()
# print(transform33)
# print(transform)
pyrender.Viewer(scene, use_raymond_lighting=True)