import os
import sys
import pdb
sys.path.append(os.getcwd())
os.environ["KMP_WARNINGS"] = "FALSE" 

''' Libaray import '''
import gc
import pdb
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pyquaternion import Quaternion
import torchvision.transforms as T
from torch import optim
import torch.nn as nn
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import glob
import cv2
import os.path as osp
import time 
from multiprocessing import Pool
import pyrender
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import special_ortho_group

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# from lib.dataloaders.dataset_shapenetv2 import Dataset_ShapenetV2

def render_pose_data(model_path, num_frames, width = 400, height = 400, show = False):
    r = pyrender.OffscreenRenderer(viewport_width=width,viewport_height=height,point_size=1.0)
    fuze_trimesh = trimesh.load(model_path)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
            [1.0, 0,   0,  0],
            [0,  1.0, 0.0, 0],
            [0.0,  0,   1,   1.3],
            [0.0,  0.0, 0.0, 1.0],])

    if isinstance(fuze_trimesh, trimesh.scene.scene.Scene):
        scene = pyrender.Scene.from_trimesh_scene(fuze_trimesh)
        nodes = list(scene.nodes)
        for i in nodes:
            scene.remove_node(i)
        meshParent = pyrender.Node("MsehNode", children = nodes)
        scene.add_node(meshParent)
        mesh_vert = np.vstack([i.vertices for i in fuze_trimesh.dump()])

    else:
        scene = pyrender.Scene()
        meshParent = pyrender.Mesh.from_trimesh(fuze_trimesh)
        scene.add(meshParent)
        meshParent = list(scene.nodes)[0]
        mesh_vert = np.array(fuze_trimesh.vertices)

    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=20.0,
                                innerConeAngle=np.pi/6.0,
                                outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    transform = np.eye(4)
    

    import time 
    t_s = time.time()
    poses = []
    images = []
    depths = []
    for i in range(num_frames):
        x = special_ortho_group.rvs(3)
        transform[:3,:3] = x
        scene.set_pose(meshParent, transform)
        color, depth = r.render(scene)
        images.append(color)
        poses.append(np.copy(transform))
        depths.append(depth)
        if show:
            plt.imshow(color)
            plt.show()
    dt = time.time() - t_s
    
    print("Rendering Time used {:.4f}s".format(dt))
    
    
    return images, depths, poses, mesh_vert


def write_pose_data(output_path, model_id, images, depth, poses, mesh_vert):
    print("Writing data to...", output_path)
    data_output = osp.join(output_path,"data", model_id)
    model_output = osp.join(output_path, "models", model_id)
    
    os.makedirs(data_output, exist_ok=True)
    os.makedirs(model_output, exist_ok=True)
    for i in range(len(images)):
        name_base = os.path.join(data_output, "{:06d}".format(i))
        curr_image = images[i]
        curr_depth = depth[i]
        curr_pose = poses[i]
        cv2.imwrite(name_base + "-color.png", cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR))
        np.savez(name_base + "-meta.npz", pose = curr_pose, model_id = model_id)
        np.savez(name_base + "-depth.npz", depth = curr_depth)
        
    with open(os.path.join(model_output, "points.xyz"), "w+")  as f:
        for i in mesh_vert:
            f.write(' '.join([str(j) for j in i]) + "\n")

def gen_data_by_path(model_path, output_path, num_frames, img_size = 400):
    model_id = model_path.split("/")[-3]
    images, depth, poses, mesh_vert = render_pose_data(model_path, num_frames, width=img_size, height= img_size)
    write_pose_data(output_path, model_id, images, depth, poses, mesh_vert)
    del(images)
    del(depth)
    del(poses)
    del(mesh_vert)
    gc.collect()
    
def gen_data_by_pathlist(paths, output_path, num_frames, img_size = 400):
    for p in paths:
        gen_data_by_path(p, output_path, num_frames, img_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Superquad networks "
    )
    
    parser.add_argument("--output_dir",
                        help="Save the output files in that directory", default="model_output")
    parser.add_argument("--num_process", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=400)
    args = parser.parse_args()

    num_jobs = args.num_process
    num_frames = args.num_frames
    image_size = args.image_size

    shapenet_paths = glob.glob("/hdd/zen/data/Reallite/ShapeNetCore.v2/[0-9]*")
    shapnet_base = "/hdd/zen/data/Reallite/ShapeNetCore.v2/"
    cat_target = "03001627"
    output_path = args.output_dir

    target_paths = glob.glob(os.path.join(shapnet_base, cat_target, "*", "models", "model_normalized.obj"))

    ids = ["99d4e65625b32a1d151f08387c3363cb", "8a9af7d8a83d90fcd53e36731300f5b4", "357e2dd1512b96168e2b488ea5fa466a"]
    jobs = []
    for idx, model_path in enumerate(target_paths):
        model_id = model_path.split("/")[-3]
        if model_id in ids:
            jobs.append(model_path)

    # num_instances = 20
    # np.random.seed(1)
    # jobs = np.random.choice(jobs, num_instances, replace=False)
    
    chunk = np.ceil(len(jobs)/num_jobs).astype(int)
    jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    args = [(jobs[i], output_path, num_frames, image_size) for i in range(len(jobs))]
    print(len(args))
    try:
        pool = Pool(num_jobs)   # multi-processing
        pool.starmap(gen_data_by_pathlist, args)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

