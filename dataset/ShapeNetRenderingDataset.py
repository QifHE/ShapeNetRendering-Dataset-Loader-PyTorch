from re import I
import open3d as o3d
from skimage.util import dtype
import torch
import random
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import random
import trimesh
import sys
from os.path import join
from numpy import linalg as LA
import json
import pickle
from skimage import io,transform
from rich.progress import track

from utils.pcutils import normalize, make_holes_pcd_2, write_ply
from utils.utils import weights_init, visdom_show_pc, save_paths, save_model, vis_curve
from utils.metrics import AverageValueMeter
from utils.pcutils import mean_min_square_distance, save_point_cloud

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def rotate_pcd_shapeNet(pcd, posA = 1, posB = 2):
    n = pcd.shape[0]
    for i in range(n):
        temp = pcd[i][posA]
        pcd[i][posA] = pcd[i][posB]
        pcd[i][posB] = temp
    return pcd

shapenet_rendering_dir = "/Data/ShapeNetP2M/"
category_file = "dataset/synsetoffset2category.txt"

class ShapeNetDataset(data.Dataset):
    def __init__(self, root_dir = shapenet_rendering_dir, npoints = 2048, do_holes = True, function = None, category_choice = None, split = 'train', hole_size=0.35):
        self.npoints = npoints      # 原始点云点数量
        self.root = root_dir       # 数据集最上级文件夹
        self.classification = False
        self.normalize = normalize
        self.do_holes = do_holes     # 生成缺失点云（按照半径）
        self.hole_size = hole_size
        self.category = category_choice   # 数据集分类

        self.split = split        # 训练还是测试
        self.file_list = os.path.join("Data/", split + "_list.txt")

        self.category_file = category_file
        self.category_id = {} # 储存选择的类别id
        
        # 选择类别下的数据
        with open(self.category_file, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.category_id[ls[1]] = ls[0]
        if not category_choice is None:
            self.category_id = {v: k for k, v in self.category_id.items() if k in category_choice}
        
        #self.meta = {}  # 字典{类别：点云ID：视图}        
        for item in self.category_id:
            #self.meta[item] = []

            self.pkl_list = []
            #self.pc_list = {}
            #self.img_list = {}

            self.pc_data = []
            self.img_data = []

            # 获取dat列表
            with open(self.file_list, 'r') as f:
                while True:
                    line = f.readline().strip()             
                    if not line:
                        break
                    else:
                        cat = line.split("/")
                        if cat[2] == item:
                            self.pkl_list.append(line)
            
            # 从dat提取点云为list
            '''
            print("dat文件点云提取和图像预处理中……")
            for idx in track(range(len(self.pkl_list))):
                pkl_path = self.pkl_list[idx]
                points = pickle.load(open(pkl_path, 'rb'), encoding='bytes')
                points = [(points[i,0], points[i,1], points[i,2]) for i in range(len(points))]
                #self.pc_list[pc_id].append(points)
                
                self.pc_data.append(points)

                img_path = pkl_path.replace('.dat', '.png')

                
                img = io.imread(img_path)
                img[np.where(img[:,:,3]==0)] = 255
                img = transform.resize(img, (224,224))
                img = img[:,:,:3].astype('float32')
                #self.img_list[pc_id].append(img)
                
                img = 0
                self.img_data.append(img)

                #print("{} / {}".format(idx, len(self.pkl_list))) # 测试用

            #self.pc_meta[item].append(self.pc_list)  # self.meta 类别→点云id→点云数据(list)  
            #self.img_meta[item].append(self.img_list)  # self.img_meta 类别→点云id→点云图片  
                '''
    
    def __getitem__(self, index):
        
        pkl_path = self.pkl_list[index]
        points = pickle.load(open(pkl_path, 'rb'), encoding='bytes')
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(len(points))]
        #self.pc_list[pc_id].append(points)
        
        #self.pc_data.append(points)

        point_set = np.asarray(points, dtype = np.float32)
        #point_set = np.asarray(self.pc_data[index], dtype = np.float32)
        point_set = resample_pcd(point_set, self.npoints)
        #point_set = rotate_pcd_shapeNet(point_set)

        filename = self.pkl_list[index]
        #img_file = self.img_data[index]
        img_file = []

        if self.normalize:
            point_set = normalize(point_set, unit_ball = True)
        
        if self.do_holes:
            partial, hole = make_holes_pcd_2(point_set, hole_size=self.hole_size)
        else:
            partial = point_set
            hole = point_set
        
        return filename, img_file, resample_pcd(partial, self.npoints), resample_pcd(hole, self.npoints // 2), resample_pcd(point_set, self.npoints)
            
    def __len__(self):
        return len(self.pkl_list)

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA activated")
    torch.cuda.set_device(device)


    dataset_dir = "Data/ShapeNetP2M"
    category_choice = "airplane"
    dataset_train = ShapeNetDataset(root_dir=dataset_dir, category_choice=category_choice, npoints=2048, split='short')
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=2)

    output_dir = os.path.join("dataset_test_only/")

    for i, data in enumerate(dataloader_train, 0):
        filename, img, in_partial, in_hole, in_complete = data

        in_partial = in_partial.contiguous().float().to(device)
        in_hole = in_hole.contiguous().float().to(device)
        in_complete = in_complete.contiguous().float().to(device)

        gt = in_complete.cpu().numpy()
        partial = in_partial.cpu().numpy()
        hole = in_hole.cpu().numpy()

        pc_id = []
        view_id = []
        for item in filename:
            pc_id.append(item.split("/")[3])
            view_id.append(item.split("/")[5][:2])
        
        for j in range(len(pc_id)):
            pc_folder = os.path.join(output_dir, category_choice, pc_id[j])
            if not os.path.exists(pc_folder):
                os.makedirs(pc_folder)

            write_ply(gt[j], os.path.join(pc_folder, view_id[j] + '_gt.ply'))
            write_ply(partial[j], os.path.join(pc_folder, view_id[j] + '_partial.ply'))
            write_ply(hole[j], os.path.join(pc_folder, view_id[j] + '_hole.ply'))