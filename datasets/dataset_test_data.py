import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from natsort import natsorted
import re
import cv2


class NOT156(Dataset):
    def __init__(self, root_dir, seq_len=1, overlap=0):
        super(NOT156, self).__init__()
        self.seq_len = seq_len
        self.step = seq_len - overlap
        self.transform = transforms.Compose([transforms.ToTensor()]) 
        self.index_list = [] 
        self.root_dir = os.path.join(root_dir, 'NOT156/sub_NOT156_test')
        if not os.path.isdir(self.root_dir):
            self.root_dir = os.path.join(root_dir, 'NOT156')
        for video_name in natsorted(os.listdir(self.root_dir)):
            vd = os.path.join(self.root_dir, video_name)
            if not os.path.isdir(vd): 
                continue
            self.vis_paths = natsorted(glob(os.path.join(vd, 'channel', '*.jpg')))
            self.ir_paths  = natsorted(glob(os.path.join(vd, 'channel2',  '*.jpg')))
            if len(self.vis_paths) != len(self.ir_paths):
                raise RuntimeError(f"[Error] {video_name} 下 vis/ 和 ir/ 帧数不一致")
            N = len(self.vis_paths)
            starts = list(range(0, N - seq_len + 1, self.step))
            last_start = N - seq_len
            if (not starts) or (starts[-1] != last_start):
                starts.append(last_start)
            for st in starts:
                self.index_list.append((video_name, self.vis_paths, self.ir_paths, st))

    def __len__(self):
        return len(self.index_list)
    
    def imread(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = cv2.resize(img, [self.train_size_h, self.train_size_w])
        return img

    def __getitem__(self, idx):
        video_name, vis_paths, ir_paths, start = self.index_list[idx]
        vis_seq = []
        ir_seq  = []

        frame_names = []
        for i in range(start, start + self.seq_len):
            vis_p = vis_paths[i]
            ir_p  = ir_paths[i]
            vis_frame_names = os.path.basename(vis_p)
            frame_names.append(vis_frame_names.replace('ll', ''))
            
            vis = self.imread(vis_p)
            ir  = self.imread(ir_p)
            vis = Image.fromarray(vis)
            ir = Image.fromarray(ir)

            if self.transform:
                vis = self.transform(vis)
                ir  = self.transform(ir)    
            vis_seq.append(vis)
            ir_seq.append(ir)
        vis_seq = torch.stack(vis_seq, dim=0)
        ir_seq  = torch.stack(ir_seq,  dim=0)
        return video_name, frame_names, vis_seq, ir_seq

class HDO(Dataset):
    def __init__(self, root_dir, seq_len=1, overlap=0):
        super(HDO, self).__init__()
        self.seq_len = seq_len
        self.step = seq_len - overlap
        self.transform = transforms.Compose([transforms.ToTensor()]) 
        self.index_list = [] 
        self.root_dir = os.path.join(root_dir, 'sub_HDO')
        for video_name in natsorted(os.listdir(self.root_dir)):
            vd = os.path.join(self.root_dir, video_name)
            if not os.path.isdir(vd): 
                continue
            self.vis_paths = natsorted(glob(os.path.join(vd, 'visible', '*.jpg')))
            self.ir_paths  = natsorted(glob(os.path.join(vd, 'infrared',  '*.jpg')))
            if len(self.vis_paths) != len(self.ir_paths):
                raise RuntimeError(f"[Error] {video_name} 下 vis/ 和 ir/ 帧数不一致")
            N = len(self.vis_paths)
            starts = list(range(0, N - seq_len + 1, self.step))
            last_start = N - seq_len
            if (not starts) or (starts[-1] != last_start):
                starts.append(last_start)
            for st in starts:
                self.index_list.append((video_name, self.vis_paths, self.ir_paths, st))

    def __len__(self):
        return len(self.index_list)
    
    def imread(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = cv2.resize(img, [self.train_size_h, self.train_size_w])
        return img

    def __getitem__(self, idx):
        video_name, vis_paths, ir_paths, start = self.index_list[idx]
        vis_seq = []
        ir_seq  = []

        frame_names = []
        for i in range(start, start + self.seq_len):
            vis_p = vis_paths[i]
            ir_p  = ir_paths[i]
            vis_frame_names = os.path.basename(vis_p)
            # frame_names.append(vis_frame_names.replace('v', ''))
            frame_names.append(vis_frame_names)
            
            vis = self.imread(vis_p)
            ir  = self.imread(ir_p)
            vis = Image.fromarray(vis)
            ir = Image.fromarray(ir)

            if self.transform:
                vis = self.transform(vis)
                ir  = self.transform(ir)    
            vis_seq.append(vis)
            ir_seq.append(ir)
        vis_seq = torch.stack(vis_seq, dim=0)
        ir_seq  = torch.stack(ir_seq,  dim=0)
        return video_name, frame_names, vis_seq, ir_seq

class M3SVD(Dataset):
    def __init__(self, root_dir, seq_len=1, overlap=0):
        super(M3SVD, self).__init__()
        self.seq_len = seq_len
        self.step = seq_len - overlap
        self.transform = transforms.Compose([transforms.ToTensor()]) 
        self.index_list = [] 
        self.root_dir = os.path.join(root_dir, 'M3SVD')
        for video_name in natsorted(os.listdir(self.root_dir)):
            vd = os.path.join(self.root_dir, video_name)
            if not os.path.isdir(vd): 
                continue
            self.vis_paths = natsorted(glob(os.path.join(vd, 'visible', '*.png')))
            self.ir_paths  = natsorted(glob(os.path.join(vd, 'infrared',  '*.png')))
            if len(self.vis_paths) != len(self.ir_paths):
                raise RuntimeError(f"[Error] {video_name} 下 vis/ 和 ir/ 帧数不一致")
            N = len(self.vis_paths)
            starts = list(range(0, N - seq_len + 1, self.step))
            last_start = N - seq_len
            if (not starts) or (starts[-1] != last_start):
                starts.append(last_start)
            for st in starts:
                self.index_list.append((video_name, self.vis_paths, self.ir_paths, st))

    def __len__(self):
        return len(self.index_list)
    
    def imread(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = cv2.resize(img, [self.train_size_h, self.train_size_w])
        return img

    def __getitem__(self, idx):
        video_name, vis_paths, ir_paths, start = self.index_list[idx]
        vis_seq = []
        ir_seq  = []

        frame_names = []
        for i in range(start, start + self.seq_len):
            vis_p = vis_paths[i]
            ir_p  = ir_paths[i]
            vis_frame_names = os.path.basename(vis_p)
            # frame_names.append(vis_frame_names.replace('v', ''))
            frame_names.append(vis_frame_names)
            
            vis = self.imread(vis_p)
            ir  = self.imread(ir_p)
            vis = Image.fromarray(vis)
            ir = Image.fromarray(ir)

            if self.transform:
                vis = self.transform(vis)
                ir  = self.transform(ir)    
            vis_seq.append(vis)
            ir_seq.append(ir)
        vis_seq = torch.stack(vis_seq, dim=0)
        ir_seq  = torch.stack(ir_seq,  dim=0)
        return video_name, frame_names, vis_seq, ir_seq


class VTMOT(Dataset):
    def __init__(self, root_dir, seq_len=1, overlap=0):
        super(VTMOT, self).__init__()
        self.seq_len = seq_len
        self.step = seq_len - overlap
        self.transform = transforms.Compose([transforms.ToTensor()]) 
        self.index_list = [] 
        self.root_dir = os.path.join(root_dir, 'VTMOT_split')
        if not os.path.isdir(self.root_dir):
            self.root_dir = os.path.join(root_dir, 'VTMOT')
        for video_name in natsorted(os.listdir(self.root_dir)):
            vd = os.path.join(self.root_dir, video_name)
            if not os.path.isdir(vd): 
                continue
            self.vis_paths = natsorted(glob(os.path.join(vd, 'visible', '*.jpg')))
            self.ir_paths  = natsorted(glob(os.path.join(vd, 'infrared',  '*.jpg')))
            if len(self.vis_paths) != len(self.ir_paths):
                raise RuntimeError(f"[Error] {video_name} 下 vis/ 和 ir/ 帧数不一致")
            N = len(self.vis_paths)
            starts = list(range(0, N - seq_len + 1, self.step))
            last_start = N - seq_len
            if (not starts) or (starts[-1] != last_start):
                starts.append(last_start)
            for st in starts:
                self.index_list.append((video_name, self.vis_paths, self.ir_paths, st))

    def __len__(self):
        return len(self.index_list)
    
    def imread(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = cv2.resize(img, [self.train_size_h, self.train_size_w])
        return img

    def __getitem__(self, idx):
        video_name, vis_paths, ir_paths, start = self.index_list[idx]
        vis_seq = []
        ir_seq  = []

        frame_names = []
        for i in range(start, start + self.seq_len):
            vis_p = vis_paths[i]
            ir_p  = ir_paths[i]
            vis_frame_names = os.path.basename(vis_p)
            # frame_names.append(vis_frame_names.replace('v', ''))
            frame_names.append(vis_frame_names)
            
            vis = self.imread(vis_p)
            ir  = self.imread(ir_p)
            vis = Image.fromarray(vis)
            ir = Image.fromarray(ir)

            if self.transform:
                vis = self.transform(vis)
                ir  = self.transform(ir)    
            vis_seq.append(vis)
            ir_seq.append(ir)
        vis_seq = torch.stack(vis_seq, dim=0)
        ir_seq  = torch.stack(ir_seq,  dim=0)
        return video_name, frame_names, vis_seq, ir_seq
