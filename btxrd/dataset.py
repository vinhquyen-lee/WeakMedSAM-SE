import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random
import os
import pickle

# 9 loại u gốc của BTRXD
TUMOR_CLASSES = [
    'osteochondroma', 'multiple osteochondromas', 'simple bone cyst', 
    'giant cell tumor', 'osteofibroma', 'synovial osteochondroma', 
    'other bt', 'osteosarcoma', 'other mt'
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(TUMOR_CLASSES)}
PIC_SIZE = 256

def aug(img: torch.Tensor, seg: torch.Tensor, lab: torch.Tensor = None):
    rotate_angle = random.randrange(-20, 20)
    img = TF.rotate(img, rotate_angle)
    seg = TF.rotate(seg, rotate_angle, interpolation=TF.InterpolationMode.NEAREST)
    if lab is not None:
        lab = TF.rotate(lab, rotate_angle, interpolation=TF.InterpolationMode.NEAREST)
        
    if random.random() > 0.5:
        img = TF.hflip(img)
        seg = TF.hflip(seg)
        if lab is not None:
            lab = TF.hflip(lab)
            
    jitter = TF.adjust_brightness(img, brightness_factor=random.uniform(0.5, 1.5))
    img = TF.adjust_contrast(jitter, contrast_factor=random.uniform(0.5, 1.5))
    
    if lab is not None:
        return img, seg, lab
    return img, seg


class BTRXDDataset(Dataset):
    def __init__(self, data_dir: str, file_names: list, train: bool, child_classes: int, cluster_file: str) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.file_names = file_names
        self.train = train
        self.k_per_class = child_classes 
        
        if self.k_per_class != 0 and cluster_file and os.path.exists(cluster_file):
            with open(cluster_file, "rb") as f:
                self.clabs = pickle.load(f)
        else:
            self.clabs = None

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int) -> dict:
        sample_name = self.file_names[index]
        
        img_path = os.path.join(self.data_dir, f"{sample_name}.jpg")
        img = TF.to_tensor(Image.open(img_path).convert("RGB"))
        
        mask_path = os.path.join(self.data_dir, f"{sample_name}_mask.npy")
        seg = torch.from_numpy(np.load(mask_path)).float()
        if len(seg.shape) == 2:
            seg = seg.unsqueeze(0)
        
        if self.train:
            img, seg = aug(img, seg)
            


        if self.k_per_class == 0:
            # 9 nhãn cho cluster.py
            plab = torch.zeros(9).float()
            for i in range(9):
                if torch.any(seg == (i + 1)):
                    plab[i] = 1.0
        else:
            # 1 nhãn (Nhị phân) cho train.py
            plab = torch.zeros(1).float()
            if torch.any(seg > 0):
                plab[0] = 1.0

        
        out_dict = {
            "img": img,
            "plab": plab,
            "seg": seg,
            "idx": sample_name,
            "fname": img_path
        }


        if self.k_per_class != 0:
            total_child_classes = 36  # Đóng đinh 36 cụm
            clab = torch.zeros(total_child_classes + 1).float()
            
            if plab[0] == 0.0:
                clab[0] = 1.0  # Ảnh khỏe mạnh
            else:
                for i in range(9):
                    if torch.any(seg == (i + 1)):
                        # Lấy cluster_id từ file .bin
                        if self.clabs and sample_name in self.clabs:
                            cluster_id = int(self.clabs[sample_name][i])
                        else:
                            cluster_id = 0
                            
                        # Đóng đinh mỗi bệnh có 4 hình thái
                        clab_idx = 1 + (i * 4) + cluster_id 
                        clab[clab_idx] = 1.0
            out_dict["clab"] = clab

        return out_dict


def get_dataset(preprocessed_dir: str, child_classes: int, cluster_file: str):
    all_files = [f.split('.')[0] for f in os.listdir(preprocessed_dir) if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(all_files)
    
    total = len(all_files)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_dataset = BTRXDDataset(preprocessed_dir, all_files[:train_end], True, child_classes, cluster_file)
    val_dataset = BTRXDDataset(preprocessed_dir, all_files[train_end:val_end], False, child_classes, cluster_file)
    test_dataset = BTRXDDataset(preprocessed_dir, all_files[val_end:], False, 0, cluster_file)
    
    return train_dataset, val_dataset, test_dataset


def get_all_dataset(preprocessed_dir: str, child_classes: int = 0, cluster_file: str = "") -> Dataset:
    all_files = [f.split('.')[0] for f in os.listdir(preprocessed_dir) if f.endswith('.jpg')]
    dataset = BTRXDDataset(
        data_dir=preprocessed_dir, 
        file_names=all_files, 
        train=False,  
        child_classes=child_classes, 
        cluster_file=cluster_file
    )
    return dataset


class BTRXDSegDataset(Dataset):
    def __init__(self, data_dir: str, lab_dir: str, file_names: list, train: bool) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.lab_dir = lab_dir
        self.file_names = file_names
        self.train = train

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int) -> dict:
        sample_name = self.file_names[index]
        
        img_path = os.path.join(self.data_dir, f"{sample_name}.jpg")
        img = TF.to_tensor(Image.open(img_path).convert("RGB"))
        
        mask_path = os.path.join(self.data_dir, f"{sample_name}_mask.npy")
        if os.path.exists(mask_path):
            seg = torch.from_numpy(np.load(mask_path)).float()
        else:
            seg = torch.zeros((img.shape[1], img.shape[2])).float()
        if len(seg.shape) == 2:
            seg = seg.unsqueeze(0)
            
        
        lab_path = os.path.join(self.lab_dir, f"{sample_name}.png")
        if os.path.exists(lab_path):
            lab_img = Image.open(lab_path).convert('L')
            lab_np = np.array(lab_img)
            
            true_class = int(seg.max().item())
            
            if true_class > 0:
                lab_np = np.where(lab_np > 0, true_class, 0)
            else:
                lab_np = np.zeros_like(lab_np)
                
            lab = torch.from_numpy(lab_np).float()
        else:
            lab = torch.zeros((img.shape[1], img.shape[2])).float()
            
        if len(lab.shape) == 2:
            lab = lab.unsqueeze(0)
            
        if self.train:
            img, seg, lab = aug(img, seg, lab)
            
        out_dict = {
            "img": img.float(),
            "seg": seg.squeeze(0).long(), 
            "lab": lab.squeeze(0).long(), 
            "idx": sample_name
        }
        return out_dict

    
def get_seg_dataset(data_dir: str, lab_dir: str):
    all_files = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(all_files)
    
    total = len(all_files)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_dataset = BTRXDSegDataset(data_dir, lab_dir, all_files[:train_end], True)
    val_dataset = BTRXDSegDataset(data_dir, lab_dir, all_files[train_end:val_end], False)
    test_dataset = BTRXDSegDataset(data_dir, lab_dir, all_files[val_end:], False)
    
    return train_dataset, val_dataset, test_dataset