from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_path: Path, phase, image_size, transforms=None) -> None :
        """
        data_path : 데이터 경로 (phase 포함) Example : data/new_data/train
        """
        super(CustomDataset, self).__init__()
        self.phase = phase
        self.image_size = image_size
        self.img_paths = sorted([path for path in data_path.joinpath('CXR').glob('*.jpg')])
        self.mask_paths = sorted([path for path in data_path.joinpath('Mask').glob('*.bmp')])
        self.transforms = transforms

    def __getitem__(self, idx):
        if self.phase == 'Test' : 
            img = Image.open(self.img_paths[idx]).resize(self.image_size)
            return F.to_tensor(img), self.img_paths[idx].stem
        else : 
            img = Image.open(self.img_paths[idx]).resize(self.image_size)
            mask = Image.open(self.mask_paths[idx]).resize(self.image_size)
            if self.transforms :
                img, mask = self.transforms(img, mask)

            return img, mask, self.img_paths[idx].stem
        
    def __len__(self):
        return len(self.img_paths)