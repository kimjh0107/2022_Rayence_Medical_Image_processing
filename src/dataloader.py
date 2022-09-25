from pathlib import Path
from torch.utils.data import DataLoader
import yaml
from torch.utils.data import ConcatDataset

from .dataset import CustomDataset
from .transforms import Compose, ToTensor

with open('configure.yaml', 'r') as f : 
    files = yaml.safe_load(f)

for key, value in files.items() : 
    if key.endswith('_path') : 
        globals()[key] = Path(value)
    else : 
        globals()[key] = value

class CustomDataloader : 
    def __init__(self, image_size, batch_size) :
        self.data_path = data_root_path.joinpath(new_data_name)
        self.image_size = (image_size, image_size)
        self.batch_size = batch_size
        self.transforms = Compose([
                                    ToTensor(),
                                ])
        self.train_dset = CustomDataset(self.data_path.joinpath('Train'), 'Train', self.image_size, transforms = self.transforms)
        self.valid_dset = CustomDataset(self.data_path.joinpath('Valid'), 'Valid', self.image_size, transforms = self.transforms)
        self.test_dset  = CustomDataset(self.data_path.joinpath('Test') , 'Test', self.image_size , transforms = self.transforms)

    def single_dataloader(self, phase : str) : 
        if phase == 'Train' : 
            dset, shuffle = self.train_dset, True
        elif phase == 'Valid' : 
            dset, shuffle = self.valid_dset, False
        elif phase == 'Test' : 
            dset, shuffle = self.test_dset,  False
        elif phase == 'Total' : 
            dset, shuffle = ConcatDataset([self.train_dset, self.valid_dset]), True
        return DataLoader(dset, self.batch_size,shuffle, pin_memory=True, num_workers=2)

    def set_dataloaders(self) : 
        train_loader = self.single_dataloader('Train')
        valid_loader = self.single_dataloader('Valid')
        return {
            'Train' : train_loader,
            'Valid' : valid_loader,
        }