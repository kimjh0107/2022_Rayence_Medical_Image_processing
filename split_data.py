from pathlib import Path
import shutil
from PIL import Image
from joblib import Parallel, delayed
import yaml
import numpy as np

with open('configure.yaml', 'r') as f : 
    files = yaml.safe_load(f)

for key, value in files.items() : 
    if key.endswith('_path') : 
        globals()[key] = Path(value)
    else : 
        globals()[key] = value

np.random.seed(random_seed)

class preprocess_data : 

    def __init__(self) : 
        self.data_path = data_root_path.joinpath(raw_data_name)
        self.save_path = data_root_path.joinpath(new_data_name)
        quest = self.check_directory()
        if quest is not None : 
            return
        self.process()
        

    def check_directory(self) : 
        """
        저장하려는 Data 경로가 있는지 파악하고, 없으면 경로 생성, 있으면 삭제할지 질문
        """
        if new_data_name in [path.stem for path in data_root_path.glob('*')] : 
            quest = input('New Data Directory Already Exists.\n Do you want to overwrite?[Y/n]\t : \t')
            if quest == 'n' : 
                return quest
            elif quest.lower() == 'y' : 
                shutil.rmtree(data_root_path.joinpath(new_data_name))
        self.save_path.mkdir(exist_ok = True)

    def process(self) : 
        """
        Train-Valid로 나누어 랜덤하게 샘플링하고, 모든 데이터를 Mask를 Lung, Heart, Background로 분리        
        """        
        train_names = [path.stem for path in data_root_path.joinpath(raw_data_name).joinpath('Train').joinpath('CXR').glob('*.jpg')]
        np.random.shuffle(train_names)

        train_idx = int(train_ratio * len(train_names))
        tr_names   = train_names[           : train_idx ]
        val_names  = train_names[ train_idx :           ]
        test_names = [path.stem for path in data_root_path.joinpath(raw_data_name).joinpath('Test').joinpath('CXR').glob('*.jpg')]

        for names, phase in [[tr_names, 'Train'], [val_names, 'Valid'], [test_names, 'Test']] : 
            self.process_directory(names, phase)


    def process_directory(self, names, phase) : 
        phase_save_path = self.save_path.joinpath(phase)
        phase_save_path.mkdir(exist_ok = True)
        phase_save_path.joinpath('CXR').mkdir(exist_ok = True)
        phase_save_path.joinpath('Mask').mkdir(exist_ok = True)
        
        raw_phase = 'Train' if phase != 'Test' else 'Test'
        Parallel(n_jobs = 6)(delayed(self.process_name)(name, phase_save_path, raw_phase) for name in names)

    
    def process_name(self, name, phase_save_path, raw_phase) : 
        raw_cxr_path = self.data_path.joinpath(raw_phase).joinpath('CXR').joinpath(name).with_suffix('.jpg')
        raw_bmp_path = self.data_path.joinpath(raw_phase).joinpath('Mask').joinpath(name).with_suffix('.bmp')

        new_cxr_path = phase_save_path.joinpath('CXR').joinpath(name).with_suffix('.jpg')
        new_bmp_path = phase_save_path.joinpath('Mask').joinpath(name).with_suffix('.bmp')

        shutil.copyfile(raw_cxr_path, new_cxr_path)
        try : 
            mask = Image.open(raw_bmp_path)
            mask = np.array(mask)
            mask = mask[:, :, 0:1]
            lung = (mask==255)
            heart = (mask==128)
            background = (mask==0)
            mask = np.concatenate([lung, heart, background], axis = 2, dtype = np.uint8)
            mask = Image.fromarray(mask).save(new_bmp_path)
        except : 
            pass

if __name__ == '__main__':
    preprocess_data()