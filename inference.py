from pathlib import Path
import yaml
import numpy as np
import torch
from torchvision.utils import save_image

from src.dataloader import CustomDataloader
from src.unet.UNet import UNet
from src.unet.UNet2Plus import UNet2Plus
from src.unet.UNet3Plus import UNet3Plus
from src.logger import print_logger


with open('configure.yaml', 'r') as f : 
    files = yaml.safe_load(f)

for key, value in files.items() : 
    if key.endswith('_path') : 
        globals()[key] = Path(value)
    else : 
        globals()[key] = value


# inference_model      ( EX ) "UNnet3Plus"
# inference_img_size   ( EX ) 1024
device = 'cuda:0'

def inference() : 
    print('\n\nChecking Model Information...')
    if inference_model == 'UNet' : 
        model = UNet
    elif inference_model == 'UNet2Plus' : 
        model = UNet2Plus
    elif inference_model == 'UNet3Plus' : 
        model = UNet3Plus
    else : 
        print('Please Check your Model Name.')
        return
        
    log_root = [path for path in Path('Result').joinpath(inference_model).glob('*') if str(inference_img_size) in path.name] [0]

    batch_size = int(log_root.name.split('_')[-1])
    model_weight = log_root.joinpath('Model_Weight.pth')
    model_log = log_root.joinpath('LOG.txt')
    with open(model_log, 'r') as f : 
        logs = f.readlines()[:-3]
    best_val_score = sorted([log.split('|')[2][-8:-2] for log in logs])[-1]    
    print(f'MODEL : {inference_model} | BATCH SIZE : {batch_size} | VALID SCORE : {best_val_score}\n')

    
    print('Loading Model Weight...\n')
    model = model(n_channels = 1, n_classes = 3).to(device)
    model.load_state_dict(torch.load(model_weight))

    print('Loading Data Loader...\n')
    test_loader = CustomDataloader(image_size = inference_img_size, batch_size = batch_size).single_dataloader('Test')


    print('Model Infernece Initiated ...\n')
    image_path = Path('Inference').joinpath(inference_model).joinpath(f"IMG_{inference_img_size}")
    image_path.mkdir(parents = True, exist_ok = True)
    logger_path = image_path.joinpath('INFERENCE_LOG.txt')
    logger = print_logger(logger_path)
    with torch.no_grad() : 
        model.eval()
        for image, path_name in test_loader  :
            
            image = image.to(device, dtype = torch.float32)

            mask_pred = model(image)
            mask_pred = mask_pred.cpu()
            mask_pred = torch.zeros(mask_pred.shape).scatter(1, mask_pred.argmax(1).unsqueeze(1), 1)
            
            logger(f'\tIMG NUM : {path_name[0]}')
            
            for img, pred, path in zip(image, mask_pred, path_name) : 
                img = torch.cat([img, img, img]).detach().cpu()
                true_and_pred = torch.cat([img, pred], dim  = -1)
                
                save_image(true_and_pred, image_path.joinpath(f"{path}.png"))
                np.save(image_path.joinpath(f'{path}.npy'), pred.numpy())
if __name__ == '__main__':
    inference()