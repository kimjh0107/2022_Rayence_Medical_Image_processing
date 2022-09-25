from src.dataloader import CustomDataloader
from src.unet.UNet import UNet
from src.unet.UNet2Plus import UNet2Plus
from src.unet.UNet3Plus import UNet3Plus
from src.BCDU.BCDUNet import BCDUNet

from src.train import train_net



from pathlib import Path

device = 'cuda:0'
save_root = Path('Result')

def train_model() : 
    for image_size, batch_size in [
                                    [ 256,  48],
                                    [ 512,  12],
                                    [1024,   3],
                                    ] : 
        dataloader = CustomDataloader(image_size, batch_size)
        dataloaders = dataloader.set_dataloaders()

        # # UNet Training
        # for model_class in [
        #                         # UNet, 
        #                         # UNet2Plus, 
        #                         # UNet3Plus,
        #                     ] : 
        #     model_name = model_class.__name__
        #     result_path = save_root.joinpath(model_name).joinpath(f"IMG_{image_size}_BS_{batch_size}")
        #     result_path.mkdir(parents = True, exist_ok=True)

        #     model = model_class(
        #                 n_channels = 1, # Input Channel의 수
        #                 n_classes = 3,  # Output Channel의 수
        #                 ).to(device)

        #     model = train_net(  net = model,
        #                 dataloaders = dataloaders,
        #                 device = device,
        #                 result_path = result_path,
        #                 learning_rate = 0.001,
        #             )

        # BCDU Training
        
        model = BCDUNet(n_channels = 1, 
                        n_classes = 3, 
                        frame_size = (image_size, image_size),
                        bidirectional=True).to(device)
        model_name = f'BCDUNet_BI'
        result_path = save_root.joinpath(model_name).joinpath(f"IMG_{image_size}_BS_{batch_size}")
        result_path.mkdir(parents = True, exist_ok=True)
        model = train_net(  
                    net = model,
                    dataloaders = dataloaders,
                    device = device,
                    result_path = result_path,
                    learning_rate = 0.0001,
                    epochs =  999
                    )
    
    

                    
if __name__ == "__main__":
    train_model()