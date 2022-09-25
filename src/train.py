import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .evaluate import evaluate
from .logger import print_logger



def train_net(net,
              dataloaders,
              device,
              result_path : Path,
              learning_rate: float = 0.1,
              epochs : int = 999,
              ):

    train_loader = dataloaders['Train']
    val_loader = dataloaders['Valid']

    early_stop = 0
    early_stop_criterion = 12
    best_val_score = 0 
    total_start_time = time.time()
    
    
    logger = print_logger(result_path.joinpath('LOG').with_suffix('.txt'))

    image_path = result_path.joinpath('Prediction')
    image_path.mkdir(exist_ok=True, parents = True)
    checkpoint = result_path.joinpath('Model_Weight').with_suffix('.pth')
    checkpoint.parent.mkdir(exist_ok = True, parents = True)

    optimizer = RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    # optimizer = Adam(net.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.1, patience = 4, min_lr = 1e-5)  # goal: maximize Dice score
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs+1):
        start_time = time.time()
        net.train()
        epoch_loss = 0
        for images, true_masks, _ in train_loader :
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            
            masks_pred = net(images)
            loss = criterion(masks_pred, true_masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(train_loader)

        dice_score, sensitivity, specificity = evaluate(net, val_loader, device, image_path)
        scheduler.step(dice_score)

        if dice_score <= best_val_score : 
            early_stop += 1
        else : 
            early_stop = 0
            best_val_score = dice_score
            torch.save(net.state_dict(), checkpoint)
        
        if early_stop == early_stop_criterion : 
            break    
        
        time_elapsed  = time.time() - start_time
        total_elapsed = time.time() - total_start_time
        total_min = total_elapsed // 60
        total_sec = total_elapsed %  60
        lr = optimizer.param_groups[0]['lr']
        logger(f'[EPOCH : {epoch:3d}/{epochs:3d}] \
| LOSS : [{epoch_loss:.4f}] \
| DICE : [{best_val_score:.4f}] \
| SENSI : [{sensitivity:.4f}] \
| SPECI : [{specificity:.4f}] \
| ES : [{early_stop}/{early_stop_criterion}] \
| LR : [{lr:.5f}] \
| TIME : [{int(time_elapsed):3d}S / {int(total_min):2d}M {int(total_sec):2d}S]'
)

    net.load_state_dict(torch.load(checkpoint))
    final_val_score = evaluate(net, val_loader, device, image_path)
    logger(f'\n\nFINAL VALIDATION SCORE : {final_val_score}')

    return net
    
