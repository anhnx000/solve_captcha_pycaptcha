from model.model import captcha_model, model_conv, model_resnet, model_efficientnet, model_vit, model_mobilenet
from data.datamodule import captcha_dm
import pytorch_lightning as pl
import torch.optim as optim
import torch
import os
from utils.config_util import configGetter
from utils.arg_parsers import train_arg_parser
import wandb 
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint


torch.set_float32_matmul_precision('high')
# Parse arguments first so they can be used later
args = train_arg_parser()

cfg  = configGetter('SOLVER')
lr = cfg['LR']
batch_size = cfg['BATCH_SIZE']
epoch = cfg['EPOCH']



# import ModelCheckpoint

def main(args):
    pl.seed_everything(42)
    
    # Force wandb to be online mode
    os.environ['WANDB_MODE'] = 'online'
    
    if args.model_name == 'resnet':
        m  = model_resnet()
    elif args.model_name == 'efficientnet':
        m  = model_efficientnet()
    elif args.model_name == 'vit':
        m  = model_vit()
    elif args.model_name == 'mobilenet':
        m  = model_mobilenet()
        
    
    model = captcha_model(
        model=m, lr=lr)
    dm = captcha_dm(batch_size=batch_size, num_workers=20)
    add_time_str = datetime.now().strftime("%Y%m%d_%H%M")
    # setting để wandb chỉ lưu online mode
    os.environ['WANDB_MODE'] = 'online'
    wandb.init(project="captcha", group="captcha_tuning", name=f'exp_{add_time_str}')
        
    trainer = pl.Trainer(deterministic=True,
                         precision='bf16-mixed', 
                         fast_dev_run=False,
                         max_epochs=epoch,
                         log_every_n_steps=100,
                         callbacks=[
                             ModelCheckpoint(
                                 monitor='val_acc', 
                                 mode='max',
                                 save_top_k=1,
                                 filename='../{val_acc:.2f}_best_val_acc-{epoch:02d}'
                             ),
                             ModelCheckpoint(
                                 monitor='train_acc',
                                 mode='max',
                                 save_top_k=1,
                                 filename='../{train_acc:.2f}_best_train_acc-{epoch:02d}'
                             )
                         ]
                        )
    
    trainer.fit(model, datamodule=dm)
    os.makedirs(args.save_path, exist_ok=True)
    trainer.save_checkpoint(os.path.join(args.save_path, 'model.pth'))
    
if __name__ == "__main__":
    main(args)

