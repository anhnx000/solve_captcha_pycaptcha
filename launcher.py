from model.model import captcha_model, model_conv, model_resnet
from data.datamodule import captcha_dm
import pytorch_lightning as pl
import torch.optim as optim
import torch
import os
from utils.config_util import configGetter
from utils.arg_parsers import train_arg_parser

# Parse arguments first so they can be used later
args = train_arg_parser()

cfg  = configGetter('SOLVER')
lr = cfg['LR']
batch_size = cfg['BATCH_SIZE']
epoch = cfg['EPOCH']

def main(args):
    pl.seed_everything(42)
    m = model_resnet()
    model = captcha_model(
        model=m, lr=lr)
    dm = captcha_dm(batch_size=batch_size)

    wandb_logger = pl.loggers.WandbLogger(
        project="captcha",
        offline=False,
        name=args.exp_name
    )
        
    trainer = pl.Trainer(deterministic=True,
                         precision='bf16-mixed',  
                         fast_dev_run=False,
                         max_epochs=epoch,
                         log_every_n_steps=50
                        )
    
    trainer.fit(model, datamodule=dm)
    os.makedirs(args.save_path, exist_ok=True)
    trainer.save_checkpoint(os.path.join(args.save_path, 'model.pth'))
    
if __name__ == "__main__":
    main(args)

