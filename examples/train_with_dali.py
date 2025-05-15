import os
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import captcha_dm, DALI_AVAILABLE
from model.model import captcha_model, model_resnet
from utils.config_util import configGetter

def main():
    # Load configuration
    solver_cfg = configGetter('SOLVER')
    logger_cfg = configGetter('LOGGER')
    trainer_cfg = configGetter('TRAINER')
    
    # Setup WandB logger
    wandb_logger = WandbLogger(project="captcha-solver-dali", name="resnet-dali")
    
    # Create model
    model = model_resnet()
    captcha = captcha_model(model, lr=solver_cfg['LR'])
    
    # Create data module with DALI enabled
    dm = captcha_dm(
        batch_size=solver_cfg.get('BATCH_SIZE', 64),
        num_workers=8,
        use_dali=True  # Enable DALI GPU data loading
    )
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger_cfg.get('CHECKPOINT_DIR', './checkpoint'),
        filename='captcha-solver-dali-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    # Create trainer with mixed precision
    trainer = pl.Trainer(
        max_epochs=solver_cfg.get('EPOCH', 20),
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision=trainer_cfg.get('PRECISION', '16-mixed')
    )
    
    # Train model
    trainer.fit(captcha, dm)

if __name__ == "__main__":
    if not DALI_AVAILABLE:
        print("NVIDIA DALI is not available. Please install it first:")
        print("pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110")
    else:
        main() 
