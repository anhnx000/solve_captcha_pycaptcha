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
from pytorch_lightning.callbacks import ModelCheckpoint, Callback


torch.set_float32_matmul_precision('medium')
# Parse arguments first so they can be used later
args = train_arg_parser()

# Load configurations
cfg_solver = configGetter('SOLVER')
lr = cfg_solver['LR']
batch_size = cfg_solver['BATCH_SIZE']
epoch = cfg_solver['EPOCH']

cfg_trainer = configGetter('TRAINER')
precision_setting = cfg_trainer['PRECISION']

# Custom callback to display training accuracy
class TrainAccuracyCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:  # Print every 10 batches
            metrics = trainer.callback_metrics
            if 'train acc' in metrics:
                print(f"Batch {batch_idx}: Train Accuracy: {metrics['train acc']:.4f}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'train acc' in metrics:
            print(f"\nEpoch {trainer.current_epoch} ended: Train Accuracy: {metrics['train acc']:.4f}")
            if 'val acc' in metrics:
                print(f"Epoch {trainer.current_epoch} ended: Val Accuracy: {metrics['val acc']:.4f}")


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
        model=m, lr=lr, use_ctc=args.use_ctc)
    dm = captcha_dm(batch_size=batch_size, num_workers=20)
    add_time_str = datetime.now().strftime("%Y%m%d_%H%M")
    # setting để wandb chỉ lưu online mode
    os.environ['WANDB_MODE'] = 'online'
    
    # Add CTC to experiment name if using CTC loss
    exp_name = f'exp_ctc_{add_time_str}' if args.use_ctc else f'exp_{add_time_str}'
    wandb.init(project="captcha", group="ctc_loss", name=exp_name)
    
    # Setup model checkpoint callbacks
    checkpoint_callbacks = [
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
        ),
        TrainAccuracyCallback()
    ]
    
    trainer_kwargs = {
        'deterministic': 'warn',
        'precision': precision_setting, # Read from config
        'fast_dev_run': False,
        'max_epochs': epoch,
        'log_every_n_steps': 100,
        'callbacks': checkpoint_callbacks
    }
    
    # If checkpoint path is provided, use it to resume training
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer_kwargs['ckpt_path'] = args.resume_from_checkpoint
        
    trainer = pl.Trainer(**trainer_kwargs)
    
    trainer.fit(model, datamodule=dm)
    os.makedirs(args.save_path, exist_ok=True)
    trainer.save_checkpoint(os.path.join(args.save_path, 'model.pth'))
    
if __name__ == "__main__":
    main(args)

