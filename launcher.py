from model.model import captcha_model, model_resnet, model_efficientnet, model_vit, model_mobilenet, model_trocr
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


torch.set_float32_matmul_precision('high')
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
    # pl.seed_everything(42)
    
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
    elif args.model_name == 'trocr':
        m  = model_trocr()
    
    model = captcha_model(
        model=m, lr=lr, use_ctc=args.use_ctc)
    dm = captcha_dm(batch_size=batch_size, num_workers=20)
    add_time_str = datetime.now().strftime("%Y%m%d_%H%M")
    # setting để wandb chỉ lưu online mode
    os.environ['WANDB_MODE'] = 'online'
    
    # Add CTC to experiment name if using CTC loss
    exp_name = f'exp_ctc_{add_time_str}' if args.use_ctc else f'exp_{add_time_str}'
    
    exp_name = exp_name + '_' + args.model_name
    # if args.use_ctc = true thì group là ctc_loss, nếu false thì group là no_ctc_loss  
    if args.use_ctc:
        wandb.init(project="captcha", group="ctc_loss", name=exp_name)
    else:
        wandb.init(project="captcha", group="no_ctc_loss", name=exp_name)
    
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
        'callbacks': checkpoint_callbacks,
    }
    
    # Create trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # If checkpoint path is provided, use it in fit() method to resume training
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model, datamodule=dm)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # import ipdb; ipdb.set_trace() 
    
    
    val_acc = trainer.callback_metrics['val_acc']
    model_name = args.model_name
    
    # Save PyTorch Lightning checkpoint (full model for resuming training)
    trainer.save_checkpoint(os.path.join(args.save_path, f'{val_acc:.2f}_model_{model_name}.pth'))
    
    # Save model state dict (weights only)
    torch.save(model.model.state_dict(), os.path.join(args.save_path, f'{val_acc:.2f}_model_{model_name}_weights.pt'))
    
    # Save model in TorchScript format
    print("Saving model in TorchScript format...")
    model.eval()  # Set model to evaluation mode
    # scripted_model = torch.jit.script(model.model)  # Script the base model, not the Lightning module
    # scripted_model.save(os.path.join(args.save_path, f'{val_acc:.2f}_model_{model_name}_scripted.pt'))
    
    
    # Save model in ONNX format
    print("Saving model in ONNX format...")
    # Create dummy input based on the dataset dimensions
    dummy_input = torch.randn(1, 3, 50, 200)  # Batch size 1, 3 channels, HEIGHT=224, WIDTH=224
    
    # Export to ONNX format
    torch.onnx.export(model.model,                    # model being exported
                      dummy_input,                     # model input example
                      os.path.join(args.save_path, f'{val_acc:.2f}_model_{model_name}.onnx'),  # output file
                      export_params=True,              # store the trained weights
                      opset_version=12,                # ONNX version
                      do_constant_folding=True,        # optimization
                      input_names=['input'],           # input layer names
                      output_names=['output'],         # output layer names
                      dynamic_axes={'input': {0: 'batch_size'},  # dynamic batch size
                                   'output': {0: 'batch_size'}})
    
    print(f"Models saved in {args.save_path} in multiple formats:")
    print(f"1. PyTorch Lightning checkpoint: {val_acc:.2f}_model_{model_name}.pth")
    print(f"2. Model weights only: {val_acc:.2f}_model_{model_name}_weights.pt") 
    # print(f"3. TorchScript format: {val_acc:.2f}_model_{model_name}_scripted.pt")
    print(f"4. ONNX format: {val_acc:.2f}_model_{model_name}.onnx")
    
if __name__ == "__main__":
    main(args)

