import pytorch_lightning as pl
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
from data.dataset import HEIGHT, WIDTH, CLASS_NUM, CHAR_LEN, list_to_str
import wandb
import os 
        # Import transformers library
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def eval_acc(label: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # label: CHAR_LEN x batchsize
    # pred: CHAR_LEN x batchsize x CLASS_NUM
    pred_res = pred.argmax(dim=2) # CHAR_LEN x batchsize
    eq = ((pred_res == label).float().sum(dim=0)==CHAR_LEN).float() #batchsize
    return eq.sum()/eq.size(0)

class captcha_model(pl.LightningModule):
    def __init__(self, model, lr=1e-4, optimizer=None, use_ctc=False):
        pl.LightningModule.__init__(self)
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.use_ctc = use_ctc
        
        # Add a blank token for CTC loss (using the last class index)
        self.blank_index = CLASS_NUM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.model(x)
        return x

    def step(self, batch, batch_idx) -> tuple:
        x, y = batch
        y_hat = self(x)
        
        if self.use_ctc:
            # For CTC loss, reshape the output:
            # [batch_size, CHAR_LEN, CLASS_NUM] -> [CHAR_LEN, batch_size, CLASS_NUM+1]
            # Add one extra dimension for blank token
            log_probs = F.log_softmax(y_hat, dim=2)
            
            # Add blank token probability column
            blank_probs = torch.zeros(log_probs.shape[0], log_probs.shape[1], 1).to(log_probs.device)
            log_probs_with_blank = torch.cat([log_probs, blank_probs], dim=2)
            
            # Reshape for CTC: [batch_size, CHAR_LEN, CLASS_NUM+1] -> [CHAR_LEN, batch_size, CLASS_NUM+1]
            log_probs_with_blank = log_probs_with_blank.permute(1, 0, 2)
            
            # Convert targets to proper format for CTC
            y = y.permute(1, 0)  # [CHAR_LEN, batch_size]
            
            # Calculate input lengths (all the same: CHAR_LEN)
            input_lengths = torch.full((x.size(0),), CHAR_LEN, dtype=torch.long).to(y.device)
            
            # Calculate target lengths (all the same: CHAR_LEN)
            target_lengths = torch.full((x.size(0),), CHAR_LEN, dtype=torch.long).to(y.device)
            
            # Flatten labels
            y_flat = y.reshape(-1)
            
            # Calculate CTC loss
            loss = F.ctc_loss(
                log_probs_with_blank, y_flat, input_lengths, target_lengths,
                blank=self.blank_index, reduction='mean', zero_infinity=True
            )
            
            # For accuracy calculation, convert back to original format
            y_hat_permuted = log_probs.permute(1, 0, 2)
            
            return loss, y, y_hat_permuted
        else:
            # Original implementation without CTC
            y_hat_permuted = y_hat.permute(1, 0, 2)
            y_permuted = y.permute(1, 0)
            loss = 0
            for i in range(CHAR_LEN):
                loss += F.cross_entropy(y_hat_permuted[i], y_permuted[i])
                
            return loss, y_permuted, y_hat_permuted

    def training_step(self, batch, batch_idx) -> dict:
        loss, label, y_pred = self.step(batch, batch_idx)
        # log label and y to file './logs/train_label_y.txt'
      
        import os
        
        # Ensure logs directory exists
        os.makedirs('./logs', exist_ok=True)
       
        # if batch_idx == 0:
        #     if os.path.exists('./logs/train_label_y.txt'):
        #         with open('./logs/train_label_y.txt', 'a') as f:
        #             f.write(f"****************************\n")
        #             f.write(f"label: {label[1]}\n")
        #             f.write(f"----------------------------\n")
        #             f.write(f"y: {y_pred[1]}\n")
            
        #     else:
        #         with open('./logs/train_label_y.txt', 'w') as f:
        #             f.write(f"****************************\n")
        #             f.write(f"label: {label[1]}\n")
        #             f.write(f"----------------------------\n")
        #             f.write(f"y: {y_pred[1]}\n")
        
        
        eval_acc_score = eval_acc(label, y_pred)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", eval_acc_score, on_step=True, on_epoch=True, prog_bar=True)
        
        
        if batch_idx % 100 == 0:
            # Log directly using wandb
            wandb.log({"train_loss_step": loss.item(), "train_acc_step": eval_acc_score})
            
            # Check if optimizer exists before logging learning rate
            # Note: In PyTorch Lightning, you can get optimizer through trainer
            # if self.trainer and hasattr(self.trainer, 'optimizers') and self.trainer.optimizers:
            wandb.log({"learning_rate": self.trainer.optimizers[0].param_groups[0]['lr']})
            
        # Store outputs for epoch-end collection
        self.train_step_outputs = getattr(self, 'train_step_outputs', [])
        self.train_step_outputs.append({"loss": loss, "train_loss": loss.item(), "train_acc": eval_acc_score})
        
        return {"loss": loss}

    def on_train_epoch_end(self):
        # Calculate average metrics across all batches in epoch
        if hasattr(self, 'train_step_outputs') and self.train_step_outputs:
            avg_loss = torch.stack([x["loss"] for x in self.train_step_outputs]).mean()
            avg_train_loss = sum([x["train_loss"] for x in self.train_step_outputs]) / len(self.train_step_outputs)
            avg_train_acc = sum([x["train_acc"] for x in self.train_step_outputs]) / len(self.train_step_outputs)
            
            # Log epoch-level metrics to wandb
            wandb.log({
                "train_loss_epoch": avg_train_loss,
                "train_acc_epoch": avg_train_acc,
                "epoch": self.current_epoch
            })
            
            # Reset outputs
            self.train_step_outputs = []

    def validation_step(self, batch, batch_idx):
        loss, label, y = self.step(batch, batch_idx)
        # log label and y to file './logs/val_label_y.txt'
        import os
        
        # Ensure logs directory exists
        os.makedirs('./logs', exist_ok=True)
        
        # # chuyển label và y thành string và lưu lại vào file './logs/val_label_y.txt'
        # with open('./logs/val_label_y.txt', 'a') as f:
        #     f.write(f"****************************\n")
        #     for i in range(label.size(0)):
        #         f.write(f"----------------------------\n")
        #         label_str = lst_to_str(label[i].tolist())
        #         y_str = lst_to_str(y[i].tolist())
                
        #         f.write(f"label: {label_str}\n")
        #         f.write(f"y: {y_str}\n")
            
        eval_acc_score = eval_acc(label, y)
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", eval_acc_score, on_step=False, on_epoch=True, prog_bar=True)
        wandb.log({"val_loss_step": loss.item(), "val_acc_step": eval_acc_score})
        
        # Store metrics for epoch-end logging
        self.val_step_outputs = getattr(self, 'val_step_outputs', [])
        self.val_step_outputs.append({"val_loss": loss.item(), "val_acc": eval_acc_score})
        
        return {"val_loss": loss.item(), "val_acc": eval_acc_score}
        
    def on_validation_epoch_end(self):
        # Log epoch-level metrics to wandb
        if hasattr(self, 'val_step_outputs') and self.val_step_outputs:
            avg_loss = sum([x["val_loss"] for x in self.val_step_outputs]) / len(self.val_step_outputs)
            avg_acc = sum([x["val_acc"] for x in self.val_step_outputs]) / len(self.val_step_outputs)
            
            # Log to wandb with epoch info
            wandb.log({
                "val_loss_epoch": avg_loss,
                "val_acc_epoch": avg_acc,
                "epoch": self.current_epoch
            })
            
            # Reset outputs
            self.val_step_outputs = []

    def test_step(self, batch, batch_idx):
        loss, label, y = self.step(batch, batch_idx)
        self.log("test_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        eval_acc_score = eval_acc(label, y)
        self.log("test_acc", eval_acc_score, on_step=False, on_epoch=True, prog_bar=True)
        wandb.log({"test_loss": loss.item(), "test_acc": eval_acc_score})
        if batch_idx == 0:
            if self.use_ctc:
                # Convert back to [batch_size, CHAR_LEN] for display
                label = label.permute(1, 0) if label.dim() > 1 else label.unsqueeze(0)
                y = y.permute(1, 0, 2) if y.dim() > 2 else y.unsqueeze(0)
            else:
                label = label.permute(1, 0)
                y = y.permute(1, 0, 2)
            pred = y.argmax(dim=2)
            res = [f"pred:{lst_to_str(pred[i])}, true:{lst_to_str(label[i])}" for i in range(pred.size(0))]
            print("\n".join(res))
            wandb.log({"test_pred": res})
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = self.optimizer
        return optimizer


class model_resnet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.resnet = models.resnet18(weights=None)
        # self.resnet.fc = nn.Linear(512, CHAR_LEN*CLASS_NUM)
        self.resnet.fc = nn.Linear(512, 5*63)
        
        # self.resnet = models.resnext101_32x8d(weights=True)
        # self.resnet.fc = nn.Linear(2048, CHAR_LEN*CLASS_NUM)

        # # use EfficientNetV2-S
        # self.resnet = models.efficientnet_v2_s(weights=True)
        # self.resnet.classifier = nn.Linear(1280, CHAR_LEN*CLASS_NUM)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        # x = x.view(x.size(0), CHAR_LEN, CLASS_NUM)
        x = x.view(x.size(0), 5, 63)
        return x

class model_efficientnet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.efficientnet = models.efficientnet_v2_s(weights=None)
        self.efficientnet.classifier = nn.Linear(1280, CHAR_LEN*CLASS_NUM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet(x)
        x = x.view(x.size(0), CHAR_LEN, CLASS_NUM)
        return x
    

class model_vit(torch.nn.Module):
    # use vision transformer  ocr pretrain on ocr dataset
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.vit = models.vit_b_16(weights=True)
        self.vit.heads = nn.Linear(768, CHAR_LEN*CLASS_NUM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        x = x.view(x.size(0), CHAR_LEN, CLASS_NUM)
        return x
     
class model_mobilenet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.mobilenet = models.mobilenet_v3_small(weights=False)
        self.mobilenet.classifier = nn.Linear(576, CHAR_LEN*CLASS_NUM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mobilenet(x)
        x = x.view(x.size(0), CHAR_LEN, CLASS_NUM)
        return x
     
     
     
     
     

# use TrOcr pretrain 
class model_trocr(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        
        # Load pretrained TrOCR model
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Replace the decoder head for our specific captcha task
        self.model.decoder.lm_head = nn.Linear(self.model.decoder.config.hidden_size, CHAR_LEN*CLASS_NUM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TrOCR expects images in a different format, so we may need to adapt the input
        # This is a simplified version; you might need to adjust based on your requirements
        features = self.model.encoder(x).last_hidden_state
        outputs = self.model.decoder(inputs_embeds=features)
        logits = outputs.logits
        
        # Reshape to match the expected captcha format
        x = logits[:, -1].view(logits.size(0), CHAR_LEN, CLASS_NUM)
        return x
    

if __name__ == "__main__":
    # Create a properly shaped 3D tensor (CHAR_LEN x batchsize x CLASS_NUM)
    pred = torch.zeros(5, 1, 10)  # Assuming CHAR_LEN=5, batchsize=1, CLASS_NUM=10
    
    # Set the highest probability for each character
    for i in range(5):
        pred[i, 0, i+1] = 1.0
    
    # Create a label tensor (CHAR_LEN x batchsize)
    label = torch.tensor([[1, 2, 3, 3, 5]]).transpose(0, 1)  # Shape: 5x1
    
    print(eval_acc(label, pred))
