from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from data.dataset import captcha_dataset
from torch.utils.data import DataLoader
import torch 
import torchvision.transforms as transforms
import os
from PIL import Image
import PIL

def collate_fn_ignore_errors(batch):
    # Lọc bỏ các phần tử None (đại diện cho các item lỗi)
    batch = [item for item in batch if item is not None]
    
    # Nếu không còn phần tử nào sau khi lọc thì trả về batch rỗng để tránh lỗi
    if len(batch) == 0:
        return None
    
    # Sử dụng default_collate cho các phần tử còn lại
    return torch.utils.data.default_collate(batch)

class captcha_dm(LightningDataModule):
    def __init__(self, batch_size=20, num_workers=8):
        super(captcha_dm,self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage:  Optional[str]) -> None:
        # train_dataset_1  = captcha_dataset('train')
        # train_dataset_2  = captcha_dataset('../dataset_real/train')
        # train_dataset_3 = torch.utils.data.ConcatDataset([train_dataset_2] * 250)
        # self.train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_3])
        
        # val_dataset_1  = captcha_dataset('val')
        # val_dataset_2  = captcha_dataset('../dataset_real/val')
        # self.val_dataset = torch.utils.data.ConcatDataset([val_dataset_1, val_dataset_2])
        
        # # test_dataset_1  = captcha_dataset('test')
        # # test_dataset_2  = captcha_dataset('../dataset_real/test')
        # # self.test_dataset = torch.utils.data.ConcatDataset([test_dataset_1, test_dataset_2])
        
        self.train_dataset = captcha_dataset('train')
        self.val_dataset = captcha_dataset('val')        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=collate_fn_ignore_errors)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn_ignore_errors)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn_ignore_errors)
    
    
if __name__ == '__main__':
    dm = captcha_dm()
    dm.setup(stage=None)
    it = dm.train_dataloader()
    print(next(iter(it)))
