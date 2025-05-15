from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from data.dataset import captcha_dataset, DALI_AVAILABLE
from torch.utils.data import DataLoader
import torch 
import torchvision.transforms as transforms
import os
from PIL import Image
import PIL

# Import DALI components if available
if DALI_AVAILABLE:
    from data.dataset import (
        captcha_dali_pipeline, 
        DALICaptchaIterator, 
        create_dali_file_list
    )

def collate_fn_ignore_errors(batch):
    # Lọc bỏ các phần tử None (đại diện cho các item lỗi)
    batch = [item for item in batch if item is not None]
    
    # Nếu không còn phần tử nào sau khi lọc thì trả về batch rỗng để tránh lỗi
    if len(batch) == 0:
        return None
    
    try:
        # Kiểm tra kích thước của images trong batch
        img_shapes = [item[0].shape for item in batch]
        label_shapes = [item[1].shape for item in batch]
        
        # Kiểm tra xem có sự không nhất quán về kích thước không
        if len(set(str(s) for s in img_shapes)) > 1:
            print(f"Warning: Inconsistent image shapes in batch: {img_shapes}")
            # Lọc chỉ giữ lại các items có kích thước phổ biến nhất
            most_common_shape = max(set(img_shapes), key=img_shapes.count)
            batch = [item for item in batch if item[0].shape == most_common_shape]
        
        # Thực hiện default_collate với batch đã được chuẩn hóa
        return torch.utils.data.default_collate(batch)
    except RuntimeError as e:
        print(f"Collate error: {e}")
        # Trong trường hợp lỗi, trả về một batch an toàn từ item đầu tiên
        if len(batch) > 0:
            img_shape = batch[0][0].shape
            label_shape = batch[0][1].shape
            
            # Tạo một batch đồng nhất từ item đầu tiên
            imgs = torch.stack([batch[0][0] for _ in range(len(batch))])
            labels = torch.stack([batch[0][1] for _ in range(len(batch))])
            return imgs, labels
        return None

# Wrapper cho DALI iterator để đảm bảo batch format nhất quán với PyTorch DataLoader
class DALIDataLoader:
    def __init__(self, dali_iterator):
        self.dali_iterator = dali_iterator
        
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            # DALI iterator trả về tuple (images, labels)
            return next(self.dali_iterator)
        except StopIteration:
            raise StopIteration

class captcha_dm(LightningDataModule):
    def __init__(self, batch_size=20, num_workers=8, use_dali=False):
        super(captcha_dm,self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_dali = use_dali and DALI_AVAILABLE
        
        if self.use_dali:
            print("Using NVIDIA DALI for GPU-accelerated data loading")
        elif use_dali and not DALI_AVAILABLE:
            print("NVIDIA DALI requested but not available. Falling back to PyTorch DataLoader.")
            print("To install DALI, run: pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110")

    def setup(self, stage:  Optional[str]) -> None:
        # Standard PyTorch datasets (used as fallback if DALI is not available)
        train_dataset_1  = captcha_dataset('train')
        train_dataset_2  = captcha_dataset('../dataset_real/train')
        train_dataset_3 = torch.utils.data.ConcatDataset([train_dataset_2] * 20)
        self.train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_3])
        
        val_dataset_1  = captcha_dataset('val')
        val_dataset_2  = captcha_dataset('../dataset_real/val')
        self.val_dataset = torch.utils.data.ConcatDataset([val_dataset_1, val_dataset_2])
        
        # Setup for DALI if enabled
        if self.use_dali:
            try:
                # Create file lists for DALI
                self.train_file_list = create_dali_file_list('./dataset/train', 'train_file_list.txt')
                self.train_real_file_list = create_dali_file_list('../dataset_real/train', 'train_real_file_list.txt')
                self.val_file_list = create_dali_file_list('./dataset/val', 'val_file_list.txt')
                self.val_real_file_list = create_dali_file_list('../dataset_real/val', 'val_real_file_list.txt')
                
                # Create character to index mapping
                self.label_map = {chr(i + ord('0')): i for i in range(10)}
                self.label_map.update({chr(i + ord('a') - 10): i+10 for i in range(10, 36)})
                self.label_map.update({chr(i + ord('A') - 36): i+36 for i in range(36, 62)})
                self.label_map['_'] = 62
                
                # Get dataset sizes
                self.train_size = len(os.listdir('./dataset/train')) + len(os.listdir('../dataset_real/train')) * 20
                self.val_size = len(os.listdir('./dataset/val')) + len(os.listdir('../dataset_real/val'))
            except (ImportError, Exception) as e:
                print(f"Error setting up DALI: {e}")
                print("Falling back to standard PyTorch DataLoader")
                self.use_dali = False
    
    def train_dataloader(self):
        if not self.use_dali:
            # Standard PyTorch DataLoader
            is_shuffle = False
            return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                             num_workers=self.num_workers, shuffle=is_shuffle, 
                             collate_fn=collate_fn_ignore_errors,
                             persistent_workers=True if self.num_workers > 0 else False)
        else:
            try:
                # DALI pipeline for training
                device_id = 0  # Assume first GPU
                
                # Create pipelines for synthetic and real datasets
                pipe1 = captcha_dali_pipeline(
                    batch_size=self.batch_size,
                    num_threads=self.num_workers,
                    device_id=device_id,
                    data_dir='./dataset/train',
                    file_list=self.train_file_list,
                    shuffle=True
                )
                
                pipe2 = captcha_dali_pipeline(
                    batch_size=self.batch_size,
                    num_threads=self.num_workers,
                    device_id=device_id,
                    data_dir='../dataset_real/train',
                    file_list=self.train_real_file_list,
                    shuffle=True
                )
                
                # Build pipelines
                pipe1.build()
                pipe2.build()
                
                # Create iterators
                train_iter1 = DALICaptchaIterator(
                    [pipe1],
                    self.label_map,
                    pipe1.epoch_size("Reader"),
                    dynamic_shape=True,  # Allow dynamic shape adjustment
                    last_batch_padded=True  # Pad last batch if needed
                )
                
                train_iter2 = DALICaptchaIterator(
                    [pipe2],
                    self.label_map,
                    pipe2.epoch_size("Reader") * 20,  # Multiply by 20 as in the original code
                    dynamic_shape=True,  # Allow dynamic shape adjustment
                    last_batch_padded=True  # Pad last batch if needed
                )
                
                # Wrap DALI iterators with DALIDataLoader for consistent format
                return [DALIDataLoader(train_iter1), DALIDataLoader(train_iter2)]
            except Exception as e:
                print(f"Error creating DALI train dataloader: {e}")
                print("Falling back to standard PyTorch DataLoader")
                is_shuffle = False
                return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                num_workers=self.num_workers, shuffle=is_shuffle, 
                                collate_fn=collate_fn_ignore_errors,
                                persistent_workers=True if self.num_workers > 0 else False)
    
    def val_dataloader(self):
        if not self.use_dali:
            # Standard PyTorch DataLoader
            return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                             num_workers=self.num_workers, shuffle=False, 
                             collate_fn=collate_fn_ignore_errors,
                             persistent_workers=True if self.num_workers > 0 else False)
        else:
            try:
                # DALI pipeline for validation
                device_id = 0  # Assume first GPU
                
                # Create pipelines for synthetic and real validation datasets
                pipe1 = captcha_dali_pipeline(
                    batch_size=self.batch_size,
                    num_threads=self.num_workers,
                    device_id=device_id,
                    data_dir='./dataset/val',
                    file_list=self.val_file_list,
                    shuffle=False
                )
                
                pipe2 = captcha_dali_pipeline(
                    batch_size=self.batch_size,
                    num_threads=self.num_workers,
                    device_id=device_id,
                    data_dir='../dataset_real/val',
                    file_list=self.val_real_file_list,
                    shuffle=False
                )
                
                # Build pipelines
                pipe1.build()
                pipe2.build()
                
                # Create iterators
                val_iter1 = DALICaptchaIterator(
                    [pipe1],
                    self.label_map,
                    pipe1.epoch_size("Reader"),
                    dynamic_shape=True,  # Allow dynamic shape adjustment
                    last_batch_padded=True  # Pad last batch if needed
                )
                
                val_iter2 = DALICaptchaIterator(
                    [pipe2],
                    self.label_map,
                    pipe2.epoch_size("Reader"),
                    dynamic_shape=True,  # Allow dynamic shape adjustment
                    last_batch_padded=True  # Pad last batch if needed
                )
                
                # Wrap DALI iterators with DALIDataLoader for consistent format
                return [DALIDataLoader(val_iter1), DALIDataLoader(val_iter2)]
            except Exception as e:
                print(f"Error creating DALI validation dataloader: {e}")
                print("Falling back to standard PyTorch DataLoader")
                return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                                num_workers=self.num_workers, shuffle=False, 
                                collate_fn=collate_fn_ignore_errors,
                                persistent_workers=True if self.num_workers > 0 else False)
    
    def test_dataloader(self):
        # If test_dataset is not set up in setup(), this will fail
        if hasattr(self, 'test_dataset'):
            return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                             num_workers=self.num_workers, shuffle=False, 
                             collate_fn=collate_fn_ignore_errors,
                             persistent_workers=True if self.num_workers > 0 else False)
        else:
            # Use validation dataset as test dataset
            return self.val_dataloader()
    
    
if __name__ == '__main__':
    # Test standard dataloader
    dm = captcha_dm()
    dm.setup(stage=None)
    it = dm.train_dataloader()
    print("Standard DataLoader:", next(iter(it)))
    
    # Test DALI dataloader if available
    if DALI_AVAILABLE:
        try:
            dm_dali = captcha_dm(use_dali=True)
            dm_dali.setup(stage=None)
            it_dali = dm_dali.train_dataloader()
            batch = next(iter(it_dali[0]))
            print("DALI DataLoader (first iterator):", type(batch), "images shape:", batch[0].shape, "labels shape:", batch[1].shape)
        except Exception as e:
            print(f"Error testing DALI dataloader: {e}")
