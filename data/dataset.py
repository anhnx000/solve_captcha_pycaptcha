import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import PIL
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.config_util import configGetter

# Add DALI imports
try:
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    import nvidia.dali.ops as ops
    from nvidia.dali.backend import TensorGPU, TensorCPU
    import nvidia.dali.plugin.pytorch as dalitorch
    import nvidia.dali.plugin.numba as dalinumba
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    print("NVIDIA DALI not found. GPU-accelerated data loading will not be available.")

cfg = configGetter('DATASET')

HEIGHT = cfg['CAPTCHA']['IMG_HEIGHT']
WIDTH = cfg['CAPTCHA']['IMG_WIDTH']
CLASS_NUM = cfg['CAPTCHA']['CLASS_NUM']
CHAR_LEN = cfg['CAPTCHA']['CHAR_LEN']


class captcha_dataset(data.Dataset):
    def __init__(self, data_path_tail: str) -> None:
        super().__init__()

        
        self.data_path = './dataset/' + data_path_tail
        
        self.data_list = os.listdir(self.data_path)
        
        # Replace torchvision transforms with Albumentations
        self.transform = A.Compose([
            # center crop 
            A.CenterCrop(height=50-5, width=200-5, p=0.1),
            # rotation 5 degrees
            A.Rotate(limit=5, p=0.2),
            
            # rotate 10 degrees
            A.Rotate(limit=10, p=0.2),
            
            # brightness/contrast adjustments - equivalent to ColorJitter
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.2),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.2),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.01, p=0.1),
            
            # dịch toàn bộ ảnh sang trái 20 pixel
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=0, p=0.2),
            
            # dịch toàn bộ ảnh sang phải 20 pixel
            A.ShiftScaleRotate(shift_limit=-0.15, scale_limit=0.2, rotate_limit=0, p=0.2),
            
            # dịch toàn bộ ảnh lên trên 10 pixel, có 50 pixel chiều cao
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.15, rotate_limit=0, p=0.2),
            
            # dịch toàn bộ ảnh xuống dưới 0.2 %
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, p=0.2),
            
                  # dịch toàn bộ ảnh sang trái 10%
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.1),
            
            # dịch toàn bộ ảnh sang phải 10%
            A.ShiftScaleRotate(shift_limit=-0.1, scale_limit=0.1, rotate_limit=0, p=0.1),
            
            # dịch toàn bộ ảnh lên trên 10%
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.1),
            # dịch toàn bộ ảnh xuống dưới 10%
            A.ShiftScaleRotate(shift_limit=-0.1, scale_limit=0.1, rotate_limit=0, p=0.1),
            
            # làm mờ nhẹ ảnh
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
            
            # làm blur ảnh
            # A.GaussianBlur(blur_limit=2, p = 0.2),
            A.GaussianBlur(blur_limit=(1, 3), p = 0.2),
            
            # resize
            A.Resize(height=224, width=224),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        
    def __getitem__(self, index):
        # Thử lấy ảnh với index hiện tại, nếu lỗi thì thử lấy ảnh tiếp theo
        max_retries = 10  # Số lần thử tối đa
        for attempt in range(max_retries):
            try:
                idx = (index + attempt) % len(self.data_list)  # Tránh vượt quá giới hạn
                img_path = os.path.join(self.data_path, self.data_list[idx])
                image = np.array(Image.open(img_path))
                transformed = self.transform(image=image)
                img = transformed["image"]
                
                # label = self.data_list[idx].split('.')[0].lower() 
                label = self.data_list[idx].split('.')[0]
                if len(label) < CHAR_LEN:
                    label = label + '_' * (CHAR_LEN - len(label))
                
                return img, str_to_vec(label)
            except (PIL.UnidentifiedImageError, OSError, IOError) as e:
                print(f"Lỗi khi load {img_path}, thử lại: {e}")
                continue
        
        # Nếu tất cả các lần thử đều thất bại, trả về một mẫu mặc định hoặc raise lỗi
        # Hoặc có thể tạo một ảnh trắng với kích thước đúng
        blank_img = torch.zeros(3, HEIGHT, WIDTH)
        blank_label = str_to_vec("_" * CHAR_LEN)  # Tạo nhãn mặc định với độ dài CHAR_LEN # _ là kí tự đặc biệt blank 
        return blank_img, blank_label

    def __len__(self):
        return len(self.data_list)

def str_to_lst(s: str):
    lst = []
    for c in s:
        if '0' <= c <= '9':
            lst.append(ord(c) - ord('0'))
        elif 'a' <= c <= 'z':
            lst.append(ord(c) - ord('a') + 10)
        elif 'A' <= c <= 'Z':
            lst.append(ord(c) - ord('A') + 36)
        elif c == '_':
            lst.append(62)
    return lst


def list_to_str(lst: list):
    """
    dựa vào hàm str_to_lst để chuyển đổi list các số từ 0 đến 62 thành list các kí tự
    
    input: list các số từ 0 đến 62
    output: string
    """
    s = ''
    for i in lst:
        if isinstance(i, torch.Tensor):
            i = i.item()
        if 0 <= i < 10:
            s += chr(i + ord('0'))
        elif 10 <= i < 36:
            s += chr(i + ord('a') - 10)
        elif 36 <= i < 62:
            s += chr(i + ord('A') - 36)
        elif i == 62:
            s += '_'
    return s
  


# def lst_to_str(lst: list):

#     s = ''
#     for i in lst:
#         if type(i) == torch.Tensor:
#             i = i.item()
#         if i < 10:
#             s += chr(i + ord('0'))
#         elif i < 36:
#             s += chr(i + ord('a') - 10)
#         else:
#             s += chr(i + ord('A') - 36)
#         if i == 62:
#             s += '_'
#     return s


def str_to_onehotvec(s: str):
    return torch.nn.functional.one_hot(torch.LongTensor(str_to_lst(s)), CLASS_NUM)


def str_to_vec(s: str):
    return torch.LongTensor(str_to_lst(s))


# Conditionally define DALI functions only if DALI is available
if DALI_AVAILABLE:
    # Define your Albumentations transforms that will be used inside DALI
    def get_album_transforms():
        return A.Compose([
            # center crop 
            A.CenterCrop(height=50-5, width=200-5, p=0.1),
            # rotation 5 degrees
            A.Rotate(limit=5, p=0.2),
            
            # rotate 10 degrees
            A.Rotate(limit=10, p=0.2),
            
            # brightness/contrast adjustments - equivalent to ColorJitter
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.2),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.2),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.01, p=0.1),
            
            # dịch toàn bộ ảnh sang trái 20 pixel
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=0, p=0.2),
            
            # dịch toàn bộ ảnh sang phải 20 pixel
            A.ShiftScaleRotate(shift_limit=-0.15, scale_limit=0.2, rotate_limit=0, p=0.2),
            
            # dịch toàn bộ ảnh lên trên 10 pixel, có 50 pixel chiều cao
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.15, rotate_limit=0, p=0.2),
            
            # dịch toàn bộ ảnh xuống dưới 0.2 %
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, p=0.2),
            
            # dịch toàn bộ ảnh sang trái 10%
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.1),
            
            # dịch toàn bộ ảnh sang phải 10%
            A.ShiftScaleRotate(shift_limit=-0.1, scale_limit=0.1, rotate_limit=0, p=0.1),
            
            # dịch toàn bộ ảnh lên trên 10%
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.1),
            # dịch toàn bộ ảnh xuống dưới 10%
            A.ShiftScaleRotate(shift_limit=-0.1, scale_limit=0.1, rotate_limit=0, p=0.1),
            
            # làm mờ nhẹ ảnh
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
            
            # làm blur ảnh
            A.GaussianBlur(blur_limit=(1, 3), p = 0.2),
            
            # resize
            A.Resize(height=224, width=224),
        ])

    # Python function that will be used as DALI External Source operator
    def albumentations_transform(image_tensor):
        # Convert DALI tensor to numpy array
        image_np = np.array(image_tensor)
        
        # Get Albumentations transforms
        album_transforms = get_album_transforms()
        
        # Apply transforms
        try:
            transformed = album_transforms(image=image_np)
            transformed_image = transformed["image"]
            return transformed_image
        except Exception as e:
            print(f"Error applying Albumentations: {e}")
            return image_np

    @pipeline_def
    def captcha_dali_pipeline(data_dir, file_list, shuffle=True, batch_size=32, num_threads=4, device_id=0):
        """
        DALI pipeline for captcha dataset that loads and preprocesses images directly on GPU
        using Albumentations through Python operator
        
        Args:
            data_dir: Directory containing the images
            file_list: Path to file containing image filenames and labels, one per line
            shuffle: Whether to shuffle the dataset
            batch_size: Batch size
            num_threads: Number of CPU threads to use
            device_id: GPU device ID
        """
        jpegs, labels = fn.readers.file(
            file_root=data_dir,
            file_list=file_list,
            random_shuffle=shuffle,
            num_threads=num_threads,
            shard_id=0,
            num_shards=1,
            stick_to_shard=False,
            pad_last_batch=True,  # Pad the last batch if it's not full
            name="file_reader"
        )
        
        # Decode images consistently to the same size
        images = fn.decoders.image(
            jpegs, 
            device="cpu",  # Decode on CPU since Albumentations runs on CPU
            output_type=types.RGB
        )
        
        # Option 1: Apply Albumentations transforms using Python operator
        # This will run on CPU, but the result will be transferred to GPU
        try:
            augmented_images = fn.python_function(images, function=albumentations_transform)
            
            # Ensure consistent output shape
            augmented_images = fn.resize(
                augmented_images,
                size=[HEIGHT, WIDTH],
                interp_type=types.INTERP_LINEAR
            )
            
            # Convert to PyTorch tensor format and transfer to GPU
            final_images = fn.crop_mirror_normalize(
                augmented_images,
                dtype=types.FLOAT,
                output_layout="CHW",
                device="gpu",
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            )
        except Exception as e:
            print(f"Error in Albumentations transform: {e}")
            # Fallback to DALI native operations if Albumentations fails
            images = fn.resize(
                images,
                size=[HEIGHT, WIDTH],
                interp_type=types.INTERP_LINEAR
            )
            
            final_images = fn.crop_mirror_normalize(
                images,
                dtype=types.FLOAT,
                output_layout="CHW",
                device="gpu",
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            )
        
        # Convert string labels to tensor of indices
        label_strings = fn.element_extract(labels, element_map=["file_id"])
        label_chars = fn.strings.substr(label_strings, 0, length=CHAR_LEN)
        
        # Return images on GPU, labels on CPU
        return final_images, label_chars


    class DALICaptchaIterator(DALIGenericIterator):
        """
        DALI iterator for captcha dataset that converts labels to the expected format
        """
        def __init__(self, pipelines, label_map, size, auto_reset=True, 
                    last_batch_policy=LastBatchPolicy.PARTIAL, prepare_first_batch=True,
                    dynamic_shape=False, last_batch_padded=False):
            self.label_map = label_map  # Mapping from characters to indices
            
            # Handle last batch policy based on parameters
            if last_batch_padded:
                last_batch_policy = LastBatchPolicy.FILL
            
            super().__init__(
                pipelines,
                ["images", "labels"],
                size,
                auto_reset=auto_reset,
                last_batch_policy=last_batch_policy,
                prepare_first_batch=prepare_first_batch,
                dynamic_shape=dynamic_shape
            )
            
            self.dynamic_shape = dynamic_shape
            
        def __next__(self):
            try:
                batch = super().__next__()[0]
                # Convert string labels to tensors
                labels = batch["labels"]
                tensor_labels = []
                
                for label_str in labels:
                    # Convert label string to the format expected by the model
                    label_str = label_str.decode() if isinstance(label_str, bytes) else str(label_str)
                    # Pad with '_' if needed
                    if len(label_str) < CHAR_LEN:
                        label_str = label_str + '_' * (CHAR_LEN - len(label_str))
                    # Ensure label has exactly CHAR_LEN characters
                    label_str = label_str[:CHAR_LEN]
                    tensor_labels.append(str_to_vec(label_str))
                
                # Stack tensor labels and move to the same device as images
                tensor_labels_stacked = torch.stack(tensor_labels)
                
                # Ensure images have consistent dimensions
                images = batch["images"]
                
                # Check and make images uniform if needed
                if self.dynamic_shape:
                    # Handle the case where images might have inconsistent shapes
                    if isinstance(images, torch.Tensor) and len(images.shape) == 4:
                        # Images are already tensors with the right shape - no need to do anything
                        pass
                    else:
                        # This should not normally happen, but just in case
                        print(f"Warning: Unexpected image shape: {images.shape}")
                        # Convert to a standard shape if needed
                
                # Move labels to the same device as images
                labels_tensor = tensor_labels_stacked.to(images.device)
                
                # Return as a tuple (images, labels) to match the expected format
                # instead of returning a dictionary
                return images, labels_tensor
            except StopIteration:
                raise
            except Exception as e:
                print(f"Error in DALI iterator: {e}")
                # Create a fallback batch with zeros
                images = torch.zeros((self.batch_size, 3, HEIGHT, WIDTH), device="cuda")
                labels = torch.zeros((self.batch_size, CHAR_LEN), dtype=torch.long, device="cuda")
                return images, labels


    def create_dali_file_list(data_dir, output_file='file_list.txt'):
        """
        Create a file list for DALI with format: /path/to/image.jpg label
        """
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w') as f:
            valid_count = 0
            for filename in os.listdir(data_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        # Verify the image can be opened
                        img_path = os.path.join(data_dir, filename)
                        Image.open(img_path)
                        
                        # Extract label
                        label = filename.split('.')[0]
                        f.write(f'{filename} {label}\n')
                        valid_count += 1
                    except (PIL.UnidentifiedImageError, OSError, IOError) as e:
                        print(f"Skipping invalid image {filename}: {e}")
                        continue
            
            print(f"Created file list with {valid_count} valid images from {data_dir}")
        return output_file


# Stub implementations for when DALI is not available
else:
    def captcha_dali_pipeline(*args, **kwargs):
        raise ImportError("NVIDIA DALI is not installed. Cannot use captcha_dali_pipeline.")
        
    class DALICaptchaIterator:
        def __init__(self, *args, **kwargs):
            raise ImportError("NVIDIA DALI is not installed. Cannot use DALICaptchaIterator.")
            
    def create_dali_file_list(data_dir, output_file='file_list.txt'):
        raise ImportError("NVIDIA DALI is not installed. Cannot use create_dali_file_list.")


if __name__ == '__main__':
    d = captcha_dataset('./dataset', 'train')
    a = d[0]
    print(a[0].size(), a[1].size(), list_to_str(a[1]))
    
    # Test DALI if available
    if DALI_AVAILABLE:
        print("Testing DALI pipeline...")
        # Create a file list
        file_list = create_dali_file_list('./dataset/train')
        # Create pipeline
        pipe = captcha_dali_pipeline(
            batch_size=32,
            num_threads=4,
            device_id=0,
            data_dir='./dataset/train',
            file_list=file_list
        )
        pipe.build()
        # Create iterator
        label_map = {chr(i + ord('0')): i for i in range(10)}
        label_map.update({chr(i + ord('a') - 10): i+10 for i in range(10, 36)})
        label_map.update({chr(i + ord('A') - 36): i+36 for i in range(36, 62)})
        label_map['_'] = 62
        
        dali_iter = DALICaptchaIterator(
            [pipe],
            label_map,
            pipe.epoch_size("Reader"),
            dynamic_shape=True,
            last_batch_padded=True
        )
        
        # Get first batch
        batch = next(dali_iter)
        print(f"DALI batch - images shape: {batch[0].shape}, labels shape: {batch[1].shape}")
