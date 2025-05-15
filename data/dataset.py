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
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0, p=0.2),
            
            # dịch toàn bộ ảnh sang phải 20 pixel
            A.ShiftScaleRotate(shift_limit=-0.2, scale_limit=0.2, rotate_limit=0, p=0.2),
            
            # dịch toàn bộ ảnh lên trên 10 pixel, có 50 pixel chiều cao
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0, p=0.2),
            
            # dịch toàn bộ ảnh xuống dưới 0.2 %
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0, p=0.2),
            
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
            A.GaussianBlur(blur_limit=3, p=0.2),
            
            # resize
            A.Resize(height=HEIGHT, width=WIDTH),
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


if __name__ == '__main__':
    d = captcha_dataset('./dataset', 'train')
    a = d[0]
    print(a[0].size(), a[1].size(), list_to_str(a[1]))
