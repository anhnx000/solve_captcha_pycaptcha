import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import PIL
import os
from torchvision.transforms.transforms import Resize
from utils.config_util import configGetter
from torch.utils.data import DataLoader

cfg = configGetter('DATASET')

HEIGHT = cfg['CAPTCHA']['IMG_HEIGHT']
WIDTH = cfg['CAPTCHA']['IMG_WIDTH']
CLASS_NUM = cfg['CAPTCHA']['CLASS_NUM']
CHAR_LEN = cfg['CAPTCHA']['CHAR_LEN']


class captcha_dataset(data.Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        data_path_root = './dataset/'
        data_path = data_path_root + data_path
        # if data_type == 'train':
        #     self.data_path = cfg['TRAINING_DIR']
        # elif data_type == 'val':
        #     self.data_path = cfg['TESTING_DIR']
        # elif data_type == 'test':
        #     self.data_path = cfg['TESTING_DIR']
        # else:
        #     raise ValueError('data_type must be train, val or test')
        
        self.data_path = data_path
        self.data_list = os.listdir(self.data_path)
        self.transform = transforms.Compose([
                # center crop 90% 
                transforms.CenterCrop(0.99),
                # random rotation 10 degree
                transforms.RandomRotation(2),
                # random affine 
                transforms.RandomAffine(0, scale=(0.97, 1.02), shear=5),
                
                transforms.Resize((HEIGHT, WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])

    def __getitem__(self, index):
        # Thử lấy ảnh với index hiện tại, nếu lỗi thì thử lấy ảnh tiếp theo
        max_retries = 10  # Số lần thử tối đa
        for attempt in range(max_retries):
            try:
                idx = (index + attempt) % len(self.data_list)  # Tránh vượt quá giới hạn
                img_path = os.path.join(self.data_path, self.data_list[idx])
                img = self.transform(Image.open(img_path))
                # label = self.data_list[idx].split('.')[0]
                label = self.data_list[idx].split('.')[0].lower() 
                
                return img, str_to_vec(label)
            except (PIL.UnidentifiedImageError, OSError, IOError) as e:
                print(f"Lỗi khi load {img_path}, thử lại: {e}")
                continue
        
        # Nếu tất cả các lần thử đều thất bại, trả về một mẫu mặc định hoặc raise lỗi
        # Hoặc có thể tạo một ảnh trắng với kích thước đúng
        blank_img = torch.zeros(3, HEIGHT, WIDTH)
        blank_label = str_to_vec("0")  # hoặc một nhãn mặc định phù hợp
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
    
    if len(lst) < CHAR_LEN:
        lst.extend([100] * (CHAR_LEN - len(lst)))
    return lst


def lst_to_str(lst: list):
    s = ''
    for i in lst:
        if i == 100:  # Padding token
            continue
        elif i < 10:
            s += chr(i + ord('0'))
        elif i < 36:
            s += chr(i + ord('a') - 10)
        elif i < 62:
            s += chr(i + ord('A') - 36)
    if len(s) < CHAR_LEN:
        s = s + ' ' * (CHAR_LEN - len(s))
    
    return s



# def str_to_onehotvec(s: str):
#     return F.one_hot(torch.LongTensor(str_to_lst(s)), CLASS_NUM)



def str_to_vec(s: str):
    return torch.LongTensor(str_to_lst(s))



if __name__ == '__main__':
    train_dataset_1  = captcha_dataset('train')
    print("len of train_dataset_1: ", len(train_dataset_1))
    train_dataset_2  = captcha_dataset('../dataset_real/train')
    print("len of train_dataset_2: ", len(train_dataset_2))
    train_dataset_3 = torch.utils.data.ConcatDataset([train_dataset_2] * 250)
    print("len of train_dataset_3: ", len(train_dataset_3))
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_3])
    print("len of train_dataset: ", len(train_dataset))

    dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        timeout=60,
    )

    
