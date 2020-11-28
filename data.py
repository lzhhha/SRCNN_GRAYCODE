from os import listdir
from os.path import join

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

import numpy as np
import torch
import cv2
import os

import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(lr_path, hr_path):
    lr = Image.open(lr_path).convert('YCbCr')
    hr = Image.open(hr_path).convert("YCbCr")
    y_lr, _, _ = lr.split()
    y_hr, _, _ = hr.split()
    return y_lr, y_hr

def random_crop(images, split):

    h, w = images[0].shape[:2]

    crops = []

    if split == 'test':
        new_h = 256
        new_w = 256

        # for image in images:
        #     imaget = image[0:0 + new_h, 0:0 + new_w]
        #     crops.append(imaget)
        lr = images[0][0:0 + (new_h // 2), 0:0 + (new_w // 2)]
        hr = images[1][0:0 + (new_h), 0:0 + (new_w)]

        crops.append(lr)
        crops.append(hr)
    else:
        new_h = 196
        new_w = 196
        y = np.random.randint(0, h-new_h)   # 随机整型数[0,h-new_h)
        x = np.random.randint(0, w-new_w)

        lr = images[0][y//2:y//2+(new_h//2), x//2:x//2+(new_w//2)]
        hr = images[1][y:y + (new_h), x:x + (new_w)]

        crops.append(lr)
        crops.append(hr)

        # for image in images:
        #     imaget = image[y:y+new_h, x:x+new_w]
        #     crops.append(imaget)

    return crops

class DatasetFromFolder(Dataset):
    def __init__(self, split, hr_path, lr_path, zoom_factor):
        super(DatasetFromFolder, self).__init__()

        self.hr_path = hr_path
        self.split = split
        self.lr_path = lr_path
        # self.hr_filenames = [(os.listdir(hr_path)).sort(key=lambda x:int(x[:-4]))]
        self.hr_filenames = os.listdir(hr_path)
        # self.hr_filenames = [join(hr_path, x) for x in listdir(hr_path) if is_image_file(x)]
        self.zoom_factor = zoom_factor

        self.input_transform = transforms.Compose(transforms.Resize(196, interpolation=Image.BICUBIC))

    def __getitem__(self, index):
        # print(self.hr_filenames)
        self.hr_filenames.sort(key=lambda x:int(x[:-4]))
        hr_filename = self.hr_path + '/' + self.hr_filenames[index]
        lr_filename = self.lr_path + '/' + "X" + self.zoom_factor + '/' + self.hr_filenames[index].split('.')[0] + "x" + self.zoom_factor + ".png"
        # hr_filename = self.hr_filenames[index]
        # lr_filename = str(os.path.dirname(self.hr_filenames[index]) +'/' + str(os.path.basename(self.hr_filenames[index]).split('.'))[0] + "x" + self.zoom_factor + ".png")
        input, target = load_img(lr_filename, hr_filename)
        # input = self.input_transform(input)

        # lrnumpy = np.transpose(input_3.cpu().detach().data.numpy()[0], [1, 2, 0])
        # hrnumpy = np.transpose(target.cpu().detach().data.numpy()[0], [1, 2, 0])
        # import skimage.measure
        #
        # print(skimage.measure.compare_psnr(lrnumpy, hrnumpy, data_range=1))

        input = np.asarray(input).astype("uint8")/255   # [702,1020]
        target = np.asarray(target).astype("uint8")/255  # [1404,2040]

        # exit(-1)
        input, target = random_crop([input, target], self.split)
        # print(target.shape)
        # print(input.shape)

        input = np.reshape(input, [1, input.shape[0], input.shape[1]])
        target = np.reshape(target, [1, target.shape[0], target.shape[1]])

        # a1 = np.zeros([1, input.shape[0], input.shape[1]])
        # a2 = np.zeros([1, target.shape[0], target.shape[1]])
        # a1[0] += input
        # a2[0] += target

        # input = np.transpose(input, (1, 2, 0))
        # target = np.transpose(target, (1, 2, 0))

        input = torch.from_numpy(np.ascontiguousarray(input)).float()
        input_1 = input.clone()

        # input_1 = torch.unsqueeze(input, dim=0)     # [1,1,h,w]
        # print(input_1.shape)
        # input_1 = F.upsample_bilinear(input, scale_factor=2)
        # input_1 = input_1[0]
        # print(input_1.shape)

        input = self.hr_convert(input)

        target = torch.from_numpy(np.ascontiguousarray(target)).float()

        target_hr = target.clone()
        # target = self.hr_convert(target)  # [8, 32, 32]的tensor

        # 保存8通道hr、lr图像
        # hr_8 = (target.cpu().detach().data.numpy()).astype('uint8')
        # sr_8 = (input.cpu().detach().data.numpy()).astype('uint8')
        # hr_8 = np.transpose(hr_8, [1, 2, 0])  # [96,96,8]
        # sr_8 = np.transpose(sr_8, [1, 2, 0])  # [96,96,8]

        # cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/hr.png', hr_8 * 255)
        # cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/sr.png', sr_8 * 255)
        # exit(-1)
        # for i in range(8):
        #     cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/hr_{}.png'.format(i), hr_8[:, :, i] * 255)
        #     cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/sr_{}.png'.format(i), sr_8[:, :, i] * 255)
        # exit(-1)

        # print(np.max(input))

        return input, target, target_hr, input_1    # tensor

    def __len__(self):
        return len(self.hr_filenames)

    def hr_convert(self, hr):
        # [1,32,32]
        img = hr
        img = (np.array(img)[0] * 255).astype("uint8")

        # img_new = img

        img_new = (img.astype("uint8") / 2).astype("uint8")  # 除2取下界等于右移位运算
        img = img.astype("uint8")
        img_new = img ^ img_new  # 异或运算

        [h, w] = img_new.shape[0], img_new.shape[1]

        image = np.empty((8, h, w), dtype=np.uint8)  # 存 余数

        for i in range(8):
            image[i, :, :] = img_new % 2  # 转格雷码8维图像
            img_new = img_new // 2

        x_finally = torch.from_numpy(np.ascontiguousarray(image)).float()  # [8,32,32]
        # image = torch.from_numpy(np.ascontiguousarray(image)).float()
        # x_finally = image.cuda()
        return x_finally
