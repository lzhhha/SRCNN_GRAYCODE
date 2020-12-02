import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import io
import torch.nn.functional as F
import skimage.measure

parser = argparse.ArgumentParser(description='SRCNN run parameters')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, default='/export/liuzhe/program2/SRCNN/Set5/x2/',
                    help='hr_dataset directory', required=True)
parser.add_argument('--zoom_factor', type=int, required=True)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

img = Image.open(args.image).convert('YCbCr')
# img = img.resize((int(img.size[0]*args.zoom_factor), int(img.size[1]*args.zoom_factor)), Image.BICUBIC)  # first, we upscale the image via bicubic interpolation
y, cb, cr = img.split()

input = np.asarray(y).astype("uint8")   # [702,1020]

# 保存原lr的y通道图
# out_img_y = Image.fromarray(np.uint8(input), mode='L')
# out_img_y.save(f"input_y_{args.image}")
# exit(-1)

input = np.reshape(input, [1, input.shape[0], input.shape[1]])
input = torch.from_numpy(np.ascontiguousarray(input)).float()
input_1 = input.clone()

def hr_convert(hr):
    # [1,32,32]
    img = hr
    img = (np.array(img)[0]).astype("uint8")

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

input = hr_convert(input)   # [8,256,256]

input = input.view(1, 8, input.shape[1], input.shape[2])
input_1 = input.view(1, -1, input_1.shape[1], input_1.shape[2])
# input = np.reshape()

# img_to_tensor = transforms.ToTensor()
# input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])  # we only work with the "Y" channel

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
print(device)
model = torch.load(args.model).to(device)
input = input.to(device)

out = model(input, input_1)

def str_reverse1(s):
    return s[::-1]
def channel_8_1(sr):
    # [1, 8, 512, 512]
    x_0 = (sr.cpu().detach().data.numpy())  # x转numpy

    x_bin = np.zeros([x_0.shape[1], x_0.shape[2], x_0.shape[3]], dtype='uint8')  # [8, h, w]

    x_1 = np.zeros([x_0.shape[0], x_0.shape[2], x_0.shape[3]], dtype='uint8')  # [1, 512 ,512]   [1,h,w]

    # x_2 = np.zeros([3, x_0.shape[2], x_0.shape[3]], dtype = np.uint8)   # [3, h ,w]

    x_finally = np.zeros([x_0.shape[0], 1, x_0.shape[2], x_0.shape[3]], dtype='uint8')  # [1,1,512,512]

    h, w = x_0.shape[2], x_0.shape[3]
    for k in range(x_0.shape[0]):
        img1 = x_0[k, :, :, :]  # [8, 344 ,228]
        # 十进制格雷码转二进制的十进制数
        for i in range(h):
            for j in range(w):
                lst = []
                for s in range(8):
                    lst.append(str(img1[s, i, j]))  #append() 方法用于在列表末尾添加新的对象。
                n = ''.join(lst)  # str从低位到高位

                # n = str(np.binary_repr(img1[:, i, j], 8))   # str从低位到高位
                n = str_reverse1(n)  # str从高位到低位
                result = ''
                for q in range(8):  # 格雷码转二进制码
                    if q != 0:
                        temp = 1
                        if result[q - 1] == n[q]:
                            temp = 0
                        result += str(temp)
                    else:
                        result += str(n[0])

                result = str_reverse1(result)  # 从低位到高位
                for m in range(8):
                    x_bin[m, i, j] = result[m]  # [8, h ,w] 解码后的二进制码

        x_temp = x_bin[0, :, :]*1 + x_bin[1, :, :]*2 + x_bin[2, :, :]*4 + x_bin[3, :, :]*8 + x_bin[4, :, :]*16 + x_bin[5, :, :]*32 + x_bin[6, :, :]*64 + x_bin[7, :, :]*128
        # x_1[x_0.shape[0]-1, :, :] = img1[0, :, :]*128 + img1[0, :, :]*64 + img1[0, :, :]*32 + img1[0, :, :]*16 + img1[0, :, :]*8 + img1[0, :, :]*4 + img1[0, :, :]*2
        x_finally[k, :, :, :] = x_temp

    x_finally = torch.from_numpy(np.ascontiguousarray(x_finally)).float()  # [1,1,512,512]

    x_finally = x_finally.cuda()
    # print(x_finally.shape)

    return x_finally

out = channel_8_1(out)  # [1,1,512,512]

# print('11111')
# print(out.shape)
# exit(-1)

#srimg = np.transpose(out.cpu().detach().data.numpy()[0], (1, 2, 0))
out = out.cpu()
out_img_y = out[0].detach().numpy()
# print(out_img_y)
# out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)

# 输出lr处理后的y通道图像
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
out_img_y.save(f"zoomed_{args.image}")


cb = np.asarray(cb).astype("uint8")
cr = np.asarray(cr).astype("uint8")
cb = np.reshape(cb, [1, 1, cb.shape[0], cb.shape[1]])
cr = np.reshape(cr, [1, 1, cr.shape[0], cr.shape[1]])
cb = torch.from_numpy(np.ascontiguousarray(cb)).float()
cr = torch.from_numpy(np.ascontiguousarray(cr)).float()

cb = F.upsample_bilinear(cb, scale_factor=2)
cr = F.upsample_bilinear(cr, scale_factor=2)
cb = cb.cpu().detach().data.numpy()[0, 0, :, :]
cr = cr.cpu().detach().data.numpy()[0, 0, :, :]
cb = Image.fromarray(np.uint8(cb))
cr = Image.fromarray(np.uint8(cr))

# c = np.asarray(y).astype("uint8")
# c = np.reshape(c, [1, 1, c.shape[0], c.shape[1]])
# c = torch.from_numpy(np.ascontiguousarray(c)).float()
# c = F.upsample_bilinear(c, scale_factor=2)
# c = c.cpu().detach().data.numpy()[0, 0, :, :]
# y = Image.fromarray(np.uint8(c))

out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')

img_pre = Image.open('/export/liuzhe/program2/SRCNN/0010.png').convert('RGB')

out_img = np.asarray(out_img)
img_pre = np.asarray(img_pre)
# print(out_img.shape)

psnr = skimage.measure.compare_psnr(out_img, img_pre, data_range=255)
print(psnr)

out_img.save(f"zoomed_{args.image}")

