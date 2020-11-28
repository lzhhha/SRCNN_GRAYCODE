import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import io

parser = argparse.ArgumentParser(description='SRCNN run parameters')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--zoom_factor', type=int, required=True)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

img = Image.open(args.image).convert('YCbCr')
img = img.resize((int(img.size[0]*args.zoom_factor), int(img.size[1]*args.zoom_factor)), Image.BICUBIC)  # first, we upscale the image via bicubic interpolation
y, cb, cr = img.split()
img_to_tensor = transforms.ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])  # we only work with the "Y" channel

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
print(device)
model = torch.load(args.model).to(device)
input = input.to(device)

out = model(input)

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
    # img_1 = x_1  # [1, 512, 512]     (y_decode.png)

    # # 保存图像
    # img_1 = np.transpose(img_1, (1, 2, 0))
    # print(np.max(img_1))
    # print("1111111")
    # print(img_1.shape)
    # plt.imshow(img_1)
    # plt.show()
    # io.imsave("D:/CV例程/SRCNN-master/y_decode.png", img_1)
    # img_cb = np.transpose(img_cb, (1, 2, 0))
    # img_cr = np.transpose(img_cr, (1, 2, 0))
    # io.imsave("/export/liuzhe/program2/RCAN_test/RCAN_TestCode/SR/BI/cb.png", img_cb)
    # io.imsave("/export/liuzhe/program2/RCAN_test/RCAN_TestCode/SR/BI/cr.png", img_cr)

    # img_2 = img_1

    '''# 十进制格雷码转二进制的十进制数
    img_2 = (img_1.astype("uint8") / 2).astype("uint8")  # 除2取下界等于右移位运算
    img = img_1.astype("uint8")
    img_2 = img ^ img_2  # 异或运算   [1, 512, 512]'''

    # x_finally[:, 0, :, :] = img_2

    # ycrcb转rgb
    # a = x_finally[0, :, :, :]  # [3, h, w]
    # a = np.transpose(a, (1, 2, 0))  # 转成  [512, 512, 3]

    # y, cb, cr = cv2.split(img)
    # img = cv2.merge([y, cr, cb])
    # img = cv2.cvtColor(a, cv2.COLOR_YCrCb2RGB)  # [512,512,3]
    # b,g,r = cv2.split(img)
    # img = cv2.merge([r,g,b])
    # img = np.transpose(img, (2, 0, 1))  # 转成[3,512,512]
    # for i in range(x_0.shape[0]):
    #     x_finally[i, :, :, :] = img
    #
    x_finally = torch.from_numpy(np.ascontiguousarray(x_finally)).float()  # [1,1,512,512]
    x_finally /= 255
    x_finally = x_finally.cuda()
    # print(x_finally.shape)

    return x_finally


out = channel_8_1(out)
#srimg = np.transpose(out.cpu().detach().data.numpy()[0], (1, 2, 0))
out = out.cpu()
out_img_y = out[0].detach().numpy()
# print(out_img_y)
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)

out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')


# plt.imshow(out_img_y)
# plt.show()

out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')  # we merge the output of our network with the upscaled Cb and Cr from before
                                                                    # before converting the result in RGB
out_img.save(f"zoomed_{args.image}")
# plt.imshow(out_img)
# plt.show()
