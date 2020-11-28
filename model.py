import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch
import cv2
# from skimage import io
import matplotlib.pyplot as plt

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, padding=0)

    # def forward(self, x):
    #
    #     # [4,1,32,32]
    #     it = x.cuda().float()
    #     x0 = (x.cpu().detach().data.numpy()).astype("uint8")  # x转numpy
    #     # print(x0)
    #     # print("x0:", x.shape)  #x0: torch.Size([4, 1, 32, 32])
    #
    #     x1 = np.zeros([x0.shape[0], x0.shape[2], x0.shape[3]], dtype='uint8')
    #
    #     # print("x1:",x1.shape)  #x1: (4, 32, 32)
    #
    #     for i in range(x0.shape[0]):
    #         img = x0[i, :, :, :]
    #         # print("img:",img.shape)  #img: (1, 32, 32)
    #         img = np.transpose(img, (1, 2, 0))  # [h, w, 1]  transpose 作用是改变序列
    #         # print(img.shape) #(32, 32, 1)
    #         x1[i, :, :] = img[:, :, 0]
    #
    #     # print("img:", img.shape) #img: (32, 32, 1)
    #     # print("x1:", x1.shape) #x1: (4, 32, 32)
    #     img = x1  #img: (4, 32, 32)
    #
    #     # 转十进制格雷码
    #     img_new = (img.astype("uint8") / 2).astype("uint8")
    #     img = img.astype("uint8")
    #     img_new = img ^ img_new
    #     # print("img_new:", img_new.shape)  #img_new: (4, 32, 32) img_new: (2, 32, 32)
    #
    #     [h, w] = img_new.shape[1], img_new.shape[2]
    #
    #     image = np.empty((x0.shape[0], 8, h, w), dtype=np.uint8)  # 存余数 image[4,8,32,32]
    #
    #     for i in range(8):
    #         image[:, i, :, :] = img_new % 2  # 转格雷码8维图像
    #         # print(image[:, :, i])
    #         img_new = img_new // 2
    #
    #     x = torch.from_numpy(np.ascontiguousarray(image)).float() #torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
    #     x = x.cuda()    # [4,8,32,32]
    #
    #     # inte = x
    #     # x /= 255
    #     out = F.relu(self.conv1(x))     # [4,64,32,32]
    #
    #     # xshow = x.cpu().detach().data.numpy()[0, 0, :, :]
    #     # out = out.cpu().detach().data.numpy()[0, 0, :, :]
    #     # addimg = (xshow + out)/2
    #     #
    #     # plt.figure()
    #     # plt.subplot(1, 3, 1)
    #     # plt.imshow(xshow)
    #     # plt.subplot(1, 3, 2)
    #     # plt.imshow(out)
    #     # plt.subplot(1, 3, 3)
    #     # plt.imshow(addimg)
    #     #
    #     # plt.show()
    #     # exit(-1)
    #
    #
    #     out = F.relu(self.conv2(out))   # [4,32,32,32]
    #     out = self.conv3(out)   # [4,8,32,32]
    #     out = out.cuda()
    #     # print(it.dtype)
    #     # print(out.dtype)
    #     out += it
    #     # out = F.sigmoid(out)
    #     # print(torch.mean(out))
    #     return out

    def forward(self, x, not_8):
        it = not_8.cuda().float()
        # print(x.shape)

        # inte = x
        # x /= 255

        out = F.relu(self.conv1(x))     # [4,64,32,32]

        '''xshow = x.cpu().detach().data.numpy()[0, 0, :, :]
        out = out.cpu().detach().data.numpy()[0, 0, :, :]
        print(xshow)
        print(out)
        exit(-1)
        addimg = (xshow + out)/2

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(xshow)
        plt.subplot(1, 3, 2)
        plt.imshow(out)
        plt.subplot(1, 3, 3)
        plt.imshow(addimg)

        plt.show()
        exit(-1)'''


        out = F.relu(self.conv2(out))   # [4,32,32,32]
        print(out)
        # out = F.relu(self.conv3(out))  # [4,8,32,32]
        out = self.conv3(out)
        print(out)

        # xshow = x.cpu().detach().data.numpy()[0, 0, :, :]
        # out = out.cpu().detach().data.numpy()[0, 0, :, :]
        # print(xshow)
        # print(out)
        # exit(-1)

        out = out.cuda()

        # out += it
        # out = F.interpolate(out, scale_factor=2, mode='nearest')

        # it = F.upsample_bilinear(it, scale_factor=2)
        # out = F.upsample_bilinear(out, scale_factor=2)

        # out = self.conv4(out)
        # out = F.relu(out)

        # out += it
        # out = self.conv4(out)
        # out = F.sigmoid(out)
        # print(torch.mean(out))
        return out

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                # m.weight.data = 0
