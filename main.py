import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import DatasetFromFolder
from model import SRCNN

import numpy as np
import cv2

parser = argparse.ArgumentParser(description='SRCNN training parameters')
parser.add_argument('--zoom_factor', type=str, required=True)
parser.add_argument('--nb_epochs', type=int, default=200)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--hr_path', type=str, default='/export/liuzhe/data/dataset/DIV2K3/DIV2K_train_HR',
                    help='hr_dataset directory')
parser.add_argument('--lr_path', type=str, default='/export/liuzhe/data/dataset/DIV2K3/DIV2K_train_LR_bicubic',
                    help='lr_dataset directory')
parser.add_argument('--hrval_path', type=str, default='/export/liuzhe/data/dataset/DIV2K3/DIV2K_val_HR',
                    help='lr_dataset directory')
parser.add_argument('--lrval_path', type=str, default='/export/liuzhe/data/dataset/DIV2K3/DIV2K_val_LR_bicubic',
                    help='lr_dataset directory')

args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Parameters
BATCH_SIZE = 4
NUM_WORKERS = 0 # on Windows, set this variable to 0

trainset = DatasetFromFolder(split='train', hr_path=args.hr_path, lr_path=args.lr_path, zoom_factor=args.zoom_factor)
testset = DatasetFromFolder(split='test', hr_path=args.hrval_path, lr_path=args.lrval_path, zoom_factor=args.zoom_factor)

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)
# testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

model = SRCNN().to(device)
criterion = nn.L1Loss()
mse = nn.MSELoss()
optimizer = optim.Adam(  # we use Adam instead of SGD like in the paper, because it's faster
    [
        {"params": model.conv1.parameters(), "lr": 0.01},
        {"params": model.conv2.parameters(), "lr": 0.01},
        {"params": model.conv3.parameters(), "lr": 0.001},
    ], lr=0.001,
)

def str_reverse1(s):
    return s[::-1]

def channel_8_1(sr):
    # [4, 8, 32, 32]
    x_0 = (sr.cpu().detach().data.numpy())  # x转numpy
    x_0 = x_0.astype('uint8')
    x_bin = np.zeros([x_0.shape[1], x_0.shape[2], x_0.shape[3]], dtype='uint8')  # [8, h, w]
    x_1 = np.zeros([x_0.shape[0], x_0.shape[2], x_0.shape[3]], dtype='uint8')  # [1, 512 ,512]   [1,h,w]
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

        # x_1[x_0.shape[0]-1, :, :] = x_bin[0, :, :]*1 + x_bin[1, :, :]*2 + x_bin[2, :, :]*4 + x_bin[3, :, :]*8 + x_bin[4, :, :]*16 + x_bin[5, :, :]*32 + x_bin[6, :, :]*64 + x_bin[7, :, :]*128
        # x_1[x_0.shape[0]-1, :, :] = img1[0, :, :]*128 + img1[0, :, :]*64 + img1[0, :, :]*32 + img1[0, :, :]*16 + img1[0, :, :]*8 + img1[0, :, :]*4 + img1[0, :, :]*2
        x_temp = x_bin[0, :, :]*1 + x_bin[1, :, :]*2 + x_bin[2, :, :]*4 + x_bin[3, :, :]*8 + x_bin[4, :, :]*16 + x_bin[5, :, :]*32 + x_bin[6, :, :]*64 + x_bin[7, :, :]*128
        # x_temp = x_bin[0, :, :] * 128 + x_bin[1, :, :] * 64 + x_bin[2, :, :] * 32 + x_bin[3, :, :] * 16 + x_bin[4, :, :] * 8 + x_bin[5, :, :] * 4 + x_bin[6, :, :] * 2 + x_bin[7, :, :] * 1
        x_finally[k, :, :, :] = x_temp

    # img_1 = x_1  # [1, 512, 512]     (y_decode.png)

    # img_2 = img_1

    '''# 十进制格雷码转二进制的十进制数
    img_2 = (img_1.astype("uint8") / 2).astype("uint8")  # 除2取下界等于右移位运算
    img = img_1.astype("uint8")
    img_2 = img ^ img_2  # 异或运算   [1, 512, 512]'''

    # x_finally[:, 0, :, :] = img_2

    x_finally = torch.from_numpy(np.ascontiguousarray(x_finally)).float()  # [4,1,32,32]
    # x_finally /= 255
    x_finally = x_finally.cuda()
    # print(x_finally)

    return x_finally


for epoch in range(args.nb_epochs):
    # Train
    epoch_loss = 0
    for iteration, batch in enumerate(trainloader):
        input, target, target_hr, input_1 = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        optimizer.zero_grad()

        # 做差验证格雷码
        # target_hr = target_hr.cpu().detach().data.numpy()   # [4,1,32,32]
        # target81 = channel_8_1(target)  # target [4,8,32,32] , target81[4,1,32,32]
        # target81 = target81.cpu().detach().data.numpy()
        # # print(target81)
        # # print('===================================')
        # # print(target_hr)
        # print(np.mean(target_hr-target81))
        # exit(-1)

        # model.initialize()
        # print(model.conv1.weight)

        # input_3 = np.transpose(input_3.cpu().detach().data.numpy()[0], [1, 2, 0])
        # target = np.transpose(target.cpu().detach().data.numpy()[0], [1, 2, 0])
        # print(input.shape)
        #
        # cv2.imwrite('/export/liuzhe/program2/XNOR/MNIST/lr.png', input_3 * 255)
        # cv2.imwrite('/export/liuzhe/program2/XNOR/MNIST/hr.png', target * 255)
        # exit(-1)

        out = model(input, input_1)      # out = [4,8,32,32], target = [4,8,32,32]

        # 保存8通道hr、lr图像
        # hr_8 = target.cpu().detach().data.numpy()
        # sr_8 = out.cpu().detach().data.numpy()
        # hr_8 = np.transpose(hr_8[0], [1, 2, 0])     # [96,96,8]
        # sr_8 = np.transpose(sr_8[0], [1, 2, 0])     # [96,96,8]

        # cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/hr.png', hr_8)
        # cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/sr.png', sr_8 * 255 * 255)

        # for i in range(8):
        #     cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/hr_{}.png'.format(i), hr_8[:, :, i] * 255)
        #     cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/sr_{}.png'.format(i), sr_8[:, :, i] * 255 * 255)
        # exit(-1)

        # out = Variable(out, requires_grad=True)
        # target = Variable(target,requires_grad=False)

        # out = out.long()
        # target = target.long()
        # print(out.dtype)
        # print('==========================\n')

        loss = criterion(out, target)

        # out = channel_8_1(out)
        # target = channel_8_1(target)    # out = [4,1,32,32], target = [4,1,32,32]

        # print("11111")
        # print(out.shape)
        # print(target.shape)

        loss = loss.requires_grad_()

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch}. Training loss: {epoch_loss / len(trainloader)}")

    # Test
    avg_psnr = 0
    num = 0
    with torch.no_grad():
        for batch in testloader:
            num = num+1
            input, target, not_8 = batch[0].to(device), batch[1].to(device), batch[3].to(device)

            # model = torch.load('model_70.pth').to(device)
            out = model(input, not_8)

            # hr_8 = target.cpu().detach().data.numpy()
            # sr_8 = out.cpu().detach().data.numpy()
            # hr_8 = np.transpose(hr_8[0], [1, 2, 0])  # [96,96,8]
            # sr_8 = np.transpose(sr_8[0], [1, 2, 0])  # [96,96,8]

            # import skimage.measure
            # ssim = skimage.measure.compare_ssim(sr_8, hr_8 , data_range=1, multichannel=True)
            # psnr = skimage.measure.compare_psnr(sr_8, hr_8 , data_range=1)
            # print(psnr)
            # exit(-1)

            # cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/hr.png', hr_8 * 255)
            # cv2.imwrite('/export/liuzhe/program2/SRCNN/hr_8_sr_8/sr.png', sr_8 * 255)
            # exit(-1)

            # loss = mse(out, target)

            out = channel_8_1(out)
            target = channel_8_1(target)  # out = [4,1,32,32], target = [4,1,32,32]

            out = out.cpu().detach().data.numpy()[0, 0, :, :]
            target = target.cpu().detach().data.numpy()[0, 0, :, :]
            import skimage.measure
            psnr = skimage.measure.compare_psnr(out, target, 255)
            # print(psnr)

            # import imageio
            # imageio.imwrite('/export/liuzhe/program2/SRCNN/data/'
            #                 +str(num)+'out.png',out)
            # imageio.imwrite('/export/liuzhe/program2/SRCNN/data/'
            #                 + str(num) + 'target.png', target)

            # psnr = 10 * log10(1 / loss.item())
            # print(psnr)
            avg_psnr += psnr
    print(f"Average PSNR: {avg_psnr / len(testloader)} dB.")

    # Save model
    torch.save(model, f"model_{epoch}.pth")
