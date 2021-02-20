# ## written by Satyam Gaba <sgaba@ucsd.edu>

import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import models.model as model
import torch.nn as nn
import os
import numpy as np
import utils
import scipy.io as io
from tqdm import tqdm
from PIL import Image
import imageio
# import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# The location of testing set
parser.add_argument('--imageRoot', default='rebar_data/testing/', help='path to input images folder' )
parser.add_argument('--image','-i', help='input images' )
parser.add_argument('--modelRoot', default='experiments/checkpoint/', help='the path to load the model')
parser.add_argument('--epochId','-e', type=int, default=1200, help='the epoch number of model to load')
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training' )
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network' )

# The detail network setting
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and opt.noCuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

encoder = model.encoder()
decoder = model.decoder()

checkpoint = torch.load('%s/model_%d.pth' % (opt.modelRoot, opt.epochId) )
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder = encoder.eval()
decoder = decoder.eval()

# Move network and containers to gpu
if not opt.noCuda:
    device = 'cuda'
else:
    device = 'cpu'

encoder = encoder.to(device)
decoder = decoder.to(device)

# mean = np.array([153.76539744, 146.25521783, 140.10715883])
# std = np.array([32.70991896, 32.16577284, 31.63130951])
image_file = os.path.join(opt.imageRoot,opt.image)
image = Image.open(image_file)
image = np.asarray(image)
# print(image[1,1,:])

# image = (image - mean)/std

image = np.transpose(image, [2,0,1])
# print("********",image.shape)

image = image[np.newaxis,...]
image = torch.from_numpy(image)

image = Variable(image).float().to(device)

# Test network
x1, x2, x3, x4, x5 = encoder(image )
pred = decoder(image, x1, x2, x3, x4, x5 )

pred = pred.cpu().detach().numpy()
pred = pred[0,0]
print("pred shape: ",pred.shape)

plt.figure(figsize=(32,24))
plt.imshow(pred)
plt.axis('off')
plt.savefig("results/"+opt.image.split('.')[0]+"_hm.png")




# def normalize(img):
#     ma = np.max(img)
#     mi = np.min(img)
#     img = ((255-0)/(ma-mi))*img - mi
#     return img.astype(np.uint8)

# print(np.max(pred), np.min(pred))
# # top1 = pred[0,...]
# # # top2 = pred[1,...]

# # cv2.normalize(pred,  pred, 0, 255, cv2.NORM_MINMAX)
# # # cv2.normalize(top2,  top2, 0, 255, cv2.NORM_MINMAX)

# pred= normalize(pred)
# print(np.max(pred), np.min(pred))


# imageio.imwrite("pred.png",pred)
# # cv2.imwrite("pred.png",pred)
# # # cv2.imwrite("top2.png",top2)


# import imageio
# import numpy as np
# from scipy.ndimage.filters import gaussian_filter
# from scipy.signal import convolve
# from matplotlib import pyplot as plt



# hmap = imageio.imread('heatmap_sample.png', as_gray = True)/255
# # print(hmap)
# # print(hmap[...,3])
# # hmap =hmap[...,:3]

# imWd,imHt = 540,540
# sigma = np.minimum(imWd,imHt)/186
# kernel_size = [15,15]
# kernel = np.zeros(kernel_size)
# kernel[kernel_size[0]//2, kernel_size[1]//2] = 1
# kernel = gaussian_filter(kernel, sigma=sigma, mode='constant', cval=0)
# # kernel = normalize(kernel)
# # print(np.max(hmap_filter), np.min(hmap_filter))
# kernel_mean = np.mean(kernel)
# kernel_ = kernel - kernel_mean

# out = convolve(hmap-kernel_mean, kernel_, mode ='same')

# # print(out)
# # print(np.where(out>250))
# # print(kernel_)
# # x,y = np.where(out>0.5)

# y,x = (np.unravel_index(out.argmax(), out.shape))

# # print(y,x)
# # print(len(x), len(y))

# plt.figure(figsize=(10,10))
# plt.subplot(2,2,1)
# plt.imshow(hmap, cmap='gray')
# plt.title("Predictions from Network")
# plt.axis('off')

# plt.subplot(2,2,2)
# plt.imshow(kernel, cmap='gray')
# plt.title("Heatmap Kernel")
# plt.axis('off')

# plt.subplot(2,2,3)
# plt.imshow(out, cmap='gray')
# plt.title("Convolution Output")
# plt.axis('off')

# plt.subplot(2,2,4)
# plt.imshow(out, cmap='gray')
# plt.plot(x,y,'rx',ms=10, mew=3)
# plt.title("Rebar Intersection Prediction")
# plt.axis('off')


# plt.show()
