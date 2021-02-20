## written by Satyam Gaba <sgaba@ucsd.edu>

import torch
import numpy as np
import os.path as osp
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json
from PIL import Image
# import torchvision.transforms.functional as F


class BatchLoader(Dataset ):
    def __init__(self, imageRoot, heatmapRoot, labelFile, transforms = None, imageRGB = None):
        super(BatchLoader, self).__init__()

        self.imageRoot = imageRoot
        self.heatmapRoot = heatmapRoot
        self.labelFile = labelFile
        self.transforms = transforms

        with open(self.labelFile) as json_file:
            self.labels = json.load(json_file)

        self.labelNames = [i for i in sorted(self.labels.keys())]
        self.imgNames = [osp.join(imageRoot,i) for i in self.labelNames]
        self.heatmapNames = [osp.join(heatmapRoot, i.split('.')[0]+"_hm.png") for i in self.labelNames]

        self.count = len(self.imgNames )
        self.perm = list(range(self.count ) )
        random.shuffle(self.perm )
        print('Image Num: %d' % self.count )

        # MEAN and std of image
        self.imageRGB = imageRGB
        if imageRGB:
            self.imMean = np.array(imageRGB['mean'], dtype=np.float32 )
            self.imStd = np.array(imageRGB['std'], dtype=np.float32 )

            self.imMean = self.imMean.reshape([1, 1, 3] )
            self.imStd = self.imStd.reshape([1, 1, 3] )

        self.iterCount = 0

    def __len__(self):
        return self.count

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, ind ):

        imName = self.imgNames[self.perm[ind] ]
        heatmapName = self.heatmapNames[self.perm[ind] ]

        im = self.loadImage(imName)
        heatmap = self.loadHeatmap(heatmapName)

        # augmentation # seed again and again to get same order
        if self.transforms:
            seed = random.randint(0, 2**32)
            self._set_seed(seed)
            im = self.transforms(im)
            self._set_seed(seed)
            heatmap = self.transforms(heatmap)
        else:
            im = transforms.ToTensor()(im)
            heatmap = transforms.ToTensor()(heatmap)



        # imgz = im.data.cpu().numpy()
        # imgz = imgz * 255
        # print("SHAPE: ",imgz.shape)
        # print(np.max(imgz), np.min(imgz), np.mean(imgz))
        # imgz = Image.fromarray(imgz.astype('uint8'),'RGB')
        # imgz.save("impil.png")
        # print("......saved....")

        ## Load data
        # im: input immage batch, Nx3ximHeightximWidth
        # heatmap: heatmap of valid region, Nx1ximHeightximWidth

        batchDict = {
                'im' : im,
                'heatmap':heatmap,
                'imName': imName}

        return batchDict


    def loadImage(self, imName ):
        # Load inpute image

        im = Image.open(imName )
        im = np.asarray(im )

        if len(im.shape) == 2:
            print('Warning: load a gray image')
            im = im[:, :, np.newaxis]
            im = np.concatenate([im, im, im], axis=2)

        if self.imageRGB:
            im = (im - self.imMean ) / self.imStd
        im = Image.fromarray(im.astype('uint8'), 'RGB')
        # im.save("im.png")
        return im

    def loadHeatmap(self, heatmapName ):
        # Load inpute image
        hm = Image.open(heatmapName )
        return hm
