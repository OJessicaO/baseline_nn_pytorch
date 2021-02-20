## written by Satyam Gaba <sgaba@ucsd.edu>

import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim
from torch.utils.data import DataLoader
import models.model as model
import torch.nn as nn
import os
import numpy as np
import utils
import scipy.io as io
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime


parser = argparse.ArgumentParser()
# The location of training set
parser.add_argument('--imageRoot', default='./rebar_data/images1/', help='path to input images' )
parser.add_argument('--heatmapRoot', default='./rebar_data/heatmaps1/', help='path to input heatmaps' )
parser.add_argument('--trainLabelFile', default='./rebar_data/labels1/train.json', help='path to train labels')
parser.add_argument('--valLabelFile', default='./rebar_data/labels1/val.json', help='path to val labels' )
parser.add_argument('--experiment', '-e', default='checkpoint1', help='the path to store sampled images and models')
parser.add_argument('--imHeight', '-imH', type=int, default=600, help='height of input image')
parser.add_argument('--imWidth', '-imW', type=int, default=600, help='width of input image')
parser.add_argument('--batchSize', type=int, default=3, help='the size of a batch')
parser.add_argument('--nepoch', '-n', type=int, default=300, help='the training epoch')
parser.add_argument('--initLR', type=float, default=0.01, help='the initial learning rate')
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training')
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network')
parser.add_argument('--untrainedResnet', action='store_false', help='whether to train resnet block from scratch or load pretrained weights' )
parser.add_argument('--resumeTraining', '-r', action='store_true', help='resume training from a trained model?')
parser.add_argument('--savedModelPath', type=str, help='path of saved model')



# The detail network setting
opt = parser.parse_args()
print(opt)

# directory to save all experiments
if not os.path.exists('experiments'):
    os.makedirs('experiments')


opt.experiment = os.path.join('experiments', opt.experiment)

# Save all the codes
os.system('mkdir %s' % opt.experiment )
os.system('cp *.py %s' % opt.experiment )
os.system('cp -r models/ %s' % opt.experiment)

if torch.cuda.is_available() and opt.noCuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
encoder = model.encoder()
decoder = model.decoder()

if opt.untrainedResnet:
    # load pretrained weights for resBlock
    model.loadPretrainedWeight(encoder)
    
# Move network and containers to gpu
if not opt.noCuda:
    device = 'cuda'
else:
    device = 'cpu'
    
encoder = encoder.to(device)
decoder = decoder.to(device)

# Initialize optimizer
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.SGD(params, lr=opt.initLR, momentum=0.9, weight_decay=5e-5)
Loss = nn.MSELoss()

start_epoch = 0

# load the weights
if opt.resumeTraining:
    checkpoint = torch.load(opt.savedModelPath)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']



#augment the dataset with transformations
transformations = { 'train':[
    transforms.RandomRotation((0,90)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomResizedCrop((opt.imWidth, opt.imHeight)),
    transforms.ToTensor(),
    ]
                    ,'val':[
    transforms.ToTensor(),
    ]
}

tfs_train = transforms.Compose(transformations['train'])
tfs_val = transforms.Compose(transformations['val'])

# train data parameters to normalize the image
# pass both mean and std or set imageRGB to None
imageRGB = {'mean':[153.76539744/255, 146.25521783/255, 140.10715883/255],
            'std':[32.70991896/255, 32.16577284/255, 31.63130951/255]}
# Initialize dataLoader
segTrainDataset = dataLoader.BatchLoader(
        imageRoot = opt.imageRoot,
        heatmapRoot = opt.heatmapRoot,
        labelFile = opt.trainLabelFile,
        transforms = tfs_train,
        # imageRGB = imageRGB,
        )
# segValDataset = dataLoader.BatchLoader(
#         imageRoot = opt.imageRoot,
#         heatmapRoot = opt.heatmapRoot,
#         labelFile = opt.valLabelFile,
#         transforms = tfs_val,
#         # imageRGB = imageRGB,
#         )
segTrainLoader = DataLoader(segTrainDataset, batch_size=opt.batchSize, num_workers=0, shuffle=True )
# segValLoader = DataLoader(segValDataset, batch_size=opt.batchSize, num_workers=0, shuffle=True )

now = datetime.now()
tb_dir = os.path.join(opt.experiment, "runs", now.strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(tb_dir,comment="check input images")

torch.cuda.empty_cache()

print("\n meow meow meow meow meow ... \n")

def train():
    """Training Code"""
    epoch = opt.nepoch

    for epoch in tqdm(range(start_epoch, opt.nepoch )):
        encoder.train()
        decoder.train()
        
        cumm_losses = []
        total_samples = 0
        
        for i, dataBatch in enumerate(segTrainLoader ):
            # Read data
            imBatch = Variable(dataBatch['im']).to(device)
            hmBatch = Variable(dataBatch['heatmap']).to(device)

            num_samples = imBatch.size()[0]
            total_samples += num_samples

            # Train network
            optimizer.zero_grad()

            x1, x2, x3, x4, x5 = encoder(imBatch )
            pred = decoder(imBatch, x1, x2, x3, x4, x5 )
            loss = Loss(pred, hmBatch)
            loss.backward()
            optimizer.step()

            cumm_losses.append(loss.cpu().data.item()*num_samples)

        epoch_loss = np.sum(cumm_losses)/total_samples
        # write to tensorboard
        writer.add_scalar('data/train_loss', epoch_loss, epoch)
        # x = vutils.make_grid(imBatch)
        writer.add_images('input', imBatch, epoch) 
        # t = vutils.make_grid(hmBatch)
        writer.add_images('target', hmBatch, epoch)  
        # y = vutils.make_grid(pred)
        writer.add_images('pred', pred, epoch)

        if (epoch+1) % 10 == 0:
            state = {
                'epoch':epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, '%s/model_%d.pth' % (opt.experiment, epoch+1))

            # torch.save(encoder.state_dict(), '%s/encoder_%d.pth' % (opt.experiment, epoch+1) )
            # torch.save(decoder.state_dict(), '%s/decoder_%d.pth' % (opt.experiment, epoch+1) )

        # val(epoch)

def val(epoch):
    """Validation Code"""
    global val_iteration
    encoder.eval()
    decoder.eval()

    cumm_losses = []
    total_samples = 0
    
    for i, dataBatch in enumerate(segValLoader ):
        # Read data
        imBatch = Variable(dataBatch['im']).to(device)
        hmBatch = Variable(dataBatch['heatmap']).to(device)

        num_samples = imBatch.size()[0]
        total_samples += num_samples
    
        x1, x2, x3, x4, x5 = encoder(imBatch )
        pred = decoder(imBatch, x1, x2, x3, x4, x5 )
        loss = Loss(pred, hmBatch)
    
        cumm_losses.append(loss.cpu().data.item()*num_samples)

    epoch_loss = np.sum(cumm_losses)/total_samples

    # write to tensorboard
    writer.add_scalar('data/val_loss', epoch_loss, epoch)
        
train()