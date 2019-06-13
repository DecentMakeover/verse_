import argparse
import os
import shutil
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch import optim
from torch.utils.data import DataLoader
from model import Modified3DUNet
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import utils
from losses import GeneralizedDiceLoss, dice_loss
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils import get_logger
from sklearn.metrics import confusion_matrix
from utils import get_metrics
from sklearn.metrics import accuracy_score
from utils import normalize

experiment_name = 'generalizedDice_with_normalization'
logger = get_logger(experiment_name)
writer = SummaryWriter(os.path.join('../data/models/logs/',experiment_name ) )
if not os.path.exists('../data/models/{}'.format(experiment_name)):
    os.mkdir(os.path.join('../data/models/',experiment_name))
'''
This will fetch the data and give it to the network -- helps in step 2 of the repo design
'''

# get all the image and mask path and number of images
folder_data = glob.glob('../data/patches/images/*.npy')
folder_mask = glob.glob('../data/patches/masks/*.npy')

# split these path using a certain percentage
len_data = len(folder_data)
train_size = 0.8

train_image_paths = folder_data[:int(len_data*train_size)]
test_image_paths = folder_data[int(len_data*train_size):]

train_mask_paths = folder_mask[:int(len_data*train_size)]
test_mask_paths = folder_mask[int(len_data*train_size):]


class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        # print(self.image_paths[index])
        # print(self.target_paths[index])
        image = np.load(self.image_paths[index])
        mask = np.load(self.target_paths[index])
        image = torch.from_numpy(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))

        return image, mask

    def __len__(self):  # return count of sample we have
        return len(self.image_paths)


train_dataset = CustomDataset(train_image_paths, train_mask_paths, train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=12)

test_dataset = CustomDataset(test_image_paths, test_mask_paths, train=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12)


in_channels = 1
n_classes = 26
base_n_filter = 16
model = Modified3DUNet(in_channels, n_classes, base_n_filter).cuda()
loss_function = GeneralizedDiceLoss()

epochs = 200
for epoch in range(epochs):
    model.train()
    logger.info('Starting  @ epoch {}'.format(epoch))
    start = time.time()
    losses = []
    mean_accuracy = []
    for index,(image, mask) in enumerate(train_loader):
        image = normalize(image)
        image = torch.unsqueeze(image,0).float().cuda()
        label = mask.cuda().long()
        labels_for_conf = mask

        output_1, output_2 = model(image)
        one_hot_encode_labels = F.one_hot(label,n_classes)
        one_hot_encode_labels = one_hot_encode_labels.permute(0,4,1,2,3).contiguous()
        loss = loss_function(output_2,one_hot_encode_labels)
        softmax = nn.Softmax(dim=1)
        output_2 = softmax(output_2)
        conf_matrix = confusion_matrix(torch.argmax(output_2,1).view(-1).cpu().detach().numpy(), labels_for_conf.view(-1).cpu().detach().numpy())
        TPR,TNR, PPV, FPR ,FNR, ACC = get_metrics(conf_matrix)
        accuracy = accuracy_score(labels_for_conf.view(-1).cpu().detach().numpy(), torch.argmax(output_2,1).view(-1).cpu().detach().numpy())
        mean_accuracy.append(accuracy)        
        logger.info('TPR == {} | \nTNR == {} | \nPRCSN == {} | \nFPR == {}\n | \nFNR == {} | \nACC == {}.'.format(TPR,TNR,PPV,FPR, FNR, ACC))
        logger.info('Epoch = {} , Accuracy  =  {}'.format(epoch, accuracy))
        losses.append(loss.item())
        optimizer = optim.Adam(model.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train/Loss', loss, epoch)
        writer.add_scalar('Train/Accuracy', accuracy, epoch)

    logger.info('Mean Loss     =  {}'.format(sum(losses) / float(len(losses))))
    logger.info('Mean Accuracy =  {}'.format(sum(mean_accuracy) / float(len(mean_accuracy))))
    end = time.time() - start
    logger.info('Time taken to finish epoch {} is {}'.format(epoch, end))
    average_accuracy = sum(mean_accuracy) / float(len(mean_accuracy))
    average_loss =  sum(losses) / float(len(losses))
    #borrowed from https://medium.com/udacity-pytorch-challengers/saving-loading-your-model-in-pytorch-741b80daf3c
    checkpoint = {'epoch': epoch, 'state_dict' :model.state_dict(), 'optimizer':optimizer.state_dict(), 'accuracy': average_accuracy, 'loss': average_loss}
    torch.save(checkpoint, '../data/models/{}/epoch_{}_checkpoint.pth'.format(experiment_name, epoch))
writer.close()