import torch
from delira.training import Parameters
from trixi.logger import PytorchVisdomLogger
from delira.logging import TrixiHandler
import logging
from sklearn.metrics import accuracy_score
#please check logger status

params = Parameters(fixed_params={
    "model": {
        "in_channels": 1, 
        "num_classes": 26
    },
    "training": {
        "batch_size": 64, # batchsize to use
        "num_epochs": 10, # number of epochs to train
        "optimizer_cls": torch.optim.Adam, # optimization algorithm to use
        "optimizer_params": {'lr': 1e-3}, # initialization parameters for this algorithm
        "losses": {"CE": torch.nn.CrossEntropyLoss()}, # the loss function
        "lr_sched_cls": None,  # the learning rate scheduling algorithm to use
        "lr_sched_params": {}, # the corresponding initialization parameters
         "criterions":accuracy_score,
        "metrics": {'img':{ 'l1': accuracy_score}} # and some evaluation metrics
    }
})


import SimpleITK as sitk
import numpy as np

# load image and mask
img_file = '../data/images/image/verse005300.nii.gz'
mask_file = '../data/images/masks/verse005300.nii.gz'
img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
img = img.astype(np.float32)
mask = mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
mask = mask.astype(np.float32)

assert mask.shape == img.shape
print(img.shape)
print(np.unique(mask))

from delira.data_loading import AbstractDataset

class CustomDataset(AbstractDataset):
    def __init__(self, img, mask, num_samples=1000):
        super().__init__(None, None)
    #why are there 4 None?
#$$$        super().__init__(None, None, None, None)
        self.data = {"data": img.reshape(1, *img.shape), "label": mask.reshape(1, *mask.shape)}
        self.num_samples = num_samples
        
    def __getitem__(self, index):
        return self.data
    
    def __len__(self):
        return self.num_samples


dataset_train = CustomDataset(img, mask, num_samples=10000)
dataset_val = CustomDataset(img, mask, num_samples=1)

from batchgenerators.transforms import RandomCropTransform, \
                                        ContrastAugmentationTransform, Compose
from batchgenerators.transforms.spatial_transforms import ResizeTransform
from batchgenerators.transforms.sample_normalization_transforms import MeanStdNormalizationTransform

transforms = Compose([
    ContrastAugmentationTransform(), # randomly adjust contrast
    MeanStdNormalizationTransform(mean=[img.mean()], std=[img.std()])]) # use concrete values since we only have one sample (have to estimate it over whole dataset otherwise)


from delira.data_loading import BaseDataManager, SequentialSampler, RandomSampler


manager_train = BaseDataManager(dataset_train, params.nested_get("batch_size"),
                                transforms=transforms,
                                sampler_cls=RandomSampler,
                                n_process_augmentation=4)

manager_val = BaseDataManager(dataset_val, params.nested_get("batch_size"),
                              transforms=transforms,
                              sampler_cls=SequentialSampler,
                              n_process_augmentation=4)

import warnings
warnings.simplefilter("ignore", UserWarning) # ignore UserWarnings raised by dependency code
warnings.simplefilter("ignore", FutureWarning) # ignore FutureWarnings raised by dependency code


from delira.training import PyTorchExperiment
from delira.training.train_utils import create_optims_default_pytorch
from delira.models.segmentation import UNet3dPyTorch

# logger.info("Init Experiment")
print(create_optims_default_pytorch)
experiment = PyTorchExperiment(params, UNet3dPyTorch,
                               name="Segmentation3dExample",
                               save_path="./tmp/delira_Experiments",
                               optim_builder=create_optims_default_pytorch,
                               gpu_ids=[0], mixed_precision=True)
experiment.save()
print(manager_train, manager_val)
model = experiment.run(manager_train, manager_val)