import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import get_patches, recon_image
from model import Modified3DUNet
import os
from utils import normalize



image_file = '/media/bubbles/fecf5b15-5a64-477b-8192-f8508a986ffe/ai/ryan/miccai/kits19/brats/data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/tel_1/t1.nii.gz'
outpath = '../data/verse/test_images/images/'
image_name = image_file.split('/')[-1].split('.')[0]

if len(os.listdir(outpath))>1:
    print('Patches found at ',outpath)
    pass
else:
    print('Extracting patches at ',outpath)
    get_patches(image_file, outpath,'train')
    recon_image('../data/verse/test_images/images/', image_file, '../data/verse/test_images/image_recon')
    print('Done Extracting pacthes and reconstructing images')

# get all the image and mask path and number of images
test_image_paths = glob.glob('../data/verse/test_images/images/*.npy')
test_image_paths = sorted(test_image_paths, key = lambda files: files.split('/')[-1].split('.')[0][-3:] ) 

class CustomDataset(Dataset):
    def __init__(self, image_paths):

        self.image_paths = image_paths
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        image = np.array(image, dtype = np.uint8)
        image = torch.from_numpy(image)
        return image

    def __len__(self):  # return count of sample we have
        return len(self.image_paths)

test_dataset = CustomDataset(test_image_paths)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)


in_channels = 1
n_classes = 2
base_n_filter = 16
model = Modified3DUNet(in_channels, n_classes, base_n_filter).cuda()
checkpoint = '../data/verse/models/300_epochs_ce_weights/epoch_200_checkpoint.pth'
checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
prediction_save_path = '../data/verse/test_images/predictions'
print()
print('Getting Predictions')
for index, image in enumerate(test_loader):
    output_1, output_2 = model(image)
    np.save(os.path.join(prediction_save_path,image_name+str(index)), output.cpu().detach().numpy())
recon_image('../data/verse/test_images/predictions', image_file, '../data/verse/test_images/prediction_image/')