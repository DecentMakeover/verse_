import os
import SimpleITK as sitk
import numpy as np
from numpy.lib import stride_tricks
import tqdm
import shutil
import torch
import logging
import sys
import json

#Step 1 read in the images
def read_image_and_seg(image_file_path, seg_file_path):
    """
    takes in filepath and returns the numpy array of image and seg
    """
    image = sitk.ReadImage(image_file_path)
    seg   = sitk.ReadImage(seg_file_path)
    image = sitk.GetArrayFromImage(image)
    seg   = sitk.GetArrayFromImage(seg)
    return image , seg

def read_sitk_image(image_file_path):
    """
    takes in filepath and returns the numpy array of image and seg
    """
    image = sitk.ReadImage(image_file_path)
    image = sitk.GetArrayFromImage(image)
    return image

def read_sitk_image(mask_file_path):
    """
    takes in filepath and returns the numpy array of image and seg
    """
    mask = sitk.ReadImage(mask_file_path)
    mask = sitk.GetArrayFromImage(mask)
    return image

#Logging
def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def get_patches(image, out_path, phase):
    with open('../data/verse/shape_info.json', 'r') as fp:
        data = json.load(fp)
    image_name = image
    read_image = sitk.ReadImage(image)
    image = sitk.GetArrayFromImage(read_image)
    patch_size = 128

    depth, height , width  = image.shape

    if height < patch_size:
        delta_h = patch_size - height
        delta_h+=patch_size#Can you check this again
        image =np.pad(image, ((0,0), (0,delta_h), (0, 0)), 'constant')

    if width < patch_size:
        delta_w = patch_size - width
        delta_w+=patch_size
        image =np.pad(image, ((0,0), (0,0), (0, delta_w)), 'constant')

    if depth < patch_size:
        delta_z = patch_size - depth
        delta_z+=patch_size
        image =np.pad(image, ((0,delta_z), (0,0), (0, 0)), 'constant')

    depth, height , width  = image.shape
    if not height%patch_size==0:
        mod = height%patch_size
        delta_h = patch_size - mod
        image =np.pad(image, ((0,0), (0,delta_h), (0, 0)), 'constant')

    if not width%patch_size==0:
        mod = width%patch_size
        delta_w = patch_size - mod
        image =np.pad(image, ((0,0), (0,0), (0, delta_w)), 'constant')

    if not depth%patch_size==0:
        mod = depth%patch_size
        delta_z = patch_size - mod
        image =np.pad(image, ((0,delta_z), (0,0), (0, 0)), 'constant')

    count=0
    depth_step = image.shape[0] - patch_size 
    height_step  = image.shape[1] - patch_size 
    width_step  = image.shape[2] - patch_size   
    data[image_name] = image.shape

    with open('../data/verse/shape_info.json', 'w') as fp:
        json.dump(data, fp)
    for z in range(0, depth_step+1, patch_size):
        for y in range(0, height_step+1, patch_size):
            for x in range(0,width_step+1, patch_size):
                patch = image[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                if phase == 'train':
                    np.save(os.path.join('{}','{}').format(out_path,image_name.split('/')[-1].split('.')[0]+str(count)), patch)
                    count+=1
                else:
                    patch[patch>1]=1
                    np.save(os.path.join('{}','{}').format(out_path,image_name.split('/')[-1].split('_seg')[0]+str(count)), patch)
                    count+=1

def recon_image(npy_folder,original_image, out_path):
    #first convert these to a single numpy array
    with open('../data/verse/shape_info.json', 'r') as fp:
        data = json.load(fp)
    image_name = original_image
    original_image = sitk.ReadImage(original_image)
    origin = original_image.GetOrigin()
    direction = original_image.GetDirection()
    image_to_fill = np.zeros((data[image_name]))

    filenames = os.listdir(npy_folder)
    filenames = sorted(filenames, key = lambda files: files.split('/')[-1].split('.')[0][-3:] ) 
    
    patch_size = 128

    count=0
    depth_step = image_to_fill.shape[0] - patch_size 
    height_step  = image_to_fill.shape[1] - patch_size 
    width_step  = image_to_fill.shape[2] - patch_size   

    for z in range(0,depth_step+1 , patch_size):
        for y in range(0,height_step+1, patch_size):
            for x in range(0,width_step+1,patch_size):
                image_to_fill[z:z+patch_size, y:y+patch_size,x:x+patch_size] = np.load(os.path.join(npy_folder, filenames[count]))
                count+=1
    convert_to_image = sitk.GetImageFromArray(image_to_fill)
    convert_to_image.SetOrigin(origin)
    convert_to_image.SetDirection(direction)
    sitk.WriteImage(convert_to_image,os.path.join(out_path,image_name.split('/')[-1].split('.')[0]+'recon.nii.gz'))
    return 

#Borrowed from https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def get_metrics(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return  TPR , TNR , PPV , FPR , FNR , ACC

def remove_non_label_patches(mask_dir, image_dir):
    for label in os.listdir(mask_dir):
        read_label = np.load(os.path.join(mask_dir, label))
        if len(np.unique(read_label)) <2:
            os.remove(os.path.join(mask_dir, label))
            os.remove(os.path.join(image_dir, label))
def get_image_from_npy(npy_file,out_path):
    file = np.load(npy_file)
    file = sitk.GetImageFromArray(file)
    return sitk.WriteImage(file,os.path.join(out_path,npy_file.split('/')[-1].split('.')[0]+'.nii.gz') )

MIN_BOUND = -100.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

if __name__ == "__main__":
    # get_image_from_npy('../data/verse/test_images/predictions/verse0330.npy', '../data/verse/test_images/')
    # get_image_from_npy('../data/verse/test_images/images/verse0330.npy', '../data/verse/test_images/')

    image_dir = '../data/verse/patches/images/'
    mask_dir  = '../data/verse/patches/masks/'
    remove_non_label_patches(mask_dir, image_dir)

    # image_folder = '../data/verse/images/image/'
    # mask_folder  = '../data/verse/images/masks/'
    # count =0
    # for image in os.listdir(image_folder):
    #     label = image.split('/')[-1].split('.')[0]+'_seg.nii.gz'
    #     print(os.path.join(image_folder, image))
    #     print(os.path.join(mask_folder,label))
    #     get_patches(os.path.join(image_folder, image), '../data/verse/patches/images/','train')
    #     get_patches(os.path.join(mask_folder, label), '../data/verse/patches/masks/','mask')