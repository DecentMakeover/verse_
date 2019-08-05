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

def get_patches(image, out_path):
    with open('shape_info.json', 'r') as fp:
        data = json.load(fp)
    image_name = image
    read_image = sitk.ReadImage(image)
    image = sitk.GetArrayFromImage(read_image)
    print('BEFORE', image.shape)
    image = np.transpose(image, (1, 2, 0))

    print('AFTER ', image.shape)

    patch_size = 128

    height, width , depth  = image.shape

    # print(image[0,0,:])
    if height < patch_size:
        delta_h = patch_size - height
        delta_h+=patch_size#Can you check this again
        image =np.pad(image, ((0,delta_h), (0,0), (0, 0)), 'constant')

    if width < patch_size:
        delta_w = patch_size - width
        delta_w+=patch_size
        image =np.pad(image, ((0,0), (0,delta_w), (0, 0)), 'constant')

    if depth < patch_size:
        delta_z = patch_size - depth
        delta_z+=patch_size
        image =np.pad(image, ((0,0), (0,0), (0, delta_z)), 'constant')
    # print(image[0,0,:])    
    height, width , depth  = image.shape

    if not height%patch_size==0:
        mod = height%patch_size
        delta_h = patch_size - mod
        image =np.pad(image, ((0,delta_h), (0,0), (0, 0)), 'constant')
    
    if not width%patch_size==0:
        mod = width%patch_size
        delta_w = patch_size - mod
        image =np.pad(image, ((0,0), (0,delta_w), (0, 0)), 'constant')
    
    if not depth%patch_size==0:
        mod = depth%patch_size
        delta_z = patch_size - mod
        image =np.pad(image, ((0,0), (0,0), (0, delta_z)), 'constant')

    count=0
    height_step = image.shape[0] - patch_size 
    width_step  = image.shape[1] - patch_size 
    depth_step  = image.shape[2] - patch_size   

    data[image_name] = image.shape


    print('GET PATCH ', image.shape)
    with open('shape_info.json', 'w') as fp:
        json.dump(data, fp)
    for z in range(0, depth_step+1, patch_size):
        for y in range(0, width_step+1, patch_size):
            for x in range(0,height_step+1, patch_size):
                print(os.path.join(out_path,image_name.split('/')[-1].split('.')[0]+str(count)))
                patch = image[x:x+patch_size, y:y+patch_size, z:z+patch_size]
                np.save(os.path.join('{}','{}').format(out_path,image_name.split('/')[-1].split('.')[0]+str(count)), patch)
                count+=1
                print(count)
def recon_image(npy_folder,original_image, out_path):
    print('right here')
    #first convert these to a single numpy array
    with open('shape_info.json', 'r') as fp:
        data = json.load(fp)
    print('noe here')
    image_name = original_image
    print(original_image)
    original_image = sitk.ReadImage(original_image)
    # print('yo')
    # print(original_image.shape)
    # original_image = np.transpose(original_image, (1, 2, 0))
    origin = original_image.GetOrigin()
    origin=(origin[1],origin[2],origin[0])
    print(type(origin))
    direction = original_image.GetDirection()
    print(direction)
    image_to_fill = np.zeros((data[image_name]))

    print('RECON ', image_to_fill.shape)
    # filenames = os.listdir(npy_folder)
    # filenames = sorted(filenames, key = lambda files: files.split('/')[-1].split('.')[0][-3:] ) 
    
    patch_size = 128

    count=0
    height_step = image_to_fill.shape[0] - patch_size 
    width_step  = image_to_fill.shape[1] - patch_size 
    depth_step  = image_to_fill.shape[2] - patch_size   
    filepath='verse033'
    # print(filenames)
    for z in range(0,depth_step+1 , patch_size):
        for y in range(0,width_step+1, patch_size):
            for x in range(0,height_step+1,patch_size):
                filename=filepath+str(count)+'.npy'
                print(filename)
                image_to_fill[x:x+patch_size, y:y+patch_size,z:z+patch_size] = np.load(os.path.join(npy_folder, filename))
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

def remove_non_label_patches(image_patches, mask_patches):
    for mask in os.listdir(mask_patches):
        load_mask = np.load(os.path.join(mask_patches, mask))
        print(np.unique(load_mask))


MIN_BOUND = -100.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

if __name__ == "__main__":
    get_patches('verse033_seg.nii.gz', 'image_mask/')
    # recon_image('image_recon/', 'verse033.nii.gz', 'image_recon')