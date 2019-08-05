import os
import numpy as np
import json
import SimpleITK as sitk
def get_patches(image, out_path):
    with open('shape_info.json', 'r') as fp:
        data = json.load(fp)
    image_name = image
    read_image = sitk.ReadImage(image)
    image = sitk.GetArrayFromImage(read_image)
    patch_size = 128

    height, width , depth  = image.shape


    if height < patch_size:
        delta_h = patch_size - height
        delta_h+=patch_size
        image =np.pad(image, ((0,delta_h), (0,0), (0, 0)), 'constant')

    if width < patch_size:
        delta_w = patch_size - width
        delta_w+=patch_size
        image =np.pad(image, ((0,0), (0,delta_w), (0, 0)), 'constant')

    if depth < patch_size:
        delta_z = patch_size - depth
        delta_z+=patch_size
        image =np.pad(image, ((0,0), (0,0), (0, delta_z)), 'constant')

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

    with open('shape_info.json', 'w') as fp:
        json.dump(data, fp)
    for z in range(0, depth_step, patch_size):
        for y in range(0, width_step, patch_size):
            for x in range(0,height_step, patch_size):
                print(os.path.join(out_path,image_name.split('/')[-1].split('.')[0]+str(count)))
                patch = image[x:x+patch_size, y:y+patch_size, z:z+patch_size]
                np.save(os.path.join('{}','{}').format(out_path,image_name.split('/')[-1].split('.')[0]+str(count)), patch)
                count+=1

def recon_image(npy_folder,original_image, out_path):
    #first convert these to a single numpy array
    with open(' shape_info.json', 'r') as fp:
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
    height_step = image_to_fill.shape[0] - patch_size 
    width_step  = image_to_fill.shape[1] - patch_size 
    depth_step  = image_to_fill.shape[2] - patch_size   

    for z in range(0,depth_step , patch_size):
        for y in range(0,width_step, patch_size):
            for x in range(0,height_step,patch_size):
                image_to_fill[x:x+patch_size, y:y+patch_size,z:z+patch_size] = np.load(os.path.join(npy_folder, filenames[count]))
                count+=1
    convert_to_image = sitk.GetImageFromArray(image_to_fill)
    convert_to_image.SetOrigin(origin)
    convert_to_image.SetDirection(direction)
    sitk.WriteImage(convert_to_image,os.path.join(out_path,image_name.split('/')[-1].split('.')[0]+'recon.nii.gz'))
    return
    
if __name__ == '__main__':
    get_patches('verse009_seg.nii.gz', 'store/')
    recon_image('store/','verse009_seg.nii.gz', './')
