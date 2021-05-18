####




#Proir arthors 
#Nathan Reilly 
#Gregory Glatzer 

#5/14/2021 formatting and citation work
#Leslie Wubbel 

###

import os
import nibabel as nib # pip install nibabel  # https://nipy.org/nibabel/nifti_images.html
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image
import time 
start = time.time()

# path constants
current_directory = os.getcwd()
print(current_directory)
niftis_path = 'E:\Datasets\LiTS_liver_lesion\LITS17' ## Change this line to where LITS17 dataset is solved 
root_process_database = current_directory + '\liver_detection_model\data_output'   ## output path goes here

print(root_process_database)
## WINDOWS STRINGS
# niftis_path = r'E:\Datasets\LiTS_liver_lesion\LITS17'
# root_process_database = r'D:\L_pipe\liver_open\liverseg-2017-nipsws\output_folder'
## Folders to be created
folder_volumes = os.path.join(root_process_database, 'images_volumes/')
folder_seg_liver = os.path.join(root_process_database, 'liver_seg/')
folder_seg_item = os.path.join(root_process_database, 'item_seg/')


# create non-existent paths
folder_paths = [root_process_database, folder_volumes, folder_seg_liver, folder_seg_item]
    
for p in folder_paths:
    if not os.path.exists(p):
        os.mkdir(p)


# filter to only files starting with v (volume) or s (segmentation)
filenames = [filename for filename in os.listdir(niftis_path) if filename[0] in ('v', 's')]


for filename in filenames:
    path_file = os.path.join(niftis_path, filename)
    index = filename.find('.nii') + 1 # +1 to account for matlab --> python
    if filename[0] == 'v':
        print('Processing Volume {}'.format(filename))
        folder_volume = os.path.join(folder_volumes, filename[7:index-1])
        volume = nib.load(path_file) # load nifti
        imgs = volume.get_fdata() # get 3d NumPy array

        # clipping HU pixel clipping 
        imgs[imgs<-150] = -150
        imgs[imgs>250] = 250

        ## # equivalent to matlab single()
        imgs = imgs.astype(np.float32) 
        img_max, img_min = (np.max(imgs), np.min(imgs))

        # create folder_volume folder
        img_volume = 255*(imgs - img_min)/(img_max-img_min)
        if not os.path.exists(folder_volume):
            os.mkdir(folder_volume)
        
        for k in range(img_volume.shape[2]):
            section = img_volume[:,:,k]
            filename_for_section = os.path.join(folder_volume, str(k+1) + '.mat')
            scipy.io.savemat(filename_for_section, {'section': section})
    else:
        print('Processing Segmentation {}'.format(filename))
        folder_seg_item_num = os.path.join(folder_seg_item, filename[13:index-1])
        folder_seg_liver_num = os.path.join(folder_seg_liver, filename[13:index-1])
        segmentation = nib.load(path_file)
        img_seg = segmentation.get_fdata().astype(np.uint8)
        print(img_seg.shape)
        # binarize and normalize data
        img_seg_item = img_seg.copy()
        img_seg_liver = img_seg.copy()

        # create masks
        img_seg_item[img_seg_item == 1] = 0
        img_seg_item[img_seg_item == 2] = 1
        img_seg_liver[img_seg_liver == 2] = 1

        # create dirs
        if not os.path.exists(folder_seg_item_num):
            os.mkdir(folder_seg_item_num)
        if not os.path.exists(folder_seg_liver_num):
            os.mkdir(folder_seg_liver_num)
        
        # save images
        for k in range(0, img_seg_item.shape[2]):
            print(filename, ", ", k)
            # item
            item_seg_section = np.fliplr(np.flipud(img_seg_item[:,:,k]*255)) # flip on both axes
            item_seg_filename = os.path.join(folder_seg_item_num, str(k+1) + '.png')
            im_item = Image.fromarray(item_seg_section)
            im_item.save(item_seg_filename)
            
            # liver
            liver_seg_section = np.fliplr(np.flipud(img_seg_liver[:,:,k]*255))
            liver_seg_filename = os.path.join(folder_seg_liver_num, str(k+1) + '.png')
            im_liver = Image.fromarray(liver_seg_section)
            im_liver.save(liver_seg_filename)


end = time.time()
print("Elapsed Time is:", end - start)