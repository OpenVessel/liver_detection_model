import os
import tensorflow as tf
import time
import math

from class_calls.py import liver_detection

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mat_file_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes"

# Global vars and driver
if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train or Test the Liver Lesion Segmentation Model")
    parser.add_argument('mode', help="'test' or 'train' depending on what you wish to do.")
    cmdline = parser.parse_args()

    from config import Config
    
    config = Config()
    config.labels = True # Change to false if we don't have labels

    ##class_calls here ## calls the model the model is either set to training or testing
    liver_detection = liver_detection(config)
    

    if cmdline.mode == "test":
        ##implement class call 
        liver_detection.test(testing_volume)
    elif cmdline.mode == "train":
        liver_detection.train(training_volume, validation_volume = testing_volume )
    else:
        raise BaseException('Invalid mode. Must be test or train')
import slice_classification_v1 as sc
import datagenerator as dg 
import preprocess_database_liver as pdl

#generate train files 
num_patients = input("How many files (1-131) are being used in the train dataset?")
nifti_bool = input("LiTs database in nifti format still (y/n)?") 
# mat_file_path =  config file 
# liver_seg_path =  config file 
# outpath =  config file
# nifti_path = config file
# root_process_database = config file 
#first generate the .mat from nifti if not already done 
if nifti_bool == "y":
    pdl.gen_mat_pngs_from_nifti(niftis_path, root_process_database)
#generate train files for slice classification model

dg.pngs_from_mat(mat_file_path, liver_seg_path, outpath, num_patients)

if config["train"] == True:
    sc.train_model(outpath)
