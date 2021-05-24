import os
import tensorflow as tf
import time
import math

from datagenerator import pngs_from_mat
import preprocess_database_liver as pdl
from class_calls import LiverDetection
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# Global vars and driver
if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train or Test the Liver Lesion Segmentation Model")
    parser.add_argument('mode', help="'test' or 'train' depending on what you wish to do.")
    cmdline = parser.parse_args()

    from config import Config
    
    config = Config()
    config.labels = True # Change to false if we don't have labels
    num_patients = input("How many files (1-131) are being used in the train dataset?")
    nifti_bool = input("LiTs database in nifti format still (y/n)?") 

    mat_file_path =  config.mat_file_path
    liver_seg_path =  config.liver_seg_path
    preprocessing_outpath =  config.outpath
    nifti_path = config.nifti_path
    root_process_database = config.root_process_database
    output_path_of_model = config.output_model
    #first generate the .mat from nifti if not already done 
    if nifti_bool == "y":
        pdl.gen_mat_pngs_from_nifti(nifti_path, root_process_database)
    #generate train files for slice classification model
    
    #check to see if data present in processingoutpath
    print(os.path.join(preprocessing_outpath, 'Train','Liver'))
    if len(os.path.join(preprocessing_outpath, 'Train','Liver')) == 0: #could check other folders as well
        pngs_from_mat(mat_file_path, liver_seg_path, preprocessing_outpath, num_patients)
    else:
        print('Data already present... \n Beginning retraining')
    
    ##class_calls here ## calls the model the model is either set to training or testing
    liver_det = LiverDetection(config)
    

    if cmdline.mode == "test":
        ##implement class call 
        liver_det.test(config, output_path_of_model)
    elif cmdline.mode == "train":
        liver_det.train(config,  output_path_of_model)
    else:
        raise BaseException('Invalid mode. Must be test or train')


