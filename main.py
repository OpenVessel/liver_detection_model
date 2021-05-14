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
