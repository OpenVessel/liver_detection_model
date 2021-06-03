# Liver Detection Model
Liver Detection Model is preprocessing method for sorting images in CT scan between liver and non-liver


The Liver Slice Classification model was trained on 10,728 PNG's with a validation set of 6,041 PNG's after 2:29:13.6 hours.  
The dataset used for training was sampled from the LiTs Database, with PNG's generated from 22 of the 131 patients. 

Requirements to replicate this process include Python version 3.8.x, with the required packages included in requirements.txt. 
The script slice_classification.py is used to generate and test the Liver Slice Classification model. 
The model is implemented in the file file_sorter.py, which is used to generate a sub dataset of the LiTs database that attempts to remove PNGs that are not liver containing.

### Outline 
Our repository contains code to complete a few tasks. First, the script slice_detection_v1.py contains the code used to train and test the slice detection model. Second, and the most important items contained in our repo, are the scripts and steps to generate the dataset in order to train the detection model. The first of these scripts, preprocess_database.py, is used to convert the files from the LiTs database from nifti to .mat. The next of the scripts, datagenerator.py, contains functions to complete two tasks. The function pngs_from_mat is used to take the .mat files resulting from preprocess_database.py to pngs, which are the input data type for the liver detection model. The other function, new_data_with_liver_detection puts to use the resulting model after training. It attempts to successfully use the trained model to eliminiate excess png's. It acts as a preprocessing step in a larger model pipeline (link barcelona here), and attempts to remove those files that do not contain livers in order to speed up training time and decrease pixel imbalance, mentioned by Merriam Belliver (link master thesis here). 
The three scripts metioned above are called together in main.py, for ease of use. To get started first clone the repository, and then look to download your dataset of choice. 
### LiTS dataset - you will need to make account and download from their google drive
LiTS dataset --> https://competitions.codalab.org/competitions/17094#learn_the_details
Base form of data, will have to convert to .mat, then png's before training the liver detection model. 


### To skip data conversion steps and download the entire 131 patient LiTs database already converted, download this dataset --> here 



## Dataset for use with larger model pipeline -->  

## Evulation of the Liver Detection Model still needs implementation 
