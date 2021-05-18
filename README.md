# liver_detection_model
Liver Detection Model is preprocessing method for sorting images in CT scan between liver and non-liver


The Liver Slice Classification model was trained on 10,728 PNG's with a validation set of 6,041 PNG's after 2:29:13.6 hours.  
The dataset used for training was sampled from the LiTs Database, with PNG's generated from 22 of the 131 patients. 

Requirements to replicate this process include Python version 3.8.x, with the required packages included in requirements.txt. 
The script slice_classification.py is used to generate and test the Liver Slice Classification model. 
The model is implemented in the file file_sorter.py, which is used to generate a sub dataset of the LiTs database that attempts to remove PNGs that are not liver containing.


### LiTS dataset - you will need to make account and download from their google drive
LiTS dataset --> https://competitions.codalab.org/competitions/17094#learn_the_details

### You will need to convert the LiTS dataset to matlab files PNGs using the script below or download the preprocessed dataset  --> here 



## Liver Detection Model newely generated dataset 
Download the dataset from here -- > 

## Evulation of the Liver Detection Model still needs implementation 
