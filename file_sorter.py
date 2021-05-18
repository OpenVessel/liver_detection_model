from PIL import Image
import numpy as np
import os
import tensorflow as tf
from scipy.io import loadmat, savemat
import shutil
import random 

def pngs_from_mat(mat_file_path, liver_seg_path, outpath, numpatients = 25):
    # generate train, test, validation pngs from .mat files
    # mat_file_path: the path that contains each patients .mat files 
    # outpath: root path to save the generated pngs
    # numpatients: the number of patients used to train the detection model (from 1 - 131)
    # end result --> saves pngs to train, test, validation pathing separated by liver and non-liver

    assert(numpatients <= 131) 

    train = os.path.join(outpath, "Train")
    validation = os.path.join(outpath, "Validation")

    liver_train = os.path.join(train, "Liver") 
    liver_validation =  os.path.join(validation, "Liver")
    nl_train = os.path.join(train, "Non-Liver")
    nl_validation = os.path.join(validation, "Non-Liver") 

    data_outpaths = [train, validation, liver_train, liver_validation, nl_train, nl_validation]
    for path in data_outpaths:
        if not os.path.exists(path):
            os.mkdir(path)


    #generates a list of randomly generated numbers (patients in range 0,130) 
    #to be used for the train, validation data for detection model
    patient_list = random.sample(range(131), numpatients)
    cutoff = int( len(patient_list) *.8 )

    for patient in os.listdir(mat_file_path):
        if patient in patient_list:

            patient_liver_segpath = os.path.join(liver_seg_path, patient)
            mat_file_patientpath = os.path.join(mat_file_path, patient)
            for file in os.listdir(mat_file_patientpath):
                
                #reminder to make a function for nifti to .mat conversion

                number = file[:-4]
                mat_file = os.path.join(mat_file_patientpath, file) #the individual files
                
                mat_array = np.array(loadmat(mat_file)['section'])
                mat_png = Image.fromarray(mat_array).convert('RGB')

                image = Image.open(os.path.join(liver_seg_path, patient, str(number + ".png")))
                image = np.array(image)
                
                #if there is white present in the image (if the ground truth image contains liver/white pixels)
                if np.count_nonzero(image) != 0:
                    print(patient, number)
                    if patient in patient_list[:cutoff]: #first 80% of the patients in the file list 
                        x = os.path.join(liver_train, str(number + "_" + patient + ".png"))
                        mat_png.save(x)
                    else:
                        y = os.path.join(liver_validation, str(number + "_" + patient + ".png")) 
                        mat_png.save(y)             
                #if the image is all black (not containing liver/white pixels)
                else:
                    if patient in patient_list[:cutoff]:
                            X = os.path.join(nl_train, str(number +  "_" +patient + ".png"))
                            mat_png.save(X)
                    else:
                        Y = os.path.join(nl_validation, str(number +  "_" +patient + ".png"))
                        mat_png.save(Y)


def new_data_with_liver_detection_model(mat_file_path, classifier_path, new_dataset_outpath, liver_seg_path, item_seg_path):
    # sorting the files (once the models done training)


    # current working directory logic
    current_directory = os.getcwd()
    print(current_directory)
    current_directory = current_directory.replace("\\utils","")
    print(current_directory)
    niftis_path = 'E:\Datasets\LiTS_liver_lesion\LITS17' ## Change this line to where LITS17 dataset is solved 
    root_process_database = current_directory + '\data_output'   ## output path goes here
    print(root_process_database)
    
    # mat_file_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes"
    # classifier = tf.keras.models.load_model(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Liver PNGs from matlab seperated\ClassificationModel")
    # new_dataset_outpath = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\refined LiTs database"
    # liver_seg_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\liver_seg"
    # item_seg_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\item_seg"

    #load model here
    classifier = tf.keras.models.load_model(classifier_path)

    # create patient paths in new folder 
    path_list = [os.path.join(new_dataset_outpath, "image_volumes"), os.path.join(new_dataset_outpath, "item_seg"), os.path.join(new_dataset_outpath, "liver_seg")]
    
    #make each patient path in the new paths (EX: images_volumes/129)
    for i in range(0,131):
        for path in path_list:
            if not os.path.exists(path):
                os.mkdir(path)
                new_path = os.path.join(path, str(i))
                if not os.path.exists(new_path):
                    os.mkdir(new_path)

    for patient in os.listdir(mat_file_path):

        print("Patient: ", patient)
        mat_path = os.path.join(mat_file_path, patient)
    
        keep_list = []
        num_files = len(os.listdir(mat_path)) + 1

        for mat in os.listdir(mat_path):
            
            #mimic the data flow of training for VB's model
            #load and convert (mat --> png --> numpy array)

            number = mat[:-4] #just get the number of the corresponding file
            mat_file = os.path.join(mat_path, mat)
            
            mat_array = np.array(loadmat(mat_file)['section']) #['section'] what does this do????
            mat_png = Image.fromarray(mat_array).convert('RGB')

            back_to_array = np.asarray(mat_png)
            back_to_array = np.expand_dims(back_to_array, axis = 0) #also data type of uint-8 
            
            val = classifier.predict(back_to_array)
            
            if val[0] < float(.55):
                keep_list.append(int(number))
        
        max_keep = max(keep_list)
        min_keep = min(keep_list)
        min_minus = int( min_keep - (.10 * num_files) )
        max_plus = int( max_keep + (.10 * num_files) )
    
        print("Range for {} : {} to {}. ".format(patient, min_minus, max_plus))


        # Case 1: max_plus too large, min_minus is good 
        if max_plus > num_files and min_minus >= 1:
            print("Case 1")
            for i in range(min_minus, num_files):
                # i = the number corresponding to file type

                #save item_seg 
                item_png = Image.open(os.path.join(item_seg_path, patient, str(str(i) + ".png")))
                item_png.save(os.path.join(new_dataset_outpath, "item_seg", patient, str(str(i) + ".png")))

                #save image_volume mat file
                i_mat = str(str(i) + ".mat")
                mp = os.path.join(mat_file_path, patient, i_mat)
                
                shutil.copy(mp, os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")))

                #save liver_seg png 
                liver_png = Image.open(os.path.join(liver_seg_path, patient, str(str(i) + ".png")))
                liver_png.save(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))

                print("Saved....{}".format(patient))
                print(os.path.join(item_seg_path, patient, str(str(i) + ".png") + "\n"))
                print(os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")) + "\n")
                print(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))
        

        #CASE 2: min_minus too small, max_plus good 
        if min_minus < 1 and max_plus <= num_files:
            print("Case 2")
            for i in range(1, max_plus):
                # i = the number corresponding to file type

                #save item_seg 
                item_png = Image.open(os.path.join(item_seg_path, patient, str(str(i) + ".png")))
                item_png.save(os.path.join(new_dataset_outpath, "item_seg", patient, str(str(i) + ".png")))

                #save image_volume mat file
                i_mat = str(str(i) + ".mat")
                mp = os.path.join(mat_file_path, patient, i_mat)
                
                shutil.copy(mp, os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")))

                #save liver_seg png 
                liver_png = Image.open(os.path.join(liver_seg_path, patient, str(str(i) + ".png")))
                liver_png.save(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))
                
                print("Saved....{}".format(patient))
                print("Saved....\n")
                print(os.path.join(item_seg_path, patient, str(str(i) + ".png") + "\n"))
                print(os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")) + "\n")
                print(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))


        # CASE 3: both outside of range
        if min_minus < 1 and max_plus > num_files:
            print("Case 3")
            for i in range(1, num_files):
                # i = the number corresponding to file type

                #save item_seg 
                item_png = Image.open(os.path.join(item_seg_path, patient, str(str(i) + ".png")))
                item_png.save(os.path.join(new_dataset_outpath, "item_seg", patient, str(str(i) + ".png")))

                #save image_volume mat file
                i_mat = str(str(i) + ".mat")
                mp = os.path.join(mat_file_path, patient, i_mat)
                
                shutil.copy(mp, os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")))

                #save liver_seg png 
                liver_png = Image.open(os.path.join(liver_seg_path, patient, str(str(i) + ".png")))
                liver_png.save(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))
                
                print("Saved....{}".format(patient))
                print("Saved....\n")
                print(os.path.join(item_seg_path, patient, str(str(i) + ".png") + "\n"))
                print(os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")) + "\n")
                print(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))

        if min_minus >= 1 and max_plus <= num_files:
            print("Case 4")
            for i in range(min_minus, max_plus):

                item_png = Image.open(os.path.join(item_seg_path, patient, str(str(i) + ".png")))
                item_png.save(os.path.join(new_dataset_outpath, "item_seg", patient, str(str(i) + ".png")))

                #save image_volume mat file
                i_mat = str(str(i) + ".mat")
                mp = os.path.join(mat_file_path, patient, i_mat)
                
                shutil.copy(mp, os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")))

                #save liver_seg png 
                liver_png = Image.open(os.path.join(liver_seg_path, patient, str(str(i) + ".png")))
                liver_png.save(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))

                print("Saved....{}".format(patient))
                print("Saved....\n")
                print(os.path.join(item_seg_path, patient, str(str(i) + ".png") + "\n"))
                print(os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")) + "\n")
                print(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))