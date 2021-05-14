from PIL import Image
import numpy as np
import os
import tensorflow as tf
from scipy.io import loadmat, savemat
import shutil
​
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
​mat_file_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes"
​
classifier = tf.keras.models.load_model('classificationModel')
​

## Mat to png
# # #generate pngs from matlab
# # for patient in os.listdir(mat_file_path):
# #     if patient == "2":
# #         keep_list = []
# #         mat_file_patientpath = os.path.join(mat_file_path, patient)
# #         for file in os.listdir(mat_file_patientpath):
# #             file_num = file[:-4]
# #             mat_file = os.path.join(mat_file_patientpath, file)
# #             mat_array = np.array(loadmat(mat_file)['section'])
# #             mat_png = Image.fromarray(mat_array).convert('RGB')
# #             arr = np.asarray(mat_png)
# #             X = np.expand_dims(arr, axis = 0)
# #             images = np.vstack([X])
# #             val = classifier.predict(images)
# #             if val == 1:
# #                 continue 
# #             else:
#                 # print(patient, file, val)
​
# liver_seg_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\liver_seg"
# outpath = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Liver PNGs from matlab seperated"
# train_outpath = os.path.join(outpath, "Train")
# test_outpath = os.path.join(outpath, "Test")
# validation_outpath = os.path.join(outpath, "Validation")
# if not os.path.exists(train_outpath):
#     os.mkdir(train_outpath)
# if not os.path.exists(test_outpath):
#     os.mkdir(test_outpath)
# if not os.path.exists(validation_outpath):
#     os.mkdir(validation_outpath)
​
# liver_train = os.path.join(train_outpath, "Liver")
# nl_train = os.path.join(train_outpath, "Non-Liver")
# liver_test = os.path.join(test_outpath, "Liver")
# nl_test = os.path.join(test_outpath, "Non-Liver")
# liver_validation = os.path.join(validation_outpath, "Liver")
# nl_validation = os.path.join(validation_outpath, "Non-Liver")
# paths = [liver_train, nl_train, liver_test, nl_test, liver_validation, nl_validation]
# for path_ in paths:
#     if not os.path.exists(path_):
#         os.mkdir(path_)
​
​
# # generate train, test, validation pngs from .mat files
# for patient in os.listdir(mat_file_path):
#     if patient in ["62","7","72","30","127","128","1","37","21",
#                    "16","39","80","87","19","102","10","9","39",
#                    "6","8","11","73"]:
​
​
#         patient_liver_segpath = os.path.join(liver_seg_path, patient)
#         mat_file_patientpath = os.path.join(mat_file_path, patient)
#         for file in os.listdir(mat_file_patientpath):
​
#             number = file[:-4]
#             #convert matlab to pngs in a different folder
#             mat_file = os.path.join(mat_file_patientpath, file) #the individual files
#             #to pngs
#             mat_array = np.array(loadmat(mat_file)['section'])
#             mat_png = Image.fromarray(mat_array).convert('RGB')
​
#             png_outpath = os.path.join(outpath)
#             if not os.path.exists(png_outpath):
#                 os.mkdir(png_outpath)
            
#             image = Image.open(os.path.join(liver_seg_path, patient, str(number + ".png")))
#             image = np.array(image)
            
#             # curr = os.path.join(png_outpath, str(number + ".png"))
#             if np.count_nonzero(image) != 0: 
#                 print(patient, number)
#                 if patient in ["62","7","72","30","127","128","1","37","21",
#                    "16","39","80","87"]:
#                     x = os.path.join(liver_train, str(number + "_" + patient + ".png"))
#                     mat_png.save(x)
#                 if patient in ["19","102","10","9","6"]:
#                     y = os.path.join(liver_validation, str(number + "_" + patient + ".png"))
#                     mat_png.save(y)
#                 if patient in ["8","11","73"]:
#                     z = os.path.join(liver_test, str(number + "_" + patient + ".png"))
#                     mat_png.save(z)
#             else:
#                 if patient in ["62","72","30","127","128","1","37","21",
#                    "7","16","39","80","87"]:
#                         X = os.path.join(nl_train, str(number +  "_" +patient + ".png"))
#                         mat_png.save(X)
​
#                 if patient in ["19","102","10","9","6"]:
#                     Y = os.path.join(nl_validation, str(number +  "_" +patient + ".png"))
#                     mat_png.save(Y)
#                 if patient in ["8","11","73"]:
#                     Z = os.path.join(nl_test, str(number +  "_" +patient + ".png"))
#                     mat_png.save(Z)
            
​
# sorting the files (once the models done training)
​
# LOAD MODEL HERE 
mat_file_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes"
classifier = tf.keras.models.load_model(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Liver PNGs from matlab seperated\ClassificationModel")
new_dataset_outpath = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\refined LiTs database"
liver_seg_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\liver_seg"
item_seg_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\item_seg"
​
# create patient paths in new folder 
path_list = [os.path.join(new_dataset_outpath, "image_volumes"), os.path.join(new_dataset_outpath, "item_seg"), os.path.join(new_dataset_outpath, "liver_seg")]
for i in range(0,131):
    for path in path_list:
        new_path = os.path.join(path, str(i))
        if not os.path.exists(new_path):
            os.mkdir(new_path)
​
for patient in os.listdir(mat_file_path):
    if patient in ["9","86","87","88","89","90","91","92","93","94","97","95","96","98","99"]:
        print("Patient: ", patient)
        mat_path = os.path.join(mat_file_path, patient)
       
        keep_list = []
        num_files = len(os.listdir(mat_path)) + 1
​
        for mat in os.listdir(mat_path):
            
            #mimic the data flow of training for VB's model
            #load and convert (mat --> png --> numpy array)
​
            number = mat[:-4] #just get the number of the corresponding file
            mat_file = os.path.join(mat_path, mat)
            
            mat_array = np.array(loadmat(mat_file)['section']) #['section'] what does this do????
            mat_png = Image.fromarray(mat_array).convert('RGB')
            # #     COULD TRY TO INPUT AN IMAGE ?? (what exactly does the model do to data being inputted)
            back_to_array = np.asarray(mat_png)
            back_to_array = np.expand_dims(back_to_array, axis = 0) #also data type of uint-8 
            
            val = classifier.predict(back_to_array)
            print(val)
            #print(patient, number, val)
​
            if val[0] < float(.55): #from .5 
                #it seems to be classified as liver 
                keep_list.append(int(number))
        
        max_keep = max(keep_list)
        min_keep = min(keep_list)
        min_minus = int( min_keep - (.10 * num_files) )
        max_plus = int( max_keep + (.10 * num_files) )
    
        print("Range for {} : {} to {}. ".format(patient, min_minus, max_plus))
​
​
        # Case 1: max_plus too large, min_minus is good 
        if max_plus > num_files and min_minus >= 1:
            print("Case 1")
            for i in range(min_minus, num_files):
                # i = the number corresponding to file type
​
                #save item_seg 
                item_png = Image.open(os.path.join(item_seg_path, patient, str(str(i) + ".png")))
                item_png.save(os.path.join(new_dataset_outpath, "item_seg", patient, str(str(i) + ".png")))
​
                #save image_volume mat file
                i_mat = str(str(i) + ".mat")
                mp = os.path.join(mat_file_path, patient, i_mat)
                
                shutil.copy(mp, os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")))
​
                #save liver_seg png 
                liver_png = Image.open(os.path.join(liver_seg_path, patient, str(str(i) + ".png")))
                liver_png.save(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))
​
                print("Saved....{}".format(patient))
                print(os.path.join(item_seg_path, patient, str(str(i) + ".png") + "\n"))
                print(os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")) + "\n")
                print(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))
        
​
        #CASE 2: min_minus too small, max_plus good 
        if min_minus < 1 and max_plus <= num_files:
            print("Case 2")
            for i in range(1, max_plus):
                # i = the number corresponding to file type
​
                #save item_seg 
                item_png = Image.open(os.path.join(item_seg_path, patient, str(str(i) + ".png")))
                item_png.save(os.path.join(new_dataset_outpath, "item_seg", patient, str(str(i) + ".png")))
​
                #save image_volume mat file
                i_mat = str(str(i) + ".mat")
                mp = os.path.join(mat_file_path, patient, i_mat)
                
                shutil.copy(mp, os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")))
​
                #save liver_seg png 
                liver_png = Image.open(os.path.join(liver_seg_path, patient, str(str(i) + ".png")))
                liver_png.save(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))
                
                print("Saved....{}".format(patient))
                print("Saved....\n")
                print(os.path.join(item_seg_path, patient, str(str(i) + ".png") + "\n"))
                print(os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")) + "\n")
                print(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))
​
​
        # CASE 3: both outside of range
        if min_minus < 1 and max_plus > num_files:
            print("Case 3")
            for i in range(1, num_files):
                # i = the number corresponding to file type
​
                #save item_seg 
                item_png = Image.open(os.path.join(item_seg_path, patient, str(str(i) + ".png")))
                item_png.save(os.path.join(new_dataset_outpath, "item_seg", patient, str(str(i) + ".png")))
​
                #save image_volume mat file
                i_mat = str(str(i) + ".mat")
                mp = os.path.join(mat_file_path, patient, i_mat)
                
                shutil.copy(mp, os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")))
​
                #save liver_seg png 
                liver_png = Image.open(os.path.join(liver_seg_path, patient, str(str(i) + ".png")))
                liver_png.save(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))
                
                print("Saved....{}".format(patient))
                print("Saved....\n")
                print(os.path.join(item_seg_path, patient, str(str(i) + ".png") + "\n"))
                print(os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")) + "\n")
                print(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))
​
        if min_minus >= 1 and max_plus <= num_files:
            print("Case 4")
            for i in range(min_minus, max_plus):
​
                item_png = Image.open(os.path.join(item_seg_path, patient, str(str(i) + ".png")))
                item_png.save(os.path.join(new_dataset_outpath, "item_seg", patient, str(str(i) + ".png")))
​
                #save image_volume mat file
                i_mat = str(str(i) + ".mat")
                mp = os.path.join(mat_file_path, patient, i_mat)
                
                shutil.copy(mp, os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")))
​
                #save liver_seg png 
                liver_png = Image.open(os.path.join(liver_seg_path, patient, str(str(i) + ".png")))
                liver_png.save(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))
​
                print("Saved....{}".format(patient))
                print("Saved....\n")
                print(os.path.join(item_seg_path, patient, str(str(i) + ".png") + "\n"))
                print(os.path.join(new_dataset_outpath, "image_volumes", patient, str(str(i) + ".mat")) + "\n")
                print(os.path.join(new_dataset_outpath, "liver_seg", patient, str(str(i) + ".png")))