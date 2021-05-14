###
# Model to sort slices either liver or not liver
#
#
#
# 4/11/2021
# Vaibhav Gupta
###
## Img size is now 512,512 so that implementation into the rest of the model is easier
## The idea here is to call one function to other models as preprocess all functions defined here 
## will be called through def model_call_function():
############## import statements 
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import datetime

def test(test_imgs_path):
    # https://machinelearningspace.com/yolov3-tensorflow-2-part-4/

    #load trained model
    model = tf.keras.models.load_model(r'C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Liver PNGs from matlab seperated\ClassificationModel')
    class_names = {'Liver': 0, 'Non-Liver': 1}
    for i in os.listdir(test_imgs_path):
        
        img = image.load_img(test_imgs_path + '\\' + i, target_size = (512,512))
        plt.imshow(img)
        plt.show()
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis = 0)
        images = np.vstack([X])
        val = model.predict(images)
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(val)], 100 * np.max(val)))

def slice_classification(train_imgs_path, test_imgs_path, val_imgs_path, save_model_path):
    
    #images are extracted from the path or folder


    train = ImageDataGenerator( rescale = 1./255 ) 
    #print(type(train))                                     #HERE IS WHERE THE ISSUE MAY BE???
    validation = ImageDataGenerator( rescale = 1./255 )


    #print(type(validation)) 
    train_dataset = train.flow_from_directory(train_imgs_path,
                                            shuffle = True, 
                                            target_size = (512, 512),
                                            batch_size = 32,
                                            class_mode = 'binary',
                                            seed = 44)

    class_names = train_dataset.class_indices
    print(class_names)
    validation_dataset = train.flow_from_directory(val_imgs_path,
                                            shuffle = True,      
                                            target_size = (512, 512), #input shape here
                                            batch_size = 32,
                                            class_mode = 'binary',
                                            seed = 44)
    print(train_dataset) ## binary class 
    print(validation_dataset) # binary class 
    ## building the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (512, 512, 3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation = "relu"),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation = "relu"),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation = "relu"),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    model.save(save_model_path)
    ### record model BCE loss
    ## metric accuracy 
    model.compile(loss = 'binary_crossentropy',
                optimizer = RMSprop(lr = 1e-4),
                metrics = ['accuracy'])
    start = datetime.datetime.now()
    fitted_model = model.fit(train_dataset,
                        steps_per_epoch = 2,
                        epochs = 30,
                        validation_data = validation_dataset,
                        validation_freq = 1,
                        #callbacks = [tensorboard, early_stopping]
                            )
    end = datetime.datetime.now()
    elapsed = end - start
    print('\n ---------Elapsed Time-----------')
    print('Time to fit baseline model is:\n {}'.format(elapsed))
    print("-------Model Summary-------")
    model.summary()
    


def train_model(test_option = False):
    ## calls other functions in this script to be called into other scripts 
    train_imgs_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\LiverPNGsfrommatlabseperated\Train"
    val_imgs_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\LiverPNGsfrommatlabseperated\Validation"
    test_imgs_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\LiverPNGsfrommatlabseperated\Test\Liver"
    save_model_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\LiverPNGsfrommatlabseperated\ClassificationModel"
    slice_classification(train_imgs_path, test_imgs_path, val_imgs_path, save_model_path)
    if test_option == True:
        test(test_imgs_path)





