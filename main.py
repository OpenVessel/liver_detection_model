import os
import tensorflow as tf
import time
import math

from class_calls.py import liver_detection

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
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    mat_file_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes"

    classifier = tf.keras.models.load_model('classificationModel')


    if cmdline.mode == "test":

        ##implement class call 
        liver_lesion.test(testing_volume)
    elif cmdline.mode == "train":
        liver_lesion.train(training_volume, validation_volume = testing_volume )
    else:
        raise BaseException('Invalid mode. Must be test or train')