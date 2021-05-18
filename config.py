import os
import sys


class Config:
    '''
        Config class that contains all pathing
    '''

    def __init__(self):
        self.__database_root = 'LiTS_database'
        
        #self.__database_root = 'predict_database'

        self.root_folder = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(os.path.abspath(self.root_folder))

        self.database_root = os.path.join(self.root_folder, self.__database_root)
    
        ## the news weights are saved 
        #"D:\L_pipe\liver_open\liverseg-2017-nipsws\train_files\seg_liver\networks\train"
        self.images_volumes = 'images_volumes'
        self.item_seg = 'item_seg'
        self.liver_seg = 'liver_seg'
        self.debug = 0 # 0 for false, 1 for true
        self.phase = 'train' ## train or test


        self.labels = True
        self.fine_tune = 0

    def get_result_root(self, result_name):
        return os.path.join(self.root_folder, result_name)

    def get_crops_list_path(self):
        return os.path.join(self.root_folder, 'utils', 'crops_list', self.crops_list)

    def get_log(self, task_name):
        return os.path.join(self.root_folder, 'train_files', task_name, 'networks')