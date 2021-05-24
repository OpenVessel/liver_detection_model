import time 
import math
import tensorflow as tf
import slice_classification_v1 as sc
import os 
class LiverDetection:
    def __init__(self, config):
        self.config = config
        ## additional parameters

    def logSummary(self, phase, time_list):
        print("--- SUMMARY ({0}) ---".format(phase))
        for step in time_list:
            print("Step: ", step['name'])
            step_time = step['time']
            print("\nTime taken: {} seconds or {} minutes {}s to run\n".format(step_time, math.floor(step_time/60), step_time % 60))

        total_time = sum(time_list.map(lambda x: x['time']))
        print("\nTotal time taken: {} seconds or {} minutes {}s to run\n".format(total_time, math.floor(total_time/60), total_time % 60))

    
    def with_time(step):
        def wrapper(self, *args, **kwargs):
            # run step
            print('Running step: ' + step.__name__ + "\n")
            start_time = time.time()

            step_output = step(self, *args, **kwargs)

            print('\nDone step: '+ step.__name__)

            ## run time
            total_time = int(time.time() - start_time)
            self.time_list.append({'name': step.__name__, 'time' :total_time})
            
            floor_var = math.floor(total_time/60)
            mod_var = total_time % 60
            print("\nTime taken: {} seconds or {} minutes {}s to run\n".format(total_time, floor_var, mod_var))
            
            # reset tf graph for memory purposes
            tf.reset_default_graph()

            return step_output
        return wrapper

        
    def test(self, config):

        #classifier = tf.keras.models.load_model('classificationModel')
        
        train_img_path = os.path.join(config.outpath, 'Train')
        val_img_path = os.path.join(config.outpath, 'Validation')
        test_img_path = os.path.join(config.outpath, 'Test')
        saved_model_path = os.path.join(config.outpath, 'LD_Model')
        path_list = [train_img_path,val_img_path,
                    test_img_path, saved_model_path]
        
        
        sc.test(path_list)

        pass

    def train(self, config,  outpath):
        train_img_path = os.path.join(config.outpath, 'Train')
        val_img_path = os.path.join(config.outpath, 'Validation')
        test_img_path = os.path.join(config.outpath, 'Test')
        saved_model_path = os.path.join(config.outpath, 'LD_Model')
        path_list = [train_img_path, val_img_path, test_img_path, saved_model_path]
        sc.train_model(path_list)
        
        return
### call the model and any preprocess steps 
    def __repr__(self):
        return 