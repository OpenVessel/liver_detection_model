

class liver_detection:
    def __init_(self, config):

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

### call the model and any preprocess steps 