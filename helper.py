import os
import numpy as np

def savedatatofile(data_path,files):
    
    if not os.path.exists(data_path):
       os.makedirs(data_path)
    
    for filename, dataset in files.items():
        tmppath = data_path+"/"+filename
        print("Saving: {}".format(filename))
        np.save(tmppath, dataset)
