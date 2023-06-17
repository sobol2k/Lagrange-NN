import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from trainingdata import *
from model import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#np_config.enable_numpy_behavior()

plt.close('all')
plt.ioff()


def main():
   
    loaddata = True
    current_directory = os.path.join(os.path.dirname(__file__))
    data_path         = os.path.join(current_directory, "training_data")
    if loaddata:       
        x_train  = np.load(os.path.join(data_path , "x_train.npy"))
        xt_train = np.load(os.path.join(data_path , "xt_train.npy"))
    else:
        N         = 1500
        x0        = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)
        t         = np.arange(N, dtype=np.float32) # time steps 0 to N
        
        trainingdata  = TrainingDataDoublePendulum()
        data         = trainingdata.createdata(x0,t)
    
        trainingdata.savedata(data_path,data)

    'Create model'
    optimiser = keras.optimizers.Adam(learning_rate=1e-3)
    epochs = 10000
    model = nn_model(optimiser)
    model.train(x_train,xt_train,epochs)
    
if __name__ == "__main__":
    main()