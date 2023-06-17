import numpy as np
import os
from scipy.integrate import odeint
from helper import savedatatofile

class TrainingDataDoublePendulum:
    def __init__(self):
        pass
    
    def savedata(self,data_path,files):
        savedatatofile(data_path,files)
        
    def forward(self, state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
        
        if state.ndim == 1:
        
            t1, t2, w1, w2 = state
            a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
            a2 = (l1 / l2) * np.cos(t1 - t2)
            f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * np.sin(t1 - t2) - (g / l1) * np.sin(t1)
            f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
            g1 = (f1 - a1 * f2) / (1 - a1 * a2)
            g2 = (f2 - a2 * f1) / (1 - a1 * a2)
            return np.stack([w1, w2, g1, g2])
        
        elif state.ndim > 1:
            
            t1 = state[:,0]
            t2 = state[:,1]
            w1 = state[:,2]
            w2 = state[:,3]
            
            a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
            a2 = (l1 / l2) * np.cos(t1 - t2)
            f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * np.sin(t1 - t2) - (g / l1) * np.sin(t1)
            f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
            g1 = (f1 - a1 * f2) / (1 - a1 * a2)
            g2 = (f2 - a2 * f1) / (1 - a1 * a2)
            
            return np.stack([w1, w2, g1, g2]).T

    def createdata(self, x0, t, rtol = 1E-8, atol = 1E-8):
        """ Training data.

        In order to create training data, the analytical model is solved using odeint for N time steps.

        x_train  = [theta_1    , theta_2    , w_1    , w_2]    
                 = [theta_1    , theta_2    , dt_theta_1    , dt_theta_2]    
                 = [q_1    , q_2   , dt_q_1    , dt_q_2]
        xt_train = [dt_theta_1 , dt_theta_2 , dt_w_1 , dt_w_2] 
                 = [dt_theta_1 , dt_theta_2 , dt_dt_theta_1 , dt_dt_theta_2] 
                 = [dt_q_1 , dt_q_2, dt_dt_q_1 , dt_dt_q_2]
        
        """
        
        x_train = odeint(self.forward, x0, t=t, rtol=rtol, atol=atol)
        xt_train = self.forward(x_train)
        
        return {"x_train":x_train , "xt_train": xt_train}
