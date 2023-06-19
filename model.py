import keras 
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config

class nn_model:
    def __init__(self,dim,optimiser):

        self.dim = dim
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.dim,)),
                tf.keras.layers.Dense(128, activation="softplus"),
                tf.keras.layers.Dense(128, activation="softplus"),
                tf.keras.layers.Dense(1,)
            ]
            )
        
        self.optimiser = optimiser
        

    def train(self,x_train,xt_train,epochs,bathsize = 1):

        """  
        x_train  = [theta_1    , theta_2    , dt_theta_1    , dt_theta_2]    = [q_1    , q_2   , dt_q_1    , dt_q_2]
        xt_train = [dt_theta_1 , dt_theta_2 , dt_dt_theta_1 , dt_dt_theta_2] = [dt_q_1 , dt_q_2, dt_dt_q_1 , dt_dt_q_2]
        y_pred   = L !
        """        

        n = x_train.shape[0]*x_train.shape[1]
        alpha = 1E-4
       
        x_train  = tf.Variable(np.atleast_2d(x_train))
        xt_train = tf.constant(np.atleast_2d(xt_train))

        sl  = int(self.dim/2)
        idx = int(n/2)
        for epoch in range(epochs):
            
            ' Gradient and hessian '
            with tf.GradientTape(persistent = True) as g1:
                g1.watch(x_train)
                with tf.GradientTape(persistent = True) as g2:
                    g2.watch(x_train)
                    lagrangian = self.model(x_train, training=True)
                grad  = g2.gradient(lagrangian, x_train)
                
                hessian_qt_qt = g1.jacobian(grad, x_train)
                
                'Sub gradient'
                gradient_q = tf.reshape(grad[:,:sl],[idx,-1])
                
                'Reshape'
                h_mat = tf.reshape(hessian_qt_qt, [n, n])
                
                'Sub matrices'
                hessian_qt_qt = h_mat[:idx,:idx]
                hessian_q_qt  = h_mat[idx:,:idx]
                
                'Sub vector'
                qt = tf.reshape(x_train[:,sl:],[int(n/2),-1])
                
                'Calculate q_tt'
                reg = alpha * tf.eye(idx, dtype=tf.float64)
                qtt = tf.linalg.inv(reg+hessian_qt_qt) @ (gradient_q - hessian_q_qt@qt)
               
                'Calculate loss'
                xt_train         = tf.reshape(xt_train,[n,-1])
                predicted_values = tf.concat([qt, qtt],axis=0)
               
                'Minimise the loss with respect to the neural networks weights'
                loss  = tf.math.reduce_mean(tf.math.square(predicted_values - xt_train))
                gradients = g1.gradient(loss, self.model.trainable_weights)
                self.optimiser.apply_gradients(zip(gradients, self.model.trainable_weights))

            
    def predict(self):
        pass